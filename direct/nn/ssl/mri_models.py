# coding=utf-8
# Copyright (c) DIRECT Contributors

"""SSL MRI model engines of DIRECT."""
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import autocast

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device, normalize_image, reduce_list_of_dicts
from direct.utils.events import get_event_storage

__all__ = ["SSDUMRIModelEngine"]


class SSLMRIModelEngine(MRIModelEngine):
    """Base Engine for SSL MRI models."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`SSLMRIModelEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models.
        """
        super().__init__(
            cfg=cfg,
            model=model,
            device=device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def log_first_training_example_and_model(self, data):
        storage = get_event_storage()
        self.logger.info(f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}.")

        first_input_sampling_mask = data["input_sampling_mask"][0][0]
        first_target_sampling_mask = data["target_sampling_mask"][0][0]
        storage.add_image("train/input_mask", first_input_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image("train/target_mask", first_target_sampling_mask[..., 0].unsqueeze(0))
        first_sampling_mask = first_target_sampling_mask | first_input_sampling_mask

        first_target = data["target"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            num_slices = first_target.shape[0]
            first_target = first_target[: num_slices // 2]
            first_target = torch.cat([first_target[_] for _ in range(first_target.shape[0])], dim=-1)
        elif self.ndim > 3:
            raise NotImplementedError

        storage.add_image("train/mask", first_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image(
            "train/target",
            normalize_image(first_target.unsqueeze(0)),
        )

        if "initial_image" in data:
            storage.add_image(
                "train/initial_image",
                normalize_image(T.modulus(data["initial_image"][0]).unsqueeze(0)),
            )

        self.write_to_logs()


class SSDUMRIModelEngine(SSLMRIModelEngine):
    r"""During training loss is computed :math:`L\big(\mathcal{A}_{\text{tar}}(x_{\text{out}}), y_{\text{tar}}\big)`

    where :math:`x_{\text{out}}=f_{\theta}(y_{\text{inp}})` and :math:`y_{\text{inp}} + y_{\text{tar}}=\tilde{y}`
    are splits of the original measured k-space :math:`\tilde{y}` via two (disjoint or not) sub-sampling operators
    :math:`y_{\text{inp}}=U_{\text{inp}}(\tilde{y})` and :math:`y_{\text{tar}}=U_{\text{tar}}(\tilde{y})` and
    :math:`U_{\text{inp}} + U_{\text{tar}} = U`, where :math:`U` is the original sub-sampling operator.

    During inference, output is computed as :math:`(\mathbb{1} - U)f_{\theta}(\tilde{y}) + \tilde{y}`.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`SSDUMRIModelEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models.
        """
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    @abstractmethod
    def forward_function(self, data: Dict[str, Any]) -> Tuple[TensorOrNone, TensorOrNone]:
        """Must be implemented by child classes."""
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = {}
        if regularizer_fns is None:
            regularizer_fns = {}

        loss_dicts, regularizer_dicts = [], []

        data = dict_to_device(data, self.device)

        output_image: Union[None, torch.Tensor]
        output_kspace: Union[None, torch.Tensor]

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])
            if self.model.training:
                kspace, mask = data["input_kspace"], data["input_sampling_mask"]
            else:
                kspace, mask = data["masked_kspace"], data["sampling_mask"]

            output_image, output_kspace = self.forward_function(data)

            if self.model.training:
                if output_kspace is None:
                    output_kspace = self.forward_operator(
                        T.expand_operator(output_image, data["sensitivity_map"], dim=self._coil_dim),
                        dim=self._spatial_dims,
                    )
                # Data consistency
                output_kspace = kspace + T.apply_mask(output_kspace, ~mask, return_mask=False)
                # Apply padding if it exists
                output_kspace = T.apply_padding(output_kspace, padding=data.get("padding", None))
                # Project to target k-space
                output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)
                # SENSE reconstruction
                output_image = T.modulus(
                    T.reduce_operator(
                        self.backward_operator(output_kspace, dim=self._spatial_dims),
                        data["sensitivity_map"],
                        self._coil_dim,
                    )
                )
            else:
                if output_image is not None:
                    output_image = T.modulus(output_image)
                else:
                    # Data consistency
                    output_kspace = kspace + T.apply_mask(output_kspace, ~mask, return_mask=False)
                    # Apply padding if it exists
                    output_kspace = T.apply_padding(output_kspace, padding=data.get("padding", None))
                    # SENSE reconstruction
                    output_image = T.modulus(
                        T.reduce_operator(
                            self.backward_operator(output_kspace, dim=self._spatial_dims),
                            data["sensitivity_map"],
                            self._coil_dim,
                        )
                    )

            if self.model.training:
                loss_dict = {k: torch.tensor([0.0], dtype=output_image.dtype).to(self.device) for k in loss_fns.keys()}
                regularizer_dict = {
                    k: torch.tensor([0.0], dtype=output_image.dtype).to(self.device) for k in regularizer_fns.keys()
                }

                loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, output_kspace)
                regularizer_dict = self.compute_loss_on_data(
                    regularizer_dict, regularizer_fns, data, output_image, output_kspace
                )

                loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore
                self._scaler.scale(loss).backward()

                loss_dicts.append(detach_dict(loss_dict))
                regularizer_dicts.append(
                    detach_dict(regularizer_dict)
                )  # Need to detach dict as this is only used for logging.
                # Add the loss dicts.
                loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum")
                regularizer_dict = reduce_list_of_dicts(regularizer_dicts, mode="sum")

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict} if self.model.training else {},
        )
