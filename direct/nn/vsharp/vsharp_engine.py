# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast

from direct.config import BaseConfig
from direct.data import transforms as T
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device


class VSharpNet3DEngine(MRIModelEngine):
    """VSharpNet Engine."""

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
        """Inits :class:`VSharpNetEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
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

        self._spatial_dims = (3, 4)

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : Dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[Dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[Dict[str, Callable]]
            Callable regularization functions.

        Returns
        -------
        DoIterationOutput
            Contains outputs.
        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        data = dict_to_device(data, self.device)

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            output_images, output_kspace = self.forward_function(data)
            output_images = [T.modulus_if_complex(_, complex_axis=self._complex_dim) for _ in output_images]
            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}

            auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
            for i, output_image in enumerate(output_images):
                loss_dict = self.compute_loss_on_data(
                    loss_dict, loss_fns, data, output_image, None, auxiliary_loss_weights[i]
                )
            # Compute loss on k-space
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, None, output_kspace)

            loss = sum(loss_dict.values())  # type: ignore

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

        output_image = output_images[-1]
        return DoIterationOutput(
            output_image=output_image,
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict},
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])
        data = self.perform_sampling(data)
        output_images = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, height,  width, complex[=2])

        output_image = output_images[-1]
        output_kspace = data["masked_kspace"] + T.apply_mask(
            T.apply_padding(
                self.forward_operator(
                    T.expand_operator(output_image, data["sensitivity_map"], dim=self._coil_dim),
                    dim=self._spatial_dims,
                ),
                padding=data.get("padding", None),
            ),
            1 - data["sampling_mask"],
            return_mask=False,
        )

        return output_images, output_kspace


class VSharpNetEngine(MRIModelEngine):
    """VSharpNet Engine."""

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
        """Inits :class:`VSharpNetEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
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

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : Dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[Dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[Dict[str, Callable]]
            Callable regularization functions.

        Returns
        -------
        DoIterationOutput
            Contains outputs.
        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        data = dict_to_device(data, self.device)

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            output_images, output_kspace = self.forward_function(data)
            output_images = [T.modulus_if_complex(_, complex_axis=self._complex_dim) for _ in output_images]
            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}

            auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
            for i, output_image in enumerate(output_images):
                loss_dict = self.compute_loss_on_data(
                    loss_dict, loss_fns, data, output_image, None, auxiliary_loss_weights[i]
                )
            # Compute loss on k-space
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, None, output_kspace)

            loss = sum(loss_dict.values())  # type: ignore

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

        output_image = output_images[-1]
        return DoIterationOutput(
            output_image=output_image,
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict},
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])
        data = self.perform_sampling(data)

        output_images = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, height,  width, complex[=2])

        output_image = output_images[-1]
        output_kspace = data["masked_kspace"] + T.apply_mask(
            T.apply_padding(
                self.forward_operator(
                    T.expand_operator(output_image, data["sensitivity_map"], dim=self._coil_dim),
                    dim=self._spatial_dims,
                ),
                padding=data.get("padding", None),
            ),
            1 - data["sampling_mask"],
            return_mask=False,
        )

        return output_images, output_kspace
