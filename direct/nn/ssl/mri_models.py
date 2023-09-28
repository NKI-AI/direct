# coding=utf-8
# Copyright (c) DIRECT Contributors

"""SSL MRI model engines of DIRECT."""
from abc import abstractmethod
from itertools import combinations
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

__all__ = ["SSDUMRIModelEngine", "DualSSLMRIModelEngine", "DualSSL2MRIModelEngine", "N2NMRIModelEngine"]


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

        if "input_sampling_mask" in data:  # ssdu
            first_input_sampling_mask = data["input_sampling_mask"][0][0]
            first_target_sampling_mask = data["target_sampling_mask"][0][0]
            storage.add_image("train/input_mask", first_input_sampling_mask[..., 0].unsqueeze(0))
            storage.add_image("train/target_mask", first_target_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = first_target_sampling_mask | first_input_sampling_mask
        elif "theta_sampling_mask" in data:  # dualssl
            first_theta_sampling_mask = data["theta_sampling_mask"][0][0]
            first_lambda_sampling_mask = data["lambda_sampling_mask"][0][0]
            storage.add_image("train/theta_mask", first_theta_sampling_mask[..., 0].unsqueeze(0))
            storage.add_image("train/lambda_mask", first_lambda_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = first_theta_sampling_mask | first_lambda_sampling_mask
        else:  # noisier2noise
            first_noisier_sampling_mask = data["noisier_sampling_mask"][0][0]
            storage.add_image("train/noisier_mask", first_noisier_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = data["sampling_mask"][0][0]
        first_target = data["target"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            slice_dim = -4
            num_slices = first_target.shape[slice_dim]
            first_target = first_target[num_slices // 2]
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

            output_image, output_kspace = self.forward_function(data)
            # In SSDU training we use output kspace to compute loss
            if self.model.training:
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
                # Some models output images so transform them to k-space
                if output_image is not None:
                    output_image = T.modulus(output_image)
                else:
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


class DualSSLMRIModelEngine(SSLMRIModelEngine):
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
        """Inits :class:`DualSSLMRIModelEngine`.

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
    def forward_function(
        self,
        data: Dict[str, Any],
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> Tuple[TensorOrNone, TensorOrNone]:
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

        loss_dicts = []

        data = dict_to_device(data, self.device)

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_image, output_kspace = self.forward_function(
                data, data["masked_kspace"], data["sampling_mask"]
            )  # (x_u, y_u)

            if self.model.training:
                if output_kspace is None:
                    output_kspace = self._forward_operator(
                        output_image, data["sensitivity_map"], data["sampling_mask"]
                    )
                else:
                    output_kspace = T.apply_mask(output_kspace, data["sampling_mask"], return_mask=False)

                output_image_theta, output_kspace_theta = self.forward_function(
                    data, data["theta_kspace"], data["theta_sampling_mask"]
                )  # (x_theta or None, y_theta or None)
                output_image_lambda, output_kspace_lambda = self.forward_function(
                    data, data["lambda_kspace"], data["lambda_sampling_mask"]
                )  # (x_lambda or None, y_lambda or None)

                if (output_kspace_theta is None) or (output_kspace_lambda is None):
                    output_kspace_theta, output_kspace_lambda = [
                        self._forward_operator(img, data["sensitivity_map"], data["sampling_mask"])
                        for img in (output_image_theta, output_image_lambda)
                    ]
                else:
                    output_kspace_theta, output_kspace_lambda = [
                        T.apply_mask(kspace, data["sampling_mask"], return_mask=False)
                        for kspace in (output_kspace_theta, output_kspace_lambda)
                    ]
                # (y_theta, y_lambda)

                output_kspaces = [output_kspace, output_kspace_theta, output_kspace_lambda]  # (y_u, y_theta, y_lambda)

                (output_image, output_image_theta, output_image_lambda) = [
                    T.modulus_if_complex(img) for img in (output_image, output_image_theta, output_image_lambda)
                ]

                loss_dict = {k: torch.tensor([0.0], dtype=output_image.dtype).to(self.device) for k in loss_fns.keys()}
                for key, value in loss_dict.items():
                    loss_fn = torch.tensor([0.0], dtype=output_image.dtype).to(self.device)
                    if "kspace" in key:
                        for source in output_kspaces:
                            loss_fn = loss_fn + loss_fns[key](
                                source,
                                data["masked_kspace"],
                                reduction="mean",
                            )
                    elif "grad" in key:
                        for source, target in combinations(
                            [data["target"], output_image_theta, output_image_lambda], r=2
                        ):
                            loss_fn = loss_fn + loss_fns[key](
                                source,
                                target,
                                reduction="mean",
                                reconstruction_size=data.get("reconstruction_size", None),
                            )
                    else:
                        for source, target in combinations(
                            [output_image, output_image_theta, output_image_lambda], r=2
                        ):
                            loss_fn = loss_fn + loss_fns[key](
                                source,
                                target,
                                reduction="mean",
                                reconstruction_size=data.get("reconstruction_size", None),
                            )
                    loss_dict[key] = value + loss_fn

                loss = sum(loss_dict.values())  # type: ignore
                self._scaler.scale(loss).backward()
                loss_dicts.append(detach_dict(loss_dict))  # Need to detach dict as this is only used for logging.
                # Add the loss dicts.
                loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum")

            else:
                if output_kspace is None:
                    output_kspace = self.forward_operator(
                        T.expand_operator(
                            output_image,
                            data["sensitivity_map"],
                            dim=self._coil_dim,
                        ),
                        dim=self._spatial_dims,
                    )
                output_kspace = data["masked_kspace"] + T.apply_padding(
                    T.apply_mask(output_kspace, ~data["sampling_mask"], return_mask=False), data["padding"]
                )
                output_image = T.root_sum_of_squares(
                    self.backward_operator(output_kspace, dim=self._spatial_dims), self._coil_dim
                )

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict} if self.model.training else {},
        )


class DualSSL2MRIModelEngine(SSLMRIModelEngine):
    r"""During training loss is computed on

    .. math::
        L\big(\mathcal{A}_{\theta}(x_{\lambda}), y_{\theta}\big)
        + L\big(\mathcal{A}_{\lambda}(x_{\theta}), y_{\lambda}\big)

    where :math:`x_{\sigma}=f_{\psi}(y_{\sigma}), \sigma=\theta \text{ or } \lambda` and
    :math:`y_{\theta} + y_{\lambda}=\tilde{y}` are splits of the original measured k-space :math:`\tilde{y}`
    via two (disjoint or not) sub-sampling operators :math:`y_{\theta}=U_{\theta}(\tilde{y})` and
    :math:`y_{\lambda}=U_{\lambda}(\tilde{y})` and :math:`U_{\theta} + U_{\lambda} = U`,
    where :math:`U` is the original sub-sampling operator.

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
        """Inits :class:`DualSSL2MRIModelEngine`."""
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
    def forward_function(
        self,
        data: Dict[str, Any],
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> Tuple[TensorOrNone, TensorOrNone]:
        """Must be implemented by child classes."""
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        # loss_fns can be done, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        loss_dicts = []
        output_image: Union[None, torch.Tensor]
        output_kspace: Union[None, torch.Tensor]
        output_image_theta: Union[None, torch.Tensor]
        output_kspace_theta: Union[None, torch.Tensor]
        output_image_lambda: Union[None, torch.Tensor]
        output_kspace_lambda: Union[None, torch.Tensor]

        data = dict_to_device(data, self.device)

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            if self.model.training:
                output_image_theta, output_kspace_theta = self.forward_function(
                    data, data["theta_kspace"], data["theta_sampling_mask"]
                )  # (x_theta, y_theta)
                output_image_lambda, output_kspace_lambda = self.forward_function(
                    data, data["lambda_kspace"], data["lambda_sampling_mask"]
                )  # (x_lambda, y_lambda)

                if (output_kspace_theta is None) or (output_kspace_lambda is None):
                    output_kspaces = [
                        self._forward_operator(img, data["sensitivity_map"], mask)
                        for (img, mask) in zip(
                            (output_image_theta, output_image_lambda),
                            (data["lambda_sampling_mask"], data["theta_sampling_mask"]),
                        )
                    ]
                else:
                    output_kspaces = [
                        T.apply_mask(kspace, mask, return_mask=False)
                        for (kspace, mask) in zip(
                            (output_kspace_theta, output_kspace_lambda),
                            (data["lambda_sampling_mask"], data["theta_sampling_mask"]),
                        )
                    ]
                # (y_theta->lambda, y_lambda->theta)
                output_images = [T.modulus_if_complex(output_image_lambda), T.modulus_if_complex(output_image_theta)]
                output_image = output_images[0]
            else:
                output_image, output_kspace = self.forward_function(data, data["masked_kspace"], data["sampling_mask"])
                if output_kspace is None:
                    output_kspace = self.forward_operator(
                        T.expand_operator(
                            output_image,
                            data["sensitivity_map"],
                            dim=self._coil_dim,
                        ),
                        dim=self._spatial_dims,
                    )
                output_kspace = data["masked_kspace"] + T.apply_padding(
                    T.apply_mask(output_kspace, ~data["sampling_mask"], return_mask=False),
                    data["padding"],
                )
                output_image = T.root_sum_of_squares(
                    self.backward_operator(output_kspace, dim=self._spatial_dims), self._coil_dim
                )

            if self.model.training:
                loss_dict = {
                    k: torch.tensor([0.0], dtype=data["sensitivity_map"].dtype).to(self.device)
                    for k in loss_fns.keys()
                }
                for key, value in loss_dict.items():
                    loss = torch.tensor([0.0], dtype=data["sensitivity_map"].dtype).to(self.device)

                    if "kspace" in key:
                        sources = output_kspaces
                        targets = [data["lambda_kspace"], data["theta_kspace"]]
                        recon_size = None
                    else:
                        sources = [output_images[0]]
                        targets = [output_images[1]]
                        recon_size = data.get("reconstruction_size", None)
                    for source, target in zip(sources, targets):
                        loss = loss + loss_fns[key](source, target, reduction="mean", reconstruction_size=recon_size)
                    loss_dict[key] = value + loss

                loss = sum(loss_dict.values())  # type: ignore
                self._scaler.scale(loss).backward()

                loss_dicts.append(detach_dict(loss_dict))  # Need to detach dict as this is only used for logging.
                # Add the loss dicts.
                loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum")

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict} if self.model.training else {},
        )


class N2NMRIModelEngine(SSLMRIModelEngine):
    """Noisier to Noise SSL MRI Engine."""

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
        """Inits :class:`N2NMRIModelEngine`.

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
        """Must be implemented by child classes.

        Must use `noisier_kspace` and `noisier_sampling_mask` if not training, else `masked_kspace` and `sampling_mask`.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = {}

        loss_dicts = []

        data = dict_to_device(data, self.device)

        output_image: Union[None, torch.Tensor]
        output_kspace: Union[None, torch.Tensor]

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_image, output_kspace = self.forward_function(data)

            if self.model.training:
                if output_kspace is None:
                    output_kspace = self._forward_operator(
                        output_image, data["sensitivity_map"], data["sampling_mask"]
                    )
                else:
                    output_kspace = T.apply_mask(output_kspace, data["sampling_mask"], return_mask=False)
            else:
                if output_kspace is None:
                    output_kspace = self.forward_operator(
                        T.expand_operator(
                            output_image,
                            data["sensitivity_map"],
                            dim=self._coil_dim,
                        ),
                        dim=self._spatial_dims,
                    )
                output_kspace = data["masked_kspace"] + T.apply_padding(
                    T.apply_mask(output_kspace, ~data["sampling_mask"], return_mask=False),
                    data["padding"],
                )

            output_image = T.root_sum_of_squares(
                self.backward_operator(output_kspace, dim=self._spatial_dims), self._coil_dim
            )
            if self.model.training:
                loss_dict = {k: torch.tensor([0.0], dtype=output_image.dtype).to(self.device) for k in loss_fns.keys()}

                loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, output_kspace)

                loss = sum(loss_dict.values())  # type: ignore
                self._scaler.scale(loss).backward()

                loss_dicts.append(detach_dict(loss_dict))  # Need to detach dict as this is only used for logging.
                # Add the loss dicts.
                loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum")

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict} if self.model.training else {},
        )
