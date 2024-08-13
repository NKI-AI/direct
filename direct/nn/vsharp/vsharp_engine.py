# Copyright (c) DIRECT Contributors

"""Engines for vSHARP 2D and 3D models [1].

Includes supervised, self-supervised and joint supervised and self-supervised learning [2] engines.

References
----------
.. [1] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
    Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
    https://doi.org/10.48550/arXiv.2311.15856.
.. [2] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and 
    Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023). 
    https://doi.org/10.48550/arXiv.2311.15856.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.cuda.amp import autocast

from direct.config import BaseConfig
from direct.data import transforms as T
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import JSSLMRIModelEngine, SSLMRIModelEngine
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device


class VSharpNet3DEngine(MRIModelEngine):
    """VSharpNet 3D Model Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`VSharpNet3DEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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
        data: dict[str, Any],
        loss_fns: Optional[dict[str, callable]] = None,
        regularizer_fns: Optional[dict[str, callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[dict[str, callable]]
            callable loss functions.
        regularizer_fns : Optional[dict[str, callable]]
            callable regularization functions.

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

            if "registration_model" in self.models:
                # Perform registration and compute loss on registered image and displacement field
                registered_image, displacement_field = self.do_registration(data, output_images[-1])

                shape = data["reference_image"].shape
                loss_dict = self.compute_loss_on_data(
                    loss_dict,
                    loss_fns,
                    data,
                    output_image=registered_image,
                    target_image=(
                        data["reference_image"]
                        if shape == registered_image.shape
                        else data["reference_image"].tile((1, registered_image.shape[1], *([1] * len(shape[1:]))))
                    ),
                )
                loss_dict = self.compute_loss_on_data(
                    loss_dict,
                    loss_fns,
                    data,
                    output_displacement_field=displacement_field,
                    target_displacement_field=data["displacement_field"],
                )

            auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
            for i, output_image in enumerate(output_images):
                loss_dict = self.compute_loss_on_data(
                    loss_dict,
                    loss_fns,
                    data,
                    output_image=output_image,
                    output_kspace=None,
                    weight=auxiliary_loss_weights[i],
                )
            # Compute loss on k-space
            loss_dict = self.compute_loss_on_data(
                loss_dict, loss_fns, data, output_image=None, output_kspace=output_kspace
            )

            loss = sum(loss_dict.values())  # type: ignore

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

        output_image = output_images[-1]
        return DoIterationOutput(
            output_image=(output_image, registered_image) if "registration_model" in self.models else output_image,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
            data_dict={**loss_dict},
        )

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, None]:
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
            ~data["sampling_mask"],
            return_mask=False,
        )

        return output_images, output_kspace


class VSharpNetEngine(MRIModelEngine):
    """VSharpNet 2D Model Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ) -> None:
        """Inits :class:`VSharpNetEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models for secondary tasks, such as sensitivity map estimation model.
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
        data: dict[str, Any],
        loss_fns: Optional[dict[str, callable]] = None,
        regularizer_fns: Optional[dict[str, callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[dict[str, callable]]
            callable loss functions.
        regularizer_fns : Optional[dict[str, callable]]
            callable regularization functions.

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
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
            data_dict={**loss_dict},
        )

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

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
            ~data["sampling_mask"],
            return_mask=False,
        )

        return output_images, output_kspace


class VSharpNetSSLEngine(SSLMRIModelEngine):
    """Self-supervised Learning vSHARP Model 2D Engine.

    Used for the main experiments for SSL in the JSSL paper [1].

    Parameters
    ----------
    cfg: BaseConfig
        Configuration file.
    model: nn.Module
        Model.
    device: str
        Device. Can be "cuda:{idx}" or "cpu".
    forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The forward operator. Default: None.
    backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The backward operator. Default: None.
    mixed_precision: bool
        Use mixed precision. Default: False.
    **models: nn.Module
        Additional models.


    References
    ----------
    .. [1] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
        Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
        https://doi.org/10.48550/arXiv.2311.15856.
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
        """Inits :class:`VSharpNetSSLEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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

    def forward_function(self, data: dict[str, Any]) -> None:
        """Forward function for :class:`VSharpNetSSLEngine`."""
        raise NotImplementedError(
            "Forward function for SSL vSHARP engine is not implemented. `VSharpNetSSLEngine` "
            "implements the `_do_iteration` method itself so the forward function should not be "
            "called."
        )

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: Optional[dict[str, Callable]] = None,
        regularizer_fns: Optional[dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """This function implements the `_do_iteration` for the SSL vSHARP model.

        Returns
        -------
        DoIterationOutput
            Output of the iteration.


        It assumes different behavior for training and inference. During training, it expects the input data
        to contain keys "input_kspace" and "input_sampling_mask", otherwise, it expects the input data to contain
        keys "masked_kspace" and "sampling_mask".

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary. The dictionary should contain the following keys:
            - "input_kspace" if training, otherwise "masked_kspace".
            - "input_sampling_mask" if training, otherwise "sampling_mask".
            - "target_sampling_mask": Sampling mask for the target k-space if training.
            - "sensitivity_map": Sensitivity map.
            - "target": Target image.
            - "padding": Padding, optionally.
        loss_fns : Optional[dict[str, Callable]], optional
            Loss functions, optional.
        regularizer_fns : Optional[dict[str, Callable]], optional
            Regularizer functions, optional.

        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        # Move data to device
        data = dict_to_device(data, self.device)

        # Get the k-space and mask which differ during training and inference for SSL
        if self.model.training:
            kspace, mask = data["input_kspace"], data["input_sampling_mask"]
        else:
            kspace, mask = data["masked_kspace"], data["sampling_mask"]

        # Initialize loss and regularizer dictionaries
        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
        regularizer_dict = {
            k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
        }

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_images = self.model(
                masked_kspace=kspace,
                sampling_mask=mask,
                sensitivity_map=data["sensitivity_map"],
            )

            if self.model.training:
                if len(output_images) > 1:
                    # Initialize auxiliary loss weights with a logarithmic scale if multiple auxiliary steps
                    auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
                else:
                    # Initialize auxiliary loss weights with a single value of 1.0 if single step
                    auxiliary_loss_weights = torch.ones(1).to(output_images[0])

                for i, output_image in enumerate(output_images):
                    # Data consistency
                    output_kspace = T.apply_padding(
                        kspace + self._forward_operator(output_image, data["sensitivity_map"], ~mask),
                        padding=data.get("padding", None),
                    )
                    # Project predicted k-space onto target k-space if SSL
                    output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)

                    # Compute k-space loss per auxiliary step
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, None, output_kspace, auxiliary_loss_weights[i]
                    )
                    regularizer_dict = self.compute_loss_on_data(
                        regularizer_dict, regularizer_fns, data, None, output_kspace, auxiliary_loss_weights[i]
                    )

                    # SENSE reconstruction
                    output_images[i] = T.modulus(
                        T.reduce_operator(
                            self.backward_operator(output_kspace, dim=self._spatial_dims),
                            data["sensitivity_map"],
                            self._coil_dim,
                        )
                    )

                    # Compute image loss per auxiliary step
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                    )
                    regularizer_dict = self.compute_loss_on_data(
                        regularizer_dict, regularizer_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                    )

                loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore
                self._scaler.scale(loss).backward()

                output_image = output_images[-1]
            else:
                output_kspace = T.apply_padding(
                    kspace + self._forward_operator(output_images[-1], data["sensitivity_map"], ~mask),
                    padding=data.get("padding", None),
                )
                # SENSE reconstruction using data consistent k-space
                output_image = T.modulus(
                    T.reduce_operator(
                        self.backward_operator(output_kspace, dim=self._spatial_dims),
                        data["sensitivity_map"],
                        self._coil_dim,
                    )
                )

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.
        regularizer_dict = detach_dict(regularizer_dict)

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
            data_dict={**loss_dict, **regularizer_dict},
        )


class VSharpNetJSSLEngine(JSSLMRIModelEngine):
    """Joint Supervised and Self-supervised Learning vSHARP Model 2D Engine.

    Used for the main experiments in the JSSL paper [1].

    Parameters
    ----------
    cfg: BaseConfig
        Configuration file.
    model: nn.Module
        Model.
    device: str
        Device. Can be "cuda:{idx}" or "cpu".
    forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The forward operator. Default: None.
    backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The backward operator. Default: None.
    mixed_precision: bool
        Use mixed precision. Default: False.
    **models: nn.Module
        Additional models.


    References
    ----------
    .. [1] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
        Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
        https://doi.org/10.48550/arXiv.2311.15856.
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
        """Inits :class:`VSharpNetJSSLEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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

    def forward_function(self, data: dict[str, Any]) -> None:
        """Forward function for :class:`VSharpNetJSSLEngine`."""
        raise NotImplementedError(
            "Forward function for JSSL vSHARP is not implemented. `VSharpNetJSSLEngine` "
            "implements the `_do_iteration` method itself so the forward function should not be "
            "called."
        )

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: Optional[dict[str, Callable]] = None,
        regularizer_fns: Optional[dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """This function implements the `_do_iteration` for the JSSL vSHARP model.

        Returns
        -------
        DoIterationOutput
            Output of the iteration.


        It assumes different behavior for SSL training and inference. During SSL training, it expects the input data
        to contain keys "input_kspace" and "input_sampling_mask", otherwise, it expects the input data to contain
        keys "masked_kspace" and "sampling_mask".

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary. The dictionary should contain the following keys:
            - "is_ssl": Boolean indicating if the sample is for SSL training.
            - "input_kspace" if SSL training, otherwise "masked_kspace".
            - "input_sampling_mask" if SSL training, otherwise "sampling_mask".
            - "target_sampling_mask": Sampling mask for the target k-space if SSL training.
            - "sensitivity_map": Sensitivity map.
            - "target": Target image.
            - "padding": Padding, optionally.
        loss_fns : Optional[dict[str, Callable]], optional
            Loss functions, optional.
        regularizer_fns : Optional[dict[str, Callable]], optional
            Regularizer functions, optional.

        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        # Move data to device
        data = dict_to_device(data, self.device)

        # Get a boolean indicating if the sample is for SSL training
        # This will expect the input data to contain the keys "input_kspace" and "input_sampling_mask" if SSL training
        is_ssl = data["is_ssl"][0]

        # Get the k-space and mask which differ if SSL training or supervised training
        # The also differ during training and inference for SSL
        if is_ssl and self.model.training:
            kspace, mask = data["input_kspace"], data["input_sampling_mask"]
        else:
            kspace, mask = data["masked_kspace"], data["sampling_mask"]

        # Initialize loss and regularizer dictionaries
        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
        regularizer_dict = {
            k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
        }

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_images = self.model(
                masked_kspace=kspace,
                sampling_mask=mask,
                sensitivity_map=data["sensitivity_map"],
            )

            if self.model.training:
                if len(output_images) > 1:
                    # Initialize auxiliary loss weights with a logarithmic scale if multiple auxiliary steps
                    auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
                else:
                    # Initialize auxiliary loss weights with a single value of 1.0 if single step
                    auxiliary_loss_weights = torch.ones(1).to(output_images[0])

                for i, output_image in enumerate(output_images):
                    # Data consistency
                    output_kspace = T.apply_padding(
                        kspace + self._forward_operator(output_image, data["sensitivity_map"], ~mask),
                        padding=data.get("padding", None),
                    )
                    if is_ssl:
                        # Project predicted k-space onto target k-space if SSL
                        output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)

                    # Compute k-space loss per auxiliary step
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, None, output_kspace, auxiliary_loss_weights[i]
                    )
                    regularizer_dict = self.compute_loss_on_data(
                        regularizer_dict, regularizer_fns, data, None, output_kspace, auxiliary_loss_weights[i]
                    )

                    # SENSE reconstruction if SSL else modulus if supervised
                    output_images[i] = T.modulus(
                        T.reduce_operator(
                            self.backward_operator(output_kspace, dim=self._spatial_dims),
                            data["sensitivity_map"],
                            self._coil_dim,
                        )
                        if is_ssl
                        else output_images[i]
                    )

                    # Compute image loss per auxiliary step
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                    )
                    regularizer_dict = self.compute_loss_on_data(
                        regularizer_dict, regularizer_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                    )

                loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore
                self._scaler.scale(loss).backward()

                output_image = output_images[-1]
            else:
                output_image = T.modulus(output_images[-1])

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.
        regularizer_dict = detach_dict(regularizer_dict)

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
            data_dict={**loss_dict, **regularizer_dict},
        )
