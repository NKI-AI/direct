# Copyright (c) DIRECT Contributors

"""Engines for MEDL 2D and 3D models [1]_.

References
----------
.. [1] Qiao, X., Huang, Y., Li, W.: MEDL‐Net: A model‐based neural network for MRI reconstruction with enhanced deep
    learned regularizers. Magnetic Resonance in Med. 89, 2062–2075 (2023). https://doi.org/10.1002/mrm.29575
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
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device


class MEDL3DEngine(MRIModelEngine):
    """MEDL 3D Model Engine."""

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
        """Inits :class:`MEDL3DEngine`.

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

        loss_dict_reconstruction = {
            k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()
        }

        if "registration_model" in self.models:
            loss_dict_registration = {
                k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()
            }

        with autocast(enabled=self.mixed_precision):

            output_images, output_kspace = self.forward_function(data)
            output_images = [T.modulus_if_complex(_, complex_axis=self._complex_dim) for _ in output_images]

            auxiliary_loss_weights = torch.Tensor([1] + [0.1] * len(output_images)).to(output_images[0])
            # Compute reconstruction loss
            for i, output_image in enumerate(output_images):
                loss_dict_reconstruction = self.compute_loss_on_data(
                    loss_dict_reconstruction,
                    loss_fns,
                    data,
                    output_image=output_image,
                    output_kspace=None,
                    weight=auxiliary_loss_weights[i]
                    * (
                        1
                        if not "registration_model" in self.models
                        else self.cfg.additional_models.registration_model.rec_loss_factor
                    ),
                )

            # Compute reconstruction loss on k-space
            loss_dict_reconstruction = self.compute_loss_on_data(
                loss_dict_reconstruction, loss_fns, data, output_image=None, output_kspace=output_kspace
            )

            if "registration_model" in self.models:
                registered_image, displacement_field = self.do_registration(
                    data,
                    (
                        output_images[-1].detach()
                        if self.cfg.additional_models.registration_model.decoupled_training
                        else output_images[-1]
                    ),
                )

                # Registration loss
                shape = data["reference_image"].shape
                loss_dict_registration = self.compute_loss_on_data(
                    loss_dict_registration,
                    loss_fns,
                    data,
                    output_image=registered_image,
                    target_image=(
                        data["reference_image"]
                        if shape == registered_image.shape
                        else data["reference_image"].tile((1, registered_image.shape[1], *([1] * len(shape[1:]))))
                    ),
                    weight=self.cfg.additional_models.registration_model.reg_loss_factor,
                )

                if "displacement_field" in data:
                    target_displacement_field = data["displacement_field"]
                else:
                    target_displacement_field = None

                # Displacement field loss
                loss_dict_registration = self.compute_loss_on_data(
                    loss_dict_registration,
                    loss_fns,
                    data,
                    output_displacement_field=displacement_field,
                    target_displacement_field=target_displacement_field,
                    weight=self.cfg.additional_models.registration_model.reg_loss_factor,
                )

                loss_registration = sum(loss_dict_registration.values())  # type: ignore

            loss_reconstruction = sum(loss_dict_reconstruction.values())  # type: ignore

        if self.model.training:
            # Backpropagate registration loss only if registration model (if present) is DL-based
            if "registration_model" in self.models and len(list(self.models["registration_model"].parameters())) > 0:
                # If decoupled training freeze corresponding parameters
                if self.cfg.additional_models.registration_model.decoupled_training:
                    for param in self.models["registration_model"].parameters():
                        param.requires_grad = False  # Freeze registration model
                    # Reconstruction loss backward
                    self._scaler.scale(loss_reconstruction).backward()

                    if len(list(self.models["registration_model"].parameters())) > 0:
                        for param in self.models["registration_model"].parameters():
                            param.requires_grad = True  # Unfreeze registration model

                        # Freeze other models
                        for param in self.model.parameters():
                            param.requires_grad = False
                        for model in self.models:
                            if model != "registration_model":
                                for param in self.models[model].parameters():
                                    param.requires_grad = False
                        # Registation loss backward
                        self._scaler.scale(loss_registration).backward()

                        # Unfreeze all models
                        for param in self.model.parameters():
                            param.requires_grad = True
                        for model in self.models:
                            if model != "registration_model":
                                for param in self.models[model].parameters():
                                    param.requires_grad = True
                else:
                    # End-to-end training
                    self._scaler.scale(loss_reconstruction + loss_registration).backward()
            else:
                self._scaler.scale(loss_reconstruction).backward()

        # Detach loss dictionaries for logging
        loss_dict = {
            **(
                {f"registration_{k}": v for k, v in loss_dict_registration.items()}
                if "registration_model" in self.models
                else {}
            ),
            **{k: v for k, v in loss_dict_reconstruction.items()},
        }
        loss_dict = detach_dict(loss_dict)

        # if "masks" in data and not self.model.training:
        #     sampling_mask = torch.stack(data["masks"], -1)
        # else:
        sampling_mask = data["sampling_mask"]

        output_image = output_images[-1]
        return DoIterationOutput(
            output_image=(
                (output_image, registered_image, displacement_field)
                if "registration_model" in self.models
                else output_image
            ),
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=sampling_mask,
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


class MEDLEngine(MRIModelEngine):
    """MEDL 2D Model Engine."""

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
        """Inits :class:`MEDLEngine`.

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

            auxiliary_loss_weights = torch.Tensor([1] + [0.1] * len(output_images)).to(output_images[0])
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
