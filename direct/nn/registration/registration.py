"""Registration models for direct registration."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.registration.demons import DemonsFilterType, multiscale_demons_displacement
from direct.registration.optical_flow import OpticalFlowEstimatorType, optical_flow_displacement
from direct.registration.registration import DISCPLACEMENT_FIELD_2D_DIMENSIONS
from direct.registration.warp import warp

__all__ = [
    "OpticalFlowILKRegistration2dModel",
    "OpticalFlowTVL1Registration2dModel",
    "DemonsRegistration2dModel",
    "UnetRegistration2dModel",
]


class ClassicalRegistration2dModel(nn.Module):

    def __init__(
        self,
        displacement_transform: Callable,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.displacement_transform = displacement_transform
        self.warp_num_integration_steps = warp_num_integration_steps

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`UnetRegistrationModel`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Moving image tensor of shape (batch_size, seq_len, height, width).
        reference_image : torch.Tensor
            Reference image tensor of shape (batch_size, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the warped image tensor of shape (batch_size, seq_len, height, width)
            and the displacement field tensor of shape (batch_size, seq_len, 2, height, width).
        """
        batch_size, seq_len, height, width = moving_image.shape

        device = reference_image.device

        # Estimate the displacement field
        displacement_field = [
            self.displacement_transform(reference_image[_].cpu(), moving_image[_].cpu())
            for _ in range(moving_image.shape[0])
        ]
        displacement_field = torch.stack(displacement_field, dim=0)
        displacement_field = displacement_field.to(device).reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )

        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(moving_image, displacement_field, num_integration_steps=self.warp_num_integration_steps)

        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class OpticalFlowRegistration2dModel(ClassicalRegistration2dModel):

    def __init__(
        self,
        estimator_type: OpticalFlowEstimatorType,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            displacement_transform=partial(
                optical_flow_displacement,
                estimator_type=estimator_type,
                **kwargs,
            ),
            warp_num_integration_steps=warp_num_integration_steps,
        )

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`UnetRegistrationModel`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Moving image tensor of shape (batch_size, seq_len, height, width).
        reference_image : torch.Tensor
            Reference image tensor of shape (batch_size, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the warped image tensor of shape (batch_size, seq_len, height, width)
            and the displacement field tensor of shape (batch_size, seq_len, 2, height, width).
        """
        batch_size, seq_len, height, width = moving_image.shape

        device = reference_image.device

        # Estimate the displacement field
        displacement_field = [
            self.displacement_transform(reference_image[_].detach().cpu(), moving_image[_].detach().cpu())
            for _ in range(moving_image.shape[0])
        ]
        displacement_field = torch.stack(displacement_field, dim=0)
        displacement_field = displacement_field.to(device).reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )

        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(moving_image, displacement_field, num_integration_steps=self.warp_num_integration_steps)

        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class OpticalFlowILKRegistration2dModel(OpticalFlowRegistration2dModel):

    def __init__(
        self,
        radius: int = 7,
        num_warp: int = 10,
        gaussian: bool = False,
        prefilter: bool = True,
        warp_num_integration_steps: int = 1,
    ) -> None:
        super().__init__(
            estimator_type=OpticalFlowEstimatorType.ILK,
            warp_num_integration_steps=warp_num_integration_steps,
            radius=radius,
            num_warp=num_warp,
            gaussian=gaussian,
            prefilter=prefilter,
        )


class OpticalFlowTVL1Registration2dModel(OpticalFlowRegistration2dModel):

    def __init__(
        self,
        attachment: float = 15,
        tightness: float = 0.3,
        num_warp: int = 5,
        num_iter: int = 10,
        tol: float = 1e-3,
        prefilter: bool = True,
        warp_num_integration_steps: int = 1,
    ) -> None:
        super().__init__(
            estimator_type=OpticalFlowEstimatorType.TV_L1,
            warp_num_integration_steps=warp_num_integration_steps,
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
            tol=tol,
            prefilter=prefilter,
        )


class DemonsRegistration2dModel(ClassicalRegistration2dModel):

    def __init__(
        self,
        demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
        demons_num_iterations: int = 50,
        demons_smooth_displacement_field: bool = True,
        demons_standard_deviations: float = 1.0,
        demons_intensity_difference_threshold: float | None = None,
        demons_maximum_rms_error: float | None = None,
        warp_num_integration_steps: int = 1,
    ) -> None:
        """Inits :class:`DemonsRegistration2dModel`.

        Parameters
        ----------
        demons_filter_type : DemonsFilterType, optional
            Type of the Demons filter (DemonsFilterType.DEMONS, DemonsFilterType.FAST_SYMMETRIC_FORCES,
            DemonsFilterType.SYMMETRIC_FORCES, DemonsFilterType.DIFFEOMORPHIC). Default: DemonsFilterType.SYMMETRIC_FORCES.
        demons_num_iterations : int
            Number of iterations for the Demons filter. Default: 100.
        demons_smooth_displacement_field : bool
            Whether to smooth the displacement field. Default: True.
        demons_standard_deviations : float
            Standard deviations for Gaussian smoothing. Default: 1.5.
        demons_intensity_difference_threshold : float, optional
            Intensity difference threshold. Default: None.
        demons_maximum_rms_error : float, optional
            Maximum RMS error. Default: None.
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """

        super().__init__(
            displacement_transform=partial(
                multiscale_demons_displacement,
                filter_type=demons_filter_type,
                num_iterations=demons_num_iterations,
                smooth_displacement_field=demons_smooth_displacement_field,
                standard_deviations=demons_standard_deviations,
                intensity_difference_threshold=demons_intensity_difference_threshold,
                maximum_rms_error=demons_maximum_rms_error,
            ),
            warp_num_integration_steps=warp_num_integration_steps,
        )

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`UnetRegistrationModel`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Moving image tensor of shape (batch_size, seq_len, height, width).
        reference_image : torch.Tensor
            Reference image tensor of shape (batch_size, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the warped image tensor of shape (batch_size, seq_len, height, width)
            and the displacement field tensor of shape (batch_size, seq_len, 2, height, width).
        """
        batch_size, seq_len, height, width = moving_image.shape

        device = reference_image.device

        # Estimate the displacement field
        displacement_field = [
            self.displacement_transform(reference_image[_].detach().cpu(), moving_image[_].detach().cpu())
            for _ in range(moving_image.shape[0])
        ]
        displacement_field = torch.stack(displacement_field, dim=0)
        displacement_field = displacement_field.to(device).reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )

        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(moving_image, displacement_field, num_integration_steps=self.warp_num_integration_steps)

        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class UnetRegistration2dModel(nn.Module):

    def __init__(
        self,
        max_seq_len: int,
        unet_num_filters: int = 16,
        unet_num_pool_layers: int = 4,
        unet_dropout_probability: float = 0.0,
        unet_normalized: bool = False,
        warp_num_integration_steps: int = 1,
    ) -> None:
        """Inits :class:`UnetRegistration2dModel`.

        Parameters
        ----------
        max_seq_len : int
            Maximum sequence length expected in the moving image.
        unet_num_filters : int
            Number of filters in the first layer of the UNet. Default: 16.
        unet_num_pool_layers : int
            Number of pooling layers in the UNet. Default: 4.
        unet_dropout_probability : float
            Dropout probability. Default: 0.0.
        unet_normalized : bool
            Whether to use normalization in the UNet. Default: False.
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """
        super().__init__()

        self.max_seq_len = max_seq_len

        self.model = (UnetModel2d if not unet_normalized else NormUnetModel2d)(
            in_channels=max_seq_len + 1,
            out_channels=max_seq_len * DISCPLACEMENT_FIELD_2D_DIMENSIONS,
            num_filters=unet_num_filters,
            num_pool_layers=unet_num_pool_layers,
            dropout_probability=unet_dropout_probability,
        )
        self.warp_num_integration_steps = warp_num_integration_steps

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`UnetRegistration2dModel`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Moving image tensor of shape (batch_size, seq_len, height, width).
        reference_image : torch.Tensor
            Reference image tensor of shape (batch_size, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the warped image tensor of shape (batch_size, seq_len, height, width)
            and the displacement field tensor of shape (batch_size, seq_len, 2, height, width).
        """
        batch_size, seq_len, height, width = moving_image.shape

        # Pad the moving image to the maximum sequence length
        x = nn.functional.pad(moving_image, (0, 0, 0, 0, 0, self.max_seq_len - moving_image.shape[1]))
        # Add the reference image as the first channel
        x = torch.cat((reference_image.unsqueeze(1), x), dim=1)

        # Forward pass through the model
        displacement_field = self.model(x)

        # Model outputs the displacement field for each time step with 2 channels (x and y displacements)
        displacement_field = displacement_field.reshape(
            batch_size, self.max_seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )  # (batch_size, max_seq_len, 2, height, width)

        # Crop the displacement field to the actual sequence length
        displacement_field = displacement_field[:, :seq_len]  # (batch_size, seq_len, 2, height, width)

        # Reshape the displacement field and moving image to be compatible with the warp module
        displacement_field = displacement_field.reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )
        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(moving_image, displacement_field, num_integration_steps=self.warp_num_integration_steps)
        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )
