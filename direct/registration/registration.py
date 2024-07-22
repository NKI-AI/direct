"""Direct registration module for estimating displacement fields between a reference and a moving image."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

import torch

from direct.registration.demons import DemonsFilterType, multiscale_demons_displacement
from direct.registration.warp import warp_tensor
from direct.types import DirectEnum, TransformKey
from direct.utils import DirectModule

__all__ = ["DisplacementModule", "DisplacementTransformType", "WarpModule"]

DISCPLACEMENT_FIELD_2D_DIMENSIONS = 2
DISCPLACEMENT_FIELD_3D_DIMENSIONS = 3


class DisplacementTransformType(DirectEnum):
    MULTISCALE_DEMONS = "multiscale_demons"
    OPTICAL_FLOW = "optical_flow"


class DisplacementModule(DirectModule):
    """Module for estimating displacement fields between a reference and a moving image.

    Parameters
    ----------
    transform_type : DisplacementTransformType
        The type of displacement transform to estimate. Default: DisplacementTransformType.MULTISCALE_DEMONS.
        Currently only DisplacementTransformType.MULTISCALE_DEMONS is supported.
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

    Raises
    ------
    ValueError
        If transform_type is not DisplacementTransformType.MULTISCALE_DEMONS.
    """

    def __init__(
        self,
        transform_type: DisplacementTransformType = DisplacementTransformType.MULTISCALE_DEMONS,
        demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
        demons_num_iterations: int = 100,
        demons_smooth_displacement_field: bool = True,
        demons_standard_deviations: float = 1.5,
        demons_intensity_difference_threshold: Optional[float] = None,
        demons_maximum_rms_error: Optional[float] = None,
        reference_image_key: TransformKey = TransformKey.REFERENCE_IMAGE,
        moving_image_key: TransformKey = TransformKey.MOVING_IMAGE,
    ) -> None:
        """Inits :class:`DisplacementModule`.

        Parameters
        ----------
        transform_type : DisplacementTransformType
            The type of displacement transform to estimate. Default: DisplacementTransformType.MULTISCALE_DEMONS.
            Currently only DisplacementTransformType.MULTISCALE_DEMONS is supported.
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

        Raises
        ------
        ValueError
            If transform_type is not DisplacementTransformType.MULTISCALE_DEMONS.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if transform_type != DisplacementTransformType.MULTISCALE_DEMONS:
            raise ValueError("Currently only DisplacementTransformType.MULTISCALE_DEMONS is supported.")

        self.displacement_transform = partial(
            multiscale_demons_displacement,
            filter_type=demons_filter_type,
            num_iterations=demons_num_iterations,
            smooth_displacement_field=demons_smooth_displacement_field,
            standard_deviations=demons_standard_deviations,
            intensity_difference_threshold=demons_intensity_difference_threshold,
            maximum_rms_error=demons_maximum_rms_error,
        )

        self.reference_image_key = reference_image_key
        self.moving_image_key = moving_image_key

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Estimate the displacement field between the reference and moving images.

        Parameters
        ----------
        sample : dict[str, Any]
            A dictionary containing the reference image and a sequence of images to estimate the displacement field
            (moving image).
        """
        reference_image = sample[self.reference_image_key]
        moving_image = sample[self.moving_image_key]

        device = reference_image.device

        # Estimate the displacement field
        displacement = [
            self.displacement_transform(reference_image[_].cpu(), moving_image[_].cpu())
            for _ in range(moving_image.shape[0])
        ]
        displacement = torch.stack(displacement, dim=0)
        displacement = displacement.to(device)

        sample[TransformKey.DISPLACEMENT_FIELD] = displacement

        return sample


class WarpModule(DirectModule):
    """Module for warping an image using a displacement field.

    Parameters
    ----------
    displacement_field_key : TransformKey
        The key for the displacement field in the sample dictionary. Default: TransformKey.DISPLACEMENT_FIELD.
    """

    def __init__(
        self,
        displacement_field_key: TransformKey = TransformKey.DISPLACEMENT_FIELD,
        moving_image_key: TransformKey = TransformKey.MOVING_IMAGE,
    ) -> None:
        """Inits :class:`WarpModule`.

        Parameters
        ----------
        displacement_field_key : TransformKey
            The key for the displacement field in the sample dictionary. Default: TransformKey.DISPLACEMENT_FIELD.
        moving_image_key : TransformKey
            The key for the moving image in the sample dictionary. Default: TransformKey.MOVING_IMAGE.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.displacement_field_key = displacement_field_key
        self.moving_image_key = moving_image_key

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Warp the moving image using the displacement field.

        Parameters
        ----------
        sample : dict[str, Any]
            A dictionary containing the moving image and the displacement field.
        """
        displacement_field = sample[self.displacement_field_key]
        moving_image = sample[self.moving_image_key]

        batch_size, sequence_length = moving_image.shape[:2]

        moving_image = moving_image.reshape(batch_size * sequence_length, 1, *moving_image.shape[2:])
        displacement_field = displacement_field.reshape(batch_size * sequence_length, *displacement_field.shape[2:])

        # Warp the moving image
        warped_image = warp_tensor(moving_image, displacement_field)
        warped_image = warped_image.reshape(batch_size, sequence_length, *warped_image.shape[1:])

        sample[TransformKey.WARPED_IMAGE] = warped_image

        return sample
