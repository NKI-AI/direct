# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform modules for displacement-field estimation and image warping."""

import logging
from functools import partial
from typing import Any

import torch

from direct.registration.demons import DemonsFilterType, multiscale_demons_displacement
from direct.registration.warp import warp_tensor
from direct.types import DirectEnum, TransformKey
from direct.utils import DirectModule

__all__ = ["DisplacementModule", "DisplacementTransformType", "WarpModule"]

DISCPLACEMENT_FIELD_2D_DIMENSIONS = 2
"""Number of displacement components for 2D displacement fields."""

DISCPLACEMENT_FIELD_3D_DIMENSIONS = 3
"""Number of displacement components for 3D displacement fields."""


class DisplacementTransformType(DirectEnum):
    """Supported displacement-field estimation backends."""

    MULTISCALE_DEMONS = "multiscale_demons"
    OPTICAL_FLOW = "optical_flow"


class DisplacementModule(DirectModule):
    """Transform module for estimating displacement fields between reference and moving images."""

    def __init__(
        self,
        transform_type: DisplacementTransformType = DisplacementTransformType.MULTISCALE_DEMONS,
        demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
        demons_num_iterations: int = 100,
        demons_smooth_displacement_field: bool = True,
        demons_standard_deviations: float = 1.5,
        demons_intensity_difference_threshold: float | None = None,
        demons_maximum_rms_error: float | None = None,
        reference_image_key: TransformKey = TransformKey.REFERENCE_IMAGE,
        moving_image_key: TransformKey = TransformKey.MOVING_IMAGE,
    ) -> None:
        """Inits :class:`DisplacementModule`.

        Args:
            transform_type: The type of displacement transform to estimate. Default is
                :attr:`~direct.registration.registration.DisplacementTransformType.MULTISCALE_DEMONS`. Currently only :attr:`~direct.registration.registration.DisplacementTransformType.MULTISCALE_DEMONS` is
                supported.
            demons_filter_type: Type of the Demons filter (:attr:`~direct.registration.demons.DemonsFilterType.DEMONS`, :attr:`~direct.registration.demons.DemonsFilterType.FAST_SYMMETRIC_FORCES`,
                :attr:`~direct.registration.demons.DemonsFilterType.SYMMETRIC_FORCES`, :attr:`~direct.registration.demons.DemonsFilterType.DIFFEOMORPHIC`). Default is
                :attr:`~direct.registration.demons.DemonsFilterType.SYMMETRIC_FORCES`.
            demons_num_iterations: Number of iterations for the Demons filter. Default is ``100``.
            demons_smooth_displacement_field: Whether to smooth the displacement field. Default is ``True``.
            demons_standard_deviations: Standard deviations for Gaussian smoothing. Default is ``1.5``.
            demons_intensity_difference_threshold: Intensity difference threshold. Default is ``None``.
            demons_maximum_rms_error: Maximum RMS error. Default is ``None``.
            reference_image_key: Dictionary key for the reference image. Default is :attr:`~direct.types.TransformKey.REFERENCE_IMAGE`.
            moving_image_key: Dictionary key for the moving image sequence. Default is :attr:`~direct.types.TransformKey.MOVING_IMAGE`.

        Returns:
            ``None``.

        Raises:
            If transform_type is not :attr:`~direct.registration.registration.DisplacementTransformType.MULTISCALE_DEMONS`.
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

        Args:
            sample: A dictionary containing the reference image and a sequence of images to estimate the displacement field
                (moving image).

        Returns:
            Input sample with the displacement field stored under :attr:`~direct.types.TransformKey.DISPLACEMENT_FIELD`.
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
    """Transform module for warping a moving image with a displacement field."""

    def __init__(
        self,
        displacement_field_key: TransformKey = TransformKey.DISPLACEMENT_FIELD,
        moving_image_key: TransformKey = TransformKey.MOVING_IMAGE,
    ) -> None:
        """Inits :class:`WarpModule`.

        Args:
            displacement_field_key: The key for the displacement field in the sample dictionary. Default is
                :attr:`~direct.types.TransformKey.DISPLACEMENT_FIELD`.
            moving_image_key: The key for the moving image in the sample dictionary. Default is :attr:`~direct.types.TransformKey.MOVING_IMAGE`.

        Returns:
            ``None``.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.displacement_field_key = displacement_field_key
        self.moving_image_key = moving_image_key

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Warp the moving image using the displacement field.

        Args:
            sample: A dictionary containing the moving image and the displacement field.

        Returns:
            Input sample with the warped image stored under :attr:`~direct.types.TransformKey.WARPED_IMAGE`.
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
