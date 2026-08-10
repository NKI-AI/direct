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

"""PyTorch registration models for estimating displacement fields.

Provides classical (optical flow, demons) and learned (UNet, ViT, VoxelMorph)
models that warp a moving image sequence onto a reference image.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from direct.nn.registration.voxelmorph import VxmDense
from direct.nn.transformers.vit import VisionTransformer2D
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.registration.demons import DemonsFilterType, multiscale_demons_displacement
from direct.registration.optical_flow import (
    OpticalFlowEstimatorType,
    optical_flow_displacement,
)
from direct.registration.registration import DISCPLACEMENT_FIELD_2D_DIMENSIONS
from direct.registration.warp import warp

__all__ = [
    "DemonsRegistration2dModel",
    "OpticalFlowILKRegistration2dModel",
    "OpticalFlowTVL1Registration2dModel",
    "UnetRegistration2dModel",
    "VxmDense",
]


class ClassicalRegistration2dModel(nn.Module):
    """Base class for classical 2D registration models with a displacement transform."""

    def __init__(
        self,
        displacement_transform: Callable,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        """Inits :class:`ClassicalRegistration2dModel`.

        Parameters
        ----------
        displacement_transform : Callable
            Callable that estimates a displacement field from reference and moving images.
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """
        del kwargs
        super().__init__()
        self.displacement_transform = displacement_transform
        self.warp_num_integration_steps = warp_num_integration_steps

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`ClassicalRegistration2dModel`.

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
        warped_image = warp(
            moving_image,
            displacement_field,
            num_integration_steps=self.warp_num_integration_steps,
        )

        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class OpticalFlowRegistration2dModel(ClassicalRegistration2dModel):
    """2D registration model based on scikit-image optical-flow estimators."""

    def __init__(
        self,
        estimator_type: OpticalFlowEstimatorType,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        """Inits :class:`OpticalFlowRegistration2dModel`.

        Parameters
        ----------
        estimator_type : OpticalFlowEstimatorType
            Optical-flow estimator to use (ILK or TV-L1).
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        **kwargs
            Additional keyword arguments forwarded to the optical-flow estimator.
        """
        super().__init__(
            displacement_transform=partial(
                optical_flow_displacement,
                estimator_type=estimator_type,
                **kwargs,
            ),
            warp_num_integration_steps=warp_num_integration_steps,
        )

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`OpticalFlowRegistration2dModel`.

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
        warped_image = warp(
            moving_image,
            displacement_field,
            num_integration_steps=self.warp_num_integration_steps,
        )

        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class OpticalFlowILKRegistration2dModel(OpticalFlowRegistration2dModel):
    """2D registration model using iterative Lucas-Kanade optical flow."""

    def __init__(
        self,
        radius: int = 7,
        num_warp: int = 10,
        gaussian: bool = False,
        prefilter: bool = True,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        """Inits :class:`OpticalFlowILKRegistration2dModel`.

        Parameters
        ----------
        radius : int
            Radius of the window considered around each pixel. Default: 7.
        num_warp : int
            Number of times the moving image is warped. Default: 10.
        gaussian : bool
            If True, use a Gaussian kernel for local integration. Default: False.
        prefilter : bool
            Whether to prefilter the estimated optical flow before each warp. Default: True.
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """
        super().__init__(
            estimator_type=OpticalFlowEstimatorType.ILK,
            warp_num_integration_steps=warp_num_integration_steps,
            radius=radius,
            num_warp=num_warp,
            gaussian=gaussian,
            prefilter=prefilter,
        )


class OpticalFlowTVL1Registration2dModel(OpticalFlowRegistration2dModel):
    """2D registration model using TV-L1 optical flow."""

    def __init__(
        self,
        attachment: float = 15,
        tightness: float = 0.3,
        num_warp: int = 5,
        num_iter: int = 10,
        tol: float = 1e-3,
        prefilter: bool = True,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        """Inits :class:`OpticalFlowTVL1Registration2dModel`.

        Parameters
        ----------
        attachment : float
            Attachment parameter for TV-L1. Default: 15.
        tightness : float
            Tightness parameter for TV-L1. Default: 0.3.
        num_warp : int
            Number of times the moving image is warped. Default: 5.
        num_iter : int
            Number of fixed-point iterations. Default: 10.
        tol : float
            Stopping tolerance based on the L2 distance between consecutive flows. Default: 1e-3.
        prefilter : bool
            Whether to prefilter the estimated optical flow before each warp. Default: True.
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """
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
    """2D registration model using SimpleITK multiscale demons registration."""

    def __init__(
        self,
        demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
        demons_num_iterations: int = 50,
        demons_smooth_displacement_field: bool = True,
        demons_standard_deviations: float = 1.0,
        demons_intensity_difference_threshold: float | None = None,
        demons_maximum_rms_error: float | None = None,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        """Inits :class:`DemonsRegistration2dModel`.

        Parameters
        ----------
        demons_filter_type : DemonsFilterType
            Type of the Demons filter (DemonsFilterType.DEMONS, DemonsFilterType.FAST_SYMMETRIC_FORCES,
            DemonsFilterType.SYMMETRIC_FORCES, DemonsFilterType.DIFFEOMORPHIC). Default: DemonsFilterType.SYMMETRIC_FORCES.
        demons_num_iterations : int
            Number of iterations for the Demons filter. Default: 50.
        demons_smooth_displacement_field : bool
            Whether to smooth the displacement field. Default: True.
        demons_standard_deviations : float
            Standard deviations for Gaussian smoothing. Default: 1.0.
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

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`DemonsRegistration2dModel`.

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
        warped_image = warp(
            moving_image,
            displacement_field,
            num_integration_steps=self.warp_num_integration_steps,
        )

        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class UnetRegistration2dModel(nn.Module):
    """UNet-based 2D registration model that predicts dense displacement fields."""

    def __init__(
        self,
        max_seq_len: int,
        unet_num_filters: int = 16,
        unet_num_pool_layers: int = 4,
        unet_dropout_probability: float = 0.0,
        unet_normalized: bool = False,
        warp_num_integration_steps: int = 1,
        **kwargs,
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
        del kwargs
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

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            batch_size,
            self.max_seq_len,
            DISCPLACEMENT_FIELD_2D_DIMENSIONS,
            height,
            width,
        )  # (batch_size, max_seq_len, 2, height, width)

        # Crop the displacement field to the actual sequence length
        displacement_field = displacement_field[:, :seq_len]  # (batch_size, seq_len, 2, height, width)

        # Reshape the displacement field and moving image to be compatible with the warp module
        displacement_field = displacement_field.reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )
        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(
            moving_image,
            displacement_field,
            num_integration_steps=self.warp_num_integration_steps,
        )
        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


class ViTRegistration2dModel(nn.Module):
    """Vision Transformer registration model for 2D images."""

    def __init__(
        self,
        max_seq_len: int,
        average_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        warp_num_integration_steps: int = 1,
        **kwargs,
    ) -> None:
        """Inits :class:`ViTRegistration2dModel`.

        Parameters
        ----------
        max_seq_len : int
            Maximum sequence length expected in the moving image.
        average_size : int or tuple[int, int]
            The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and
            (average_size, average_size, average_size) for 3D. Default: 320.
        patch_size : int or tuple[int, int]
            The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
        embedding_dim : int
            Dimension of the output embedding.
        depth : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads.
        mlp_ratio : float
            The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
        qkv_bias : bool
            Whether to add bias to the query, key, and value projections. Default: False.
        qk_scale : float, optional
            The scale factor for the query-key dot product. Default: None.
        drop_rate : float
            The dropout probability for all dropout layers except dropout_path. Default: 0.0.
        attn_drop_rate : float
            The dropout probability for the attention layer. Default: 0.0.
        dropout_path_rate : float
            The dropout probability for the dropout path. Default: 0.0.
        gpsa_interval : tuple[int, int]
            The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        warp_num_integration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """
        super().__init__()
        # VisionTransformer API uses ``use_gpsa``; paper configs still pass ``gpsa_interval``.
        # ``(-1, -1)`` historically disabled GPSA for all blocks.
        use_gpsa = tuple(gpsa_interval) != (-1, -1)
        del kwargs
        self.transformer = VisionTransformer2D(
            average_img_size=average_size,
            patch_size=patch_size,
            in_channels=max_seq_len + 1,
            out_channels=max_seq_len * DISCPLACEMENT_FIELD_2D_DIMENSIONS,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_gpsa=use_gpsa,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=False,
        )
        self.max_seq_len = max_seq_len
        self.warp_num_integration_steps = warp_num_integration_steps

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`ViTRegistration2dModel`.

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
        displacement_field = self.transformer(x)

        # Model outputs the displacement field for each time step with 2 channels (x and y displacements)
        displacement_field = displacement_field.reshape(
            batch_size,
            self.max_seq_len,
            DISCPLACEMENT_FIELD_2D_DIMENSIONS,
            height,
            width,
        )  # (batch_size, max_seq_len, 2, height, width)

        # Crop the displacement field to the actual sequence length
        displacement_field = displacement_field[:, :seq_len]  # (batch_size, seq_len, 2, height, width)

        # Reshape the displacement field and moving image to be compatible with the warp module
        displacement_field = displacement_field.reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )
        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(
            moving_image,
            displacement_field,
            num_integration_steps=self.warp_num_integration_steps,
        )
        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )


# Backward-compatible alias used in some experiment dumps.
UnetRegistrationModel = UnetRegistration2dModel
