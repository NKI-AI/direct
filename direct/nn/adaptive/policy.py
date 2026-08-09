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

"""Straight-through adaptive k-space sampling policies."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import direct.data.transforms as T
from direct.constants import COMPLEX_SIZE
from direct.nn.adaptive.binarizer import ThresholdSigmoidMask
from direct.nn.adaptive.sampler import ImageLineConvSampler, KSpaceLineConvSampler
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.adaptive.utils import (
    normalize_masked_probabilities,
    reshape_acquisitions_post_sampling,
    reshape_mask_pre_sampling,
)
from direct.nn.types import ActivationType

__all__ = ["StraightThroughPolicy"]


class StraightThroughPolicyBlock(nn.Module):
    """Base straight-through policy block for a single acquisition step."""

    def __init__(
        self,
        backward_operator: Callable[..., torch.Tensor],
        kspace_shape: tuple[int, int],
        sampling_dimension: PolicySamplingDimension,
        sampling_type: PolicySamplingType = PolicySamplingType.STATIC,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKY_RELU,
    ) -> None:
        """Inits :class:`StraightThroughPolicyBlock`.

        Parameters
        ----------
        backward_operator : Callable[..., torch.Tensor]
            Adjoint Fourier operator used for coil combination.
        kspace_shape : tuple[int, int]
            Shape of the k-space region to sample.
        sampling_dimension : PolicySamplingDimension
            Sampling dimension, either 1D lines or 2D pixels.
        sampling_type : PolicySamplingType, optional
            Sampling strategy. Default: ``STATIC``.
        sampler_detach_mask : bool, optional
            Detach the mask before backpropagation. Default: ``False``.
        kspace_sampler : bool, optional
            Use k-space rather than image-domain observations. Default: ``False``.
        st_slope : float, optional
            Slope for the straight-through binarizer. Default: ``10``.
        st_clamp : bool, optional
            Clamp straight-through gradients. Default: ``False``.
        fix_sign_leakage : bool, optional
            Correct sign leakage in masked k-space. Default: ``True``.
        sampler_chans : int, optional
            Number of channels in the convolutional sampler. Default: ``16``.
        sampler_num_pool_layers : int, optional
            Number of pooling layers in the sampler. Default: ``4``.
        sampler_fc_size : int, optional
            Hidden size of sampler fully connected layers. Default: ``256``.
        sampler_drop_prob : float, optional
            Dropout probability in the sampler. Default: ``0``.
        slope : float, optional
            Slope for softplus or sigmoid probability mapping. Default: ``10``.
        use_softplus : bool, optional
            Use softplus instead of sigmoid for probabilities. Default: ``True``.
        sampler_num_fc_layers : int, optional
            Number of fully connected layers in the sampler. Default: ``3``.
        sampler_activation : ActivationType, optional
            Activation function in the sampler MLP. Default: ``LEAKY_RELU``.
        """
        super().__init__()

        if len(kspace_shape) not in [2, 3]:
            raise ValueError(
                f"`kspace_shape` should have length equal to 2 for 2D input or 3 for 3D input."
                f" Received: `kspace_shape`={kspace_shape}."
            )
        self.kspace_shape = kspace_shape

        if sampling_dimension == PolicySamplingDimension.ONE_D:
            self.num_actions = kspace_shape[-1]  # num_actions = width
        elif sampling_dimension == PolicySamplingDimension.TWO_D:
            self.num_actions = np.prod(kspace_shape[-2:])  # num_actions = height * width
        else:
            raise ValueError("Sampling dimension can be `1D` or `2D`.")

        if sampling_type != PolicySamplingType.STATIC and len(kspace_shape) != 3:
            raise ValueError(
                f"`sampling_type`={sampling_type} requires 3D `kspace_shape` (got length {len(kspace_shape)})."
            )

        if sampling_type in [
            PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
            PolicySamplingType.MULTISLICE_2D_NON_UNIFORM,
        ]:
            self.num_actions *= kspace_shape[0]

        self.sampling_dimension = sampling_dimension
        self.sampling_type = sampling_type

        sampler_num_actions = self.num_actions * (
            kspace_shape[0] if sampling_type in [PolicySamplingType.DYNAMIC_2D, PolicySamplingType.MULTISLICE_2D] else 1
        )
        self.sampler = (KSpaceLineConvSampler if kspace_sampler else ImageLineConvSampler)(
            input_dim=(COMPLEX_SIZE, *kspace_shape),
            num_actions=sampler_num_actions,  # ty: ignore[invalid-argument-type]
            chans=sampler_chans,
            num_pool_layers=sampler_num_pool_layers,
            fc_size=sampler_fc_size,
            drop_prob=sampler_drop_prob,
            num_fc_layers=sampler_num_fc_layers,
            activation=sampler_activation,
        )
        self.kspace_sampler = kspace_sampler
        self.slope = slope
        self.use_softplus = use_softplus

        self.binarizer = ThresholdSigmoidMask(st_slope, st_clamp)

        self.sampler_detach_mask = sampler_detach_mask
        self.fix_sign_leakage = fix_sign_leakage

        self.backward_operator = backward_operator
        self.coil_dim = 1

    def forward(
        self,
        mask: torch.Tensor,
        image: torch.Tensor,
        masked_kspace: torch.Tensor,
        budget: int | torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample additional k-space lines and return binary and probability masks.

        Parameters
        ----------
        mask : torch.Tensor
            Current flat sampling mask.
        image : torch.Tensor
            Coil-combined image observation.
        masked_kspace : torch.Tensor
            Currently masked k-space tensor.
        budget : int | torch.Tensor
            Remaining sampling budget per batch element.
        padding : torch.Tensor | None, optional
            Padding mask excluding invalid locations from sampling.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Binary acquisition mask and corresponding probability mask.
        """
        if len(self.kspace_shape) == 2:
            sampler_input = masked_kspace.permute(0, 1, 4, 2, 3) if self.kspace_sampler else image.permute(0, 3, 1, 2)
        else:
            sampler_input = (
                masked_kspace.permute(0, 1, 5, 2, 3, 4) if self.kspace_sampler else image.permute(0, 4, 1, 2, 3)
            )

        sampler_out = self.sampler(sampler_input, mask)

        # Mask out padded areas
        if padding is not None:
            mask = mask * (1 - padding)

        if self.sampling_type in [
            PolicySamplingType.STATIC,
            PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
            PolicySamplingType.MULTISLICE_2D_NON_UNIFORM,
        ]:
            flat_prob_mask = self.compute_prob_mask(sampler_out, mask)
            # Take out zero (masked) probabilities and normalize
            flat_prob_mask = normalize_masked_probabilities(mask, flat_prob_mask, budget)  # ty: ignore[invalid-argument-type]
            # Binarize the mask
            flat_bin_mask = self.binarizer(flat_prob_mask)
        else:
            mask = mask.reshape(masked_kspace.shape[0], masked_kspace.shape[2], -1)
            sampler_out = sampler_out.reshape(masked_kspace.shape[0], masked_kspace.shape[2], -1)

            flat_bin_mask = []
            flat_prob_mask = []

            # Broadcast a shared (batch,) / (batch, 1) budget across time when needed.
            frame_budget = None
            if budget.ndim == 1:  # ty: ignore[unresolved-attribute]
                frame_budget = budget
                per_frame = False
            elif budget.shape[1] == 1:  # ty: ignore[unresolved-attribute]
                frame_budget = budget[:, 0]  # ty: ignore[not-subscriptable]
                per_frame = False
            else:
                per_frame = True

            for i in range(masked_kspace.shape[2]):
                flat_prob_mask.append(self.compute_prob_mask(sampler_out[:, i], mask[:, i]))
                # Take out zero (masked) probabilities and normalize
                bi = budget[:, i] if per_frame else frame_budget  # ty: ignore[not-subscriptable]
                flat_prob_mask[-1] = normalize_masked_probabilities(mask[:, i], flat_prob_mask[-1], bi)  # ty: ignore[invalid-argument-type]
                # Binarize the mask
                flat_bin_mask.append(self.binarizer(flat_prob_mask[-1]))
            flat_prob_mask = torch.stack(flat_prob_mask, dim=1)
            flat_bin_mask = torch.stack(flat_bin_mask, dim=1)
        return flat_bin_mask, flat_prob_mask

    def apply_acquisition(
        self, mask: torch.Tensor, acquisitions: torch.Tensor, kspace: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sampled acquisitions to the current mask and k-space.

        Parameters
        ----------
        mask : torch.Tensor
            Current sampling mask.
        acquisitions : torch.Tensor
            Newly sampled binary acquisitions.
        kspace : torch.Tensor
            Full k-space tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated mask and masked k-space.
        """
        mask = mask + acquisitions

        masked_kspace = mask * kspace

        if self.sampler_detach_mask:
            mask = mask.detach()
        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace

    def sens_reduce(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Coil-combine k-space via the adjoint operator and sensitivity maps.

        Parameters
        ----------
        kspace : torch.Tensor
            Multi-coil k-space tensor.
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.

        Returns
        -------
        torch.Tensor
            Coil-combined image.
        """
        x = self.backward_operator(kspace, dim=self.spatial_dims)
        return T.reduce_operator(x, sensitivity_map, self.coil_dim)

    def compute_prob_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convert sampler logits into a normalized probability mask.

        Parameters
        ----------
        x : torch.Tensor
            Sampler output logits of shape ``(batch, num_actions)``.
        mask : torch.Tensor
            Current sampling mask used to exclude already sampled locations.

        Returns
        -------
        torch.Tensor
            Normalized probability mask of shape ``(batch, num_actions)``.
        """
        mask = mask.reshape(x.shape[0], self.num_actions)  # ty: ignore[invalid-argument-type]
        if self.use_softplus:
            # Softplus to make positive
            out = F.softplus(x, beta=self.slope)
            # Make sure max probability is 1, but ignore already sampled rows for this normalisation, since
            #  those get masked out later anyway.
            prob_mask = out / torch.max((1 - mask) * out, dim=1)[0].reshape(-1, 1)
        else:
            prob_mask = torch.sigmoid(self.slope * x)
        # Mask out already sampled rows
        prob_mask = prob_mask * (1 - mask)
        return prob_mask

    def pad_time_or_slice_dimension(self, kspace: torch.Tensor) -> torch.Tensor:
        """Pad the time or slice dimension to match the configured k-space shape.

        Parameters
        ----------
        kspace : torch.Tensor
            Input k-space or sensitivity map tensor.

        Returns
        -------
        torch.Tensor
            Tensor padded along the time/slice dimension when needed.
        """
        if kspace.shape[2] == self.kspace_shape[0]:
            return kspace
        padded_tensor = torch.cat(
            [
                kspace,
                torch.zeros(
                    (
                        *kspace.shape[:2],
                        self.kspace_shape[0] - kspace.shape[2],
                        *kspace.shape[3:],
                    ),
                    dtype=kspace.dtype,
                    device=kspace.device,
                    requires_grad=True,
                ),
            ],
            dim=2,
        )
        return padded_tensor


class StraightThroughPolicy2dBlock(StraightThroughPolicyBlock):
    """Straight-through policy block for static 2D k-space data."""

    def __init__(
        self,
        backward_operator: Callable[..., torch.Tensor],
        kspace_shape: tuple[int, int],
        sampling_dimension: PolicySamplingDimension,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKY_RELU,
    ) -> None:
        """Inits :class:`StraightThroughPolicy2dBlock`."""
        super().__init__(
            backward_operator=backward_operator,
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=PolicySamplingType.STATIC,
            sampler_detach_mask=sampler_detach_mask,
            kspace_sampler=kspace_sampler,
            st_slope=st_slope,
            st_clamp=st_clamp,
            fix_sign_leakage=fix_sign_leakage,
            sampler_chans=sampler_chans,
            sampler_num_pool_layers=sampler_num_pool_layers,
            sampler_fc_size=sampler_fc_size,
            sampler_drop_prob=sampler_drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            sampler_num_fc_layers=sampler_num_fc_layers,
            sampler_activation=sampler_activation,
        )
        if len(kspace_shape) != 2:
            raise ValueError("`kspace_shape` should have length equal to 2.")

        self.spatial_dims = (2, 3)

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        budget: int | torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one adaptive acquisition step for 2D k-space data.

        Parameters
        ----------
        kspace : torch.Tensor
            Full k-space tensor of shape ``(batch, coils, height, width, complex)``.
        masked_kspace : torch.Tensor
            Currently masked k-space tensor.
        mask : torch.Tensor
            Current sampling mask.
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.
        budget : int | torch.Tensor
            Remaining sampling budget.
        padding : torch.Tensor | None, optional
            Optional padding mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated mask, masked k-space, and probability mask.
        """
        # batch, coils, height, width, complex
        if kspace.ndim != 5:
            raise ValueError(f"Expected shape of k-space to have 5 dimensions, but got shape={kspace.shape}.")

        mask, padding = reshape_mask_pre_sampling(self.sampling_dimension, mask, padding, kspace.shape)

        image = self.sens_reduce(masked_kspace, sensitivity_map)

        acquisitions, flat_prob_mask = self(mask, image, masked_kspace, budget, padding)

        acquisitions, prob_mask, mask = reshape_acquisitions_post_sampling(
            self.sampling_dimension, acquisitions, flat_prob_mask, mask, kspace.shape
        )

        mask, masked_kspace = self.apply_acquisition(mask, acquisitions, kspace)

        return mask, masked_kspace, prob_mask


class StraightThroughPolicy3dBlock(StraightThroughPolicyBlock):
    """Straight-through policy block for static 3D k-space data."""

    def __init__(
        self,
        backward_operator: Callable[..., torch.Tensor],
        kspace_shape: tuple[int, int],
        sampling_dimension: PolicySamplingDimension,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKY_RELU,
    ) -> None:
        """Inits :class:`StraightThroughPolicy3dBlock`."""
        super().__init__(
            backward_operator=backward_operator,
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=PolicySamplingType.STATIC,
            sampler_detach_mask=sampler_detach_mask,
            kspace_sampler=kspace_sampler,
            st_slope=st_slope,
            st_clamp=st_clamp,
            fix_sign_leakage=fix_sign_leakage,
            sampler_chans=sampler_chans,
            sampler_num_pool_layers=sampler_num_pool_layers,
            sampler_fc_size=sampler_fc_size,
            sampler_drop_prob=sampler_drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            sampler_num_fc_layers=sampler_num_fc_layers,
            sampler_activation=sampler_activation,
        )
        if len(kspace_shape) != 3:
            raise ValueError("`kspace_shape` should have length equal to 3.")

        self.spatial_dims = (3, 4)

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        budget: int | torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one adaptive acquisition step for 3D k-space data.

        Parameters
        ----------
        kspace : torch.Tensor
            Full k-space tensor of shape ``(batch, coils, slice, height, width, complex)``.
        masked_kspace : torch.Tensor
            Currently masked k-space tensor.
        mask : torch.Tensor
            Current sampling mask.
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.
        budget : int | torch.Tensor
            Remaining sampling budget.
        padding : torch.Tensor | None, optional
            Optional padding mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated mask, masked k-space, and probability mask.
        """
        # batch, coils, slice, height, width, complex
        if kspace.ndim != 6:
            raise ValueError(f"Expected shape of k-space to have 6 dimensions, but got shape={kspace.shape}.")

        masked_kspace = self.pad_time_or_slice_dimension(masked_kspace)
        sensitivity_map = self.pad_time_or_slice_dimension(sensitivity_map)

        mask, padding = reshape_mask_pre_sampling(self.sampling_dimension, mask, padding, kspace.shape)

        image = self.sens_reduce(masked_kspace, sensitivity_map)

        acquisitions, flat_prob_mask = self(mask, image, masked_kspace, budget, padding)

        acquisitions, prob_mask, mask = reshape_acquisitions_post_sampling(
            self.sampling_dimension, acquisitions, flat_prob_mask, mask, kspace.shape
        )

        mask, masked_kspace = self.apply_acquisition(mask, acquisitions, kspace)

        return mask, masked_kspace, prob_mask


class StraightThroughPolicyDynamicOrMultislice2dBlock(StraightThroughPolicyBlock):
    """Straight-through policy block for dynamic or multislice 2D data."""

    def __init__(
        self,
        backward_operator: Callable[..., torch.Tensor],
        kspace_shape: tuple[int, int],
        sampling_dimension: PolicySamplingDimension,
        sampling_type: PolicySamplingType,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKY_RELU,
    ) -> None:
        """Inits :class:`StraightThroughPolicyDynamicOrMultislice2dBlock`."""
        super().__init__(
            backward_operator=backward_operator,
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=sampling_type,
            sampler_detach_mask=sampler_detach_mask,
            kspace_sampler=kspace_sampler,
            st_slope=st_slope,
            st_clamp=st_clamp,
            fix_sign_leakage=fix_sign_leakage,
            sampler_chans=sampler_chans,
            sampler_num_pool_layers=sampler_num_pool_layers,
            sampler_fc_size=sampler_fc_size,
            sampler_drop_prob=sampler_drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            sampler_num_fc_layers=sampler_num_fc_layers,
            sampler_activation=sampler_activation,
        )
        if len(kspace_shape) != 3:
            raise ValueError("`kspace_shape` should have length equal to 3.")

        self.spatial_dims = (3, 4)

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        budget: int | torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one adaptive acquisition step for dynamic or multislice 2D data.

        Parameters
        ----------
        kspace : torch.Tensor
            Full k-space tensor of shape ``(batch, coils, time_or_slice, height, width, complex)``.
        masked_kspace : torch.Tensor
            Currently masked k-space tensor.
        mask : torch.Tensor
            Current sampling mask.
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.
        budget : int | torch.Tensor
            Remaining sampling budget.
        padding : torch.Tensor | None, optional
            Optional padding mask.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated mask, masked k-space, and probability mask.
        """
        masked_kspace = self.pad_time_or_slice_dimension(masked_kspace)
        sensitivity_map = self.pad_time_or_slice_dimension(sensitivity_map)

        # batch, coils, time_or_slice, height, width, complex
        batch_size, _, time_or_slice, height, width, _ = masked_kspace.shape

        if mask.shape[2] == 1:
            mask = mask.expand(1, 1, time_or_slice, height, width, 1)

        if padding is not None and padding.shape[2] == 1:
            padding = padding.expand(1, 1, time_or_slice, height, width, 1)

        if self.sampling_dimension == PolicySamplingDimension.ONE_D:
            mask = mask[:, :, :, 0, :, :].reshape(batch_size, -1)
            if padding is not None:
                padding = padding[:, :, :, 0, :, :].reshape(batch_size, -1)
        else:
            mask = mask.reshape(batch_size, -1)
            if padding is not None:
                padding = padding.reshape(batch_size, -1)

        image = self.sens_reduce(masked_kspace, sensitivity_map)

        acquisitions, flat_prob_mask = self(mask, image, masked_kspace, budget, padding)

        if self.sampling_dimension == PolicySamplingDimension.ONE_D:
            acquisitions = acquisitions.reshape(batch_size, 1, time_or_slice, 1, width, 1).expand(
                batch_size, 1, time_or_slice, height, width, 1
            )
            prob_mask = flat_prob_mask.reshape(batch_size, 1, time_or_slice, 1, width, 1).expand(
                batch_size, 1, time_or_slice, height, width, 1
            )
            mask = mask.reshape(batch_size, 1, time_or_slice, 1, width, 1).expand(
                batch_size, 1, time_or_slice, height, width, 1
            )
        else:
            acquisitions = acquisitions.reshape(batch_size, 1, time_or_slice, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, time_or_slice, height, width, 1)
            mask = mask.reshape(batch_size, 1, time_or_slice, height, width, 1)

        mask, masked_kspace = self.apply_acquisition(mask, acquisitions, kspace)

        return mask, masked_kspace, prob_mask


class StraightThroughPolicy(nn.Module):
    """Multi-layer straight-through adaptive k-space sampling policy."""

    def __init__(
        self,
        backward_operator: Callable[..., torch.Tensor],
        kspace_shape: tuple[int, int],
        num_layers: int = 1,
        sampling_dimension: PolicySamplingDimension = PolicySamplingDimension.ONE_D,
        sampling_type: PolicySamplingType = PolicySamplingType.STATIC,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKY_RELU,
        num_time_steps: int | None = None,
        num_slices: int | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Inits :class:`StraightThroughPolicy`.

        Parameters
        ----------
        backward_operator : Callable[..., torch.Tensor]
            Adjoint Fourier operator used for coil combination.
        kspace_shape : tuple[int, int]
            Shape of the k-space region to sample.
        num_layers : int, optional
            Number of sequential acquisition layers. Default: ``1``.
        sampling_dimension : PolicySamplingDimension, optional
            Sampling dimension. Default: ``ONE_D``.
        sampling_type : PolicySamplingType, optional
            Sampling strategy. Default: ``STATIC``.
        sampler_detach_mask : bool, optional
            Detach the mask before backpropagation. Default: ``False``.
        kspace_sampler : bool, optional
            Use k-space rather than image-domain observations. Default: ``False``.
        st_slope : float, optional
            Slope for the straight-through binarizer. Default: ``10``.
        st_clamp : bool, optional
            Clamp straight-through gradients. Default: ``False``.
        fix_sign_leakage : bool, optional
            Correct sign leakage in masked k-space. Default: ``True``.
        sampler_chans : int, optional
            Number of channels in the convolutional sampler. Default: ``16``.
        sampler_num_pool_layers : int, optional
            Number of pooling layers in the sampler. Default: ``4``.
        sampler_fc_size : int, optional
            Hidden size of sampler fully connected layers. Default: ``256``.
        sampler_drop_prob : float, optional
            Dropout probability in the sampler. Default: ``0``.
        slope : float, optional
            Slope for softplus or sigmoid probability mapping. Default: ``10``.
        use_softplus : bool, optional
            Use softplus instead of sigmoid for probabilities. Default: ``True``.
        sampler_num_fc_layers : int, optional
            Number of fully connected layers in the sampler. Default: ``3``.
        sampler_activation : ActivationType, optional
            Activation function in the sampler MLP. Default: ``LEAKY_RELU``.
        num_time_steps : int | None, optional
            Number of time frames for dynamic 2D sampling.
        num_slices : int | None, optional
            Number of slices for multislice 2D sampling.
        acceleration : float | None, optional
            Fixed acceleration factor. When ``None``, acceleration is passed at runtime.
        """
        super().__init__()

        if len(kspace_shape) not in [2, 3]:
            raise ValueError(
                f"`kspace_shape` should have length equal to 2 for 2D input or 3 for 3D input."
                f" Received: `kspace_shape`={kspace_shape}."
            )

        if sampling_dimension == PolicySamplingDimension.ONE_D:
            self.num_actions = kspace_shape[-1]  # num_actions = width
        elif sampling_dimension == PolicySamplingDimension.TWO_D:
            self.num_actions = np.prod(kspace_shape[-2:])  # num_actions = height * width
        else:
            raise ValueError("Sampling dimension can be `1D` or `2D`.")

        if sampling_type != PolicySamplingType.STATIC:
            if len(kspace_shape) == 3:
                raise NotImplementedError("Dynamic sampling is only implemented for 2D data.")
            if sampling_type in [
                PolicySamplingType.DYNAMIC_2D,
                PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
            ]:
                self.num_time_or_slice_steps = num_time_steps
                kspace_shape = (num_time_steps, *kspace_shape)  # ty: ignore[invalid-assignment]
            else:
                self.num_time_or_slice_steps = num_slices
                kspace_shape = (num_slices, *kspace_shape)  # ty: ignore[invalid-assignment]

        if sampling_type in [
            PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
            PolicySamplingType.MULTISLICE_2D_NON_UNIFORM,
        ]:
            self.num_actions *= kspace_shape[0]

        self.kspace_shape = kspace_shape
        self.sampling_dimension = sampling_dimension
        self.sampling_type = sampling_type

        st_policy_block_kwargs = {
            "backward_operator": backward_operator,
            "kspace_shape": kspace_shape,
            "sampling_dimension": sampling_dimension,
            "sampler_detach_mask": sampler_detach_mask,
            "kspace_sampler": kspace_sampler,
            "st_slope": st_slope,
            "st_clamp": st_clamp,
            "fix_sign_leakage": fix_sign_leakage,
            "sampler_chans": sampler_chans,
            "sampler_num_pool_layers": sampler_num_pool_layers,
            "sampler_fc_size": sampler_fc_size,
            "sampler_drop_prob": sampler_drop_prob,
            "slope": slope,
            "use_softplus": use_softplus,
            "sampler_num_fc_layers": sampler_num_fc_layers,
            "sampler_activation": sampler_activation,
        }

        if sampling_type == PolicySamplingType.STATIC:
            st_policy_block = StraightThroughPolicy2dBlock if len(kspace_shape) == 2 else StraightThroughPolicy3dBlock
        else:
            st_policy_block = StraightThroughPolicyDynamicOrMultislice2dBlock
            st_policy_block_kwargs["sampling_type"] = sampling_type  # ty: ignore[invalid-assignment]

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(st_policy_block(**st_policy_block_kwargs))  # ty: ignore[invalid-argument-type]

        self.acceleration = acceleration

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        kspace: torch.Tensor,
        acceleration: float | torch.Tensor | None = None,
        padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run sequential adaptive acquisition layers.

        Parameters
        ----------
        masked_kspace : torch.Tensor
            Currently masked k-space tensor.
        mask : torch.Tensor
            Initial sampling mask.
        sensitivity_map : torch.Tensor
            Coil sensitivity maps.
        kspace : torch.Tensor
            Full k-space tensor.
        acceleration : float | torch.Tensor | None, optional
            Target acceleration factor. Required when not fixed at initialization.
        padding : torch.Tensor | None, optional
            Optional padding mask excluding invalid locations.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
            Final masked k-space, mask history, and probability mask history.
        """
        # DYNAMIC / MULTISLICE policies compute a per-frame budget from mask.shape[2].
        # Inference YAMLs often use STATIC ACS/init masks with a singleton time axis;
        # expand before budgeting so budget matches the padded k-space time dimension.
        if (
            self.sampling_type
            in [
                PolicySamplingType.DYNAMIC_2D,
                PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
                PolicySamplingType.MULTISLICE_2D,
                PolicySamplingType.MULTISLICE_2D_NON_UNIFORM,
            ]
            and mask.shape[2] < self.kspace_shape[0]
        ):
            mask = mask.expand(
                mask.shape[0],
                1,
                self.kspace_shape[0],
                mask.shape[3],
                mask.shape[4],
                1,
            )
            if padding is not None:
                padding = padding.expand(*mask.shape)

        masks = [mask]
        prob_masks = []

        if self.acceleration is not None:
            acceleration = self.acceleration
        elif acceleration is None:
            raise ValueError(
                "Argument `acceleration` received None for a value. "
                "This should not be None when `StraightThroughPolicy` is initialized "
                "with `acceleration` with None value."
            )
        elif not isinstance(acceleration, (int, float, torch.Tensor)):
            raise ValueError(f"Invalid `acceleration` type. Received `acceleration`={acceleration}.")
        elif isinstance(acceleration, torch.Tensor):
            if acceleration.shape[0] not in [1, kspace.shape[0]]:
                raise ValueError(
                    "Tensor accelerations should have first dimension equal to 1 or batch size matching the k-space."
                )
            if self.sampling_type not in [
                PolicySamplingType.DYNAMIC_2D,
                PolicySamplingType.MULTISLICE_2D,
            ]:
                acceleration = acceleration.squeeze()
                if acceleration.ndim > 1:
                    raise ValueError(
                        f"Tensor accelerations should be 1-dimensional for "
                        f"`sampling_type`={self.sampling_type}. "
                        f"Received `acceleration` of shape ={acceleration.shape}."
                    )
            else:
                if acceleration.ndim > 2:
                    raise ValueError(
                        f"Tensor accelerations should be 1 or 2-dimensional for "
                        f"`sampling_type`={self.sampling_type}. "
                        f"Received `acceleration`={acceleration}."
                    )
                if acceleration.ndim == 2 and acceleration.shape[1] != kspace.shape[2]:
                    raise ValueError(
                        f"Acceleration second dimension should match k-space 3rd dimension. "
                        f"Received acceleration of shape={acceleration.shape} and k-space "
                        f"of shape={kspace.shape}."
                    )

        frac_dtype = mask.dtype if mask.is_floating_point() else torch.float32
        if self.sampling_type not in [
            PolicySamplingType.DYNAMIC_2D,
            PolicySamplingType.MULTISLICE_2D,
        ]:
            sampled_fraction = torch.tensor(
                [mask[i].sum().item() / np.prod(mask[i].shape) for i in range(mask.shape[0])],
                dtype=frac_dtype,
            )
        else:
            sampled_fraction = []
            for i in range(kspace.shape[0]):
                sampled_fraction.append(
                    torch.tensor(
                        [mask[i, :, j].sum().item() / np.prod(mask[i, :, j].shape) for j in range(mask.shape[2])],
                        dtype=frac_dtype,
                    )
                )
            sampled_fraction = torch.stack(sampled_fraction, 0)
            if isinstance(acceleration, torch.Tensor) and acceleration.ndim == 1:
                acceleration = acceleration.unsqueeze(1)

        sampled_fraction = sampled_fraction.to(device=mask.device, dtype=frac_dtype)
        budget = self.num_actions * (1 / acceleration - sampled_fraction)  # ty: ignore[unsupported-operator]

        budget = budget.round().int()

        layer_budget = budget // len(self.layers)

        for i, layer in enumerate(self.layers):
            if i == (len(self.layers) - 1):
                layer_budget = budget - (len(self.layers) - 1) * layer_budget

            mask, masked_kspace, prob_mask = layer.do_acquisition(  # ty: ignore[call-non-callable]
                kspace, masked_kspace, mask, sensitivity_map, layer_budget, padding
            )

            masks.append(mask)
            prob_masks.append(prob_mask)

        return masked_kspace, masks, prob_masks
