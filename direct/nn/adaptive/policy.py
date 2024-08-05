# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.policy module."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    Straight through policy model block.
    """

    def __init__(
        self,
        backward_operator: Callable,
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
    ):
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
            raise ValueError(f"Sampling dimension can be `1D` or `2D`.")

        if sampling_type != PolicySamplingType.STATIC:
            if len(kspace_shape) != 3:
                raise

        if sampling_type in [PolicySamplingType.DYNAMIC_2D_NON_UNIFORM, PolicySamplingType.MULTISLICE_2D_NON_UNIFORM]:
            self.num_actions *= kspace_shape[0]

        self.sampling_dimension = sampling_dimension
        self.sampling_type = sampling_type

        sampler_num_actions = self.num_actions * (
            kspace_shape[0]
            if sampling_type in [PolicySamplingType.DYNAMIC_2D, PolicySamplingType.MULTISLICE_2D]
            else 1
        )
        self.sampler = (KSpaceLineConvSampler if kspace_sampler else ImageLineConvSampler)(
            input_dim=(COMPLEX_SIZE, *kspace_shape),
            num_actions=sampler_num_actions,
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
        padding: Optional[torch.Tensor] = None,
    ):
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
            flat_prob_mask = normalize_masked_probabilities(mask, flat_prob_mask, budget)
            # Binarize the mask
            flat_bin_mask = self.binarizer(flat_prob_mask)
        else:
            mask = mask.reshape(masked_kspace.shape[0], masked_kspace.shape[2], -1)
            sampler_out = sampler_out.reshape(masked_kspace.shape[0], masked_kspace.shape[2], -1)

            flat_bin_mask = []
            flat_prob_mask = []

            for i in range(masked_kspace.shape[2]):
                flat_prob_mask.append(self.compute_prob_mask(sampler_out[:, i], mask[:, i]))
                # Take out zero (masked) probabilities and normalize
                flat_prob_mask[-1] = normalize_masked_probabilities(mask[:, i], flat_prob_mask[-1], budget[:, i])
                # Binarize the mask
                flat_bin_mask.append(self.binarizer(flat_prob_mask[-1]))
            flat_prob_mask = torch.stack(flat_prob_mask, dim=1)
            flat_bin_mask = torch.stack(flat_bin_mask, dim=1)
        return flat_bin_mask, flat_prob_mask

    def apply_acquisition(
        self, mask: torch.Tensor, acquisitions: torch.Tensor, kspace: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = self.backward_operator(kspace, dim=self.spatial_dims)
        return T.reduce_operator(x, sensitivity_map, self.coil_dim)

    def compute_prob_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.reshape(x.shape[0], self.num_actions)
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
        if kspace.shape[2] == self.kspace_shape[0]:
            return kspace
        padded_tensor = torch.cat(
            [
                kspace,
                torch.zeros(
                    (*kspace.shape[:2], self.kspace_shape[0] - kspace.shape[2], *kspace.shape[3:]),
                    dtype=kspace.dtype,
                    device=kspace.device,
                    requires_grad=True,
                ),
            ],
            dim=2,
        )
        return padded_tensor


class StraightThroughPolicy2dBlock(StraightThroughPolicyBlock):
    """
    Straight through policy 2D model block.
    """

    def __init__(
        self,
        backward_operator: Callable,
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
    ):
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
            raise ValueError(f"`kspace_shape` should have length equal to 2.")

        self.spatial_dims = (2, 3)

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        budget: int | torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
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
    """
    Straight through policy model block for 3D input.
    """

    def __init__(
        self,
        backward_operator: Callable,
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
    ):
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
            raise ValueError(f"`kspace_shape` should have length equal to 3.")

        self.spatial_dims = (3, 4)

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        budget: int | torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
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
    """
    Straight through policy model block for 3D input.
    """

    def __init__(
        self,
        backward_operator: Callable,
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
    ):
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
            raise ValueError(f"`kspace_shape` should have length equal to 3.")

        self.spatial_dims = (3, 4)

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        budget: int | torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
        masked_kspace = self.pad_time_or_slice_dimension(masked_kspace)
        sensitivity_map = self.pad_time_or_slice_dimension(sensitivity_map)

        # batch, coils, time_or_slice, height, width, complex
        batch_size, _, time_or_slice, height, width, _ = masked_kspace.shape

        if mask.shape[2] == 1:
            mask = mask.expand(1, 1, time_or_slice, height, width, 1)

        if padding is not None:
            if padding.shape[2] == 1:
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
    """Multi layer Straight through policy model."""

    def __init__(
        self,
        backward_operator: Callable,
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
        num_time_steps: Optional[int] = None,
        num_slices: Optional[int] = None,
        acceleration: Optional[float] = None,
    ):
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
            raise ValueError(f"Sampling dimension can be `1D` or `2D`.")

        if sampling_type != PolicySamplingType.STATIC:
            if len(kspace_shape) == 3:
                raise NotImplementedError(f"Dynamic sampling is only implemented for 2D data.")
            if sampling_type in [
                PolicySamplingType.DYNAMIC_2D,
                PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
            ]:
                self.num_time_or_slice_steps = num_time_steps
                kspace_shape = (num_time_steps, *kspace_shape)
            else:
                self.num_time_or_slice_steps = num_slices
                kspace_shape = (num_slices, *kspace_shape)

        if sampling_type in [PolicySamplingType.DYNAMIC_2D_NON_UNIFORM, PolicySamplingType.MULTISLICE_2D_NON_UNIFORM]:
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
            st_policy_block_kwargs["sampling_type"] = sampling_type

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(st_policy_block(**st_policy_block_kwargs))

        self.acceleration = acceleration

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        kspace: torch.Tensor,
        acceleration: Optional[float | torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
    ):
        if self.sampling_type in [
            PolicySamplingType.DYNAMIC_2D_NON_UNIFORM,
            PolicySamplingType.MULTISLICE_2D_NON_UNIFORM,
        ]:
            if mask.shape[2] < self.kspace_shape[0]:
                mask = mask.expand(mask.shape[0], 1, self.kspace_shape[0], mask.shape[3], mask.shape[4], 1)
                if padding is not None:
                    padding = padding.expand(*mask.shape)

        masks = [mask]
        prob_masks = []

        if self.acceleration is not None:
            acceleration = self.acceleration
        else:
            if acceleration is None:
                raise ValueError(
                    f"Argument `acceleration` received None for a value. "
                    f"This should not be None when `StraightThroughPolicy` is initialized "
                    f"with `acceleration` with None value."
                )
            else:
                if not isinstance(acceleration, (int, float, torch.Tensor)):
                    raise ValueError(f"Invalid `acceleration` type. Received `acceleration`={acceleration}.")
                if isinstance(acceleration, torch.Tensor):
                    if acceleration.shape[0] not in [1, kspace.shape[0]]:
                        raise ValueError(
                            f"Tensor accelerations should have first dimension equal to 1 or "
                            f"batch size matching the k-space."
                        )
                    if self.sampling_type not in [PolicySamplingType.DYNAMIC_2D, PolicySamplingType.MULTISLICE_2D]:
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
                        elif acceleration.ndim == 2:
                            if acceleration.shape[1] != kspace.shape[2]:
                                raise ValueError(
                                    f"Acceleration second dimension should match k-space 3rd dimension. "
                                    f"Received acceleration of shape={acceleration.shape} and k-space "
                                    f"of shape={kspace.shape}."
                                )

        if self.sampling_type not in [PolicySamplingType.DYNAMIC_2D, PolicySamplingType.MULTISLICE_2D]:
            sampled_fraction = torch.tensor(
                [mask[i].sum().item() / np.prod(mask[i].shape) for i in range(mask.shape[0])]
            )
        else:
            sampled_fraction = []
            for i in range(kspace.shape[0]):
                sampled_fraction.append(
                    torch.tensor(
                        [mask[i, :, j].sum().item() / np.prod(mask[i, :, j].shape) for j in range(mask.shape[2])]
                    )
                )
            sampled_fraction = torch.stack(sampled_fraction, 0)
            if isinstance(acceleration, torch.Tensor) and acceleration.ndim == 1:
                acceleration = acceleration.unsqueeze(1)

        sampled_fraction = sampled_fraction.to(mask.device)
        budget = self.num_actions * (1 / acceleration - sampled_fraction)

        budget = budget.round().int()

        layer_budget = budget // len(self.layers)

        for i, layer in enumerate(self.layers):
            if i == (len(self.layers) - 1):
                layer_budget = budget - (len(self.layers) - 1) * layer_budget

            mask, masked_kspace, prob_mask = layer.do_acquisition(
                kspace, masked_kspace, mask, sensitivity_map, layer_budget, padding
            )

            masks.append(mask)
            prob_masks.append(prob_mask)

        return masked_kspace, masks, prob_masks