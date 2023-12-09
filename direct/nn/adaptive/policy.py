# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.policy module."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

import direct.data.transforms as T
from direct.constants import COMPLEX_SIZE
from direct.nn.adaptive.binarizer import ThresholdSigmoidMask
from direct.nn.adaptive.sampler import ImageLineConvSampler, KSpaceLineConvSampler
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.adaptive.utils import rescale_probs
from direct.nn.types import ActivationType

__all__ = ["StraightThroughPolicy"]


class StraightThroughPolicyBlock(nn.Module):
    """
    Straight through policy model block.
    """

    def __init__(
        self,
        budget: int,
        backward_operator: Callable,
        kspace_shape: tuple[int, int],
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        sampler_slope: float = 10,
        sampler_use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKYRELU,
        sampler_cwn_conv: bool = False,
    ):
        super().__init__()

        if len(kspace_shape) not in [2, 3]:
            raise ValueError(
                f"`kspace_shape` should have length equal to 2 for 2D input or 3 for 3D input."
                f" Received: `kspace_shape`={kspace_shape}."
            )

        self.sampler = (KSpaceLineConvSampler if kspace_sampler else ImageLineConvSampler)(
            input_dim=(COMPLEX_SIZE, *kspace_shape),
            num_actions=self.num_actions,
            chans=sampler_chans,
            num_pool_layers=sampler_num_pool_layers,
            fc_size=sampler_fc_size,
            drop_prob=sampler_drop_prob,
            slope=sampler_slope,
            use_softplus=sampler_use_softplus,
            num_fc_layers=sampler_num_fc_layers,
            activation=sampler_activation,
            cwn_conv=sampler_cwn_conv,
        )
        self.kspace_sampler = kspace_sampler

        self.binarizer = ThresholdSigmoidMask(st_slope, st_clamp)

        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask
        self.fix_sign_leakage = fix_sign_leakage

        self.backward_operator = backward_operator
        self.coil_dim = 1

    def forward(
        self,
        mask: torch.Tensor,
        image: torch.Tensor,
        masked_kspace: torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
        batch_size = mask.shape[0]

        if self.kspace_shape == 2:
            sampler_input = masked_kspace.permute(0, 1, 4, 2, 3) if self.kspace_sampler else image.permute(0, 3, 1, 2)
        else:
            sampler_input = (
                masked_kspace.permute(0, 1, 5, 2, 3, 4) if self.kspace_sampler else image.permute(0, 4, 1, 2, 3)
            )

        flat_prob_mask = self.sampler(sampler_input, mask)

        # Mask out padded areas
        if padding is not None:
            mask = mask * (1 - padding)
        # Take out zero (masked) probabilities, since we don't want to include those in the normalisation
        nonzero_idcs = (mask == 0).nonzero(as_tuple=True)
        probs_to_norm = flat_prob_mask[nonzero_idcs].reshape(batch_size, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = rescale_probs(probs_to_norm, self.budget)
        # Reassign to original array
        flat_prob_mask[nonzero_idcs] = normed_probs.flatten()
        # Binarize the mask
        flat_bin_mask = self.binarizer(flat_prob_mask)
        return flat_bin_mask, flat_prob_mask

    def sens_reduce(self, x: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        x = self.backward_operator(x, dim=self.spatial_dims)
        return T.reduce_operator(x, sensitivity_map, self.coil_dim)


class StraightThroughPolicy2dBlock(StraightThroughPolicyBlock):
    """
    Straight through policy 2D model block.
    """

    def __init__(
        self,
        budget: int,
        backward_operator: Callable,
        kspace_shape: tuple[int, int],
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        sampler_slope: float = 10,
        sampler_use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKYRELU,
        sampler_cwn_conv: bool = False,
    ):
        super().__init__(
            budget=budget,
            backward_operator=backward_operator,
            kspace_shape=kspace_shape,
            sampler_detach_mask=sampler_detach_mask,
            kspace_sampler=kspace_sampler,
            st_slope=st_slope,
            st_clamp=st_clamp,
            fix_sign_leakage=fix_sign_leakage,
            sampler_chans=sampler_chans,
            sampler_num_pool_layers=sampler_num_pool_layers,
            sampler_fc_size=sampler_fc_size,
            sampler_drop_prob=sampler_drop_prob,
            sampler_slope=sampler_slope,
            sampler_use_softplus=sampler_use_softplus,
            sampler_num_fc_layers=sampler_num_fc_layers,
            sampler_activation=sampler_activation,
            sampler_cwn_conv=sampler_cwn_conv,
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
        padding: Optional[torch.Tensor] = None,
    ):
        batch_size, _, height, width, _ = kspace.shape  # batch, coils, height, width, complex
        if self.PolicySamplingDimension.ONE_D:
            mask = mask[:, :, 0, :, :].reshape(batch_size, width)
            if padding:
                # This assumes padding has same pattern as mask
                padding = padding[:, :, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask.reshape(batch_size, height * width)
            if padding:
                padding = padding.reshape(batch_size, height * width)

        # BMHWC --> BCHW
        image = self.sens_reduce(masked_kspace, sensitivity_map)

        acquisitions, flat_prob_mask = self(mask, image, masked_kspace, padding)
        if self.PolicySamplingDimension.ONE_D:
            # BCHW --> BW --> B11W1
            acquisitions = acquisitions.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
        else:
            # BCHW --> BH*W --> B1HW1
            acquisitions = acquisitions.reshape(batch_size, 1, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, height, width, 1)

        # B11W1 or B1HW1
        mask = mask + acquisitions

        # BMHWC
        with torch.no_grad():
            masked_kspace = mask * kspace

        if self.sampler_detach_mask:
            mask = mask.detach()
        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, prob_mask


class StraightThroughPolicy3dBlock(StraightThroughPolicyBlock):
    """
    Straight through policy model block for 3D input.
    """

    def __init__(
        self,
        budget: int,
        backward_operator: Callable,
        kspace_shape: tuple[int, int],
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        sampler_slope: float = 10,
        sampler_use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKYRELU,
        sampler_cwn_conv: bool = False,
    ):
        super().__init__(
            budget=budget,
            backward_operator=backward_operator,
            kspace_shape=kspace_shape,
            sampler_detach_mask=sampler_detach_mask,
            kspace_sampler=kspace_sampler,
            st_slope=st_slope,
            st_clamp=st_clamp,
            fix_sign_leakage=fix_sign_leakage,
            sampler_chans=sampler_chans,
            sampler_num_pool_layers=sampler_num_pool_layers,
            sampler_fc_size=sampler_fc_size,
            sampler_drop_prob=sampler_drop_prob,
            sampler_slope=sampler_slope,
            sampler_use_softplus=sampler_use_softplus,
            sampler_num_fc_layers=sampler_num_fc_layers,
            sampler_activation=sampler_activation,
            sampler_cwn_conv=sampler_cwn_conv,
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
        padding: Optional[torch.Tensor] = None,
    ):
        masked_kspace = self.pad_slc_dimension(masked_kspace)
        sensitivity_map = self.pad_slc_dimension(sensitivity_map)

        batch_size, _, slc, height, width, _ = masked_kspace.shape  # batch, coils, height, width, complex
        if self.PolicySamplingDimension.ONE_D:
            mask = mask[:, :, :, 0, :, :].reshape(batch_size, width)
            if padding:
                # This assumes padding has same pattern as mask
                padding = padding[:, :, :, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask.reshape(batch_size, height * width)
            if padding:
                padding = padding.reshape(batch_size, height * width)

        # BMHWC --> BCHW
        image = self.sens_reduce(masked_kspace, sensitivity_map)

        acquisitions, flat_prob_mask = self(mask, image, masked_kspace, padding)
        if self.PolicySamplingDimension.ONE_D:
            # BC1HW --> BW --> B111W1
            acquisitions = acquisitions.reshape(batch_size, 1, 1, 1, width, 1).expand(
                batch_size, 1, 1, height, width, 1
            )
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, 1, width, 1).expand(
                batch_size, 1, 1, height, width, 1
            )
            mask = mask.reshape(batch_size, 1, 1, 1, width, 1).expand(batch_size, 1, 1, height, width, 1)
        else:
            # BC1HW --> BH*W --> B11HW1
            acquisitions = acquisitions.reshape(batch_size, 1, 1, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, 1, height, width, 1)

        # B11HW1
        mask = mask + acquisitions

        # BMHWC
        with torch.no_grad():
            masked_kspace = mask * kspace

        if self.sampler_detach_mask:
            mask = mask.detach()
        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, prob_mask

    def pad_slc_dimension(self, kspace: torch.Tensor) -> torch.Tensor:
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


class StraightThroughPolicyDynamic2dBlock(StraightThroughPolicyBlock):
    """
    Straight through policy model block for 3D input.
    """

    def __init__(
        self,
        budget: int,
        backward_operator: Callable,
        kspace_shape: tuple[int, int],
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        sampler_slope: float = 10,
        sampler_use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKYRELU,
        sampler_cwn_conv: bool = False,
    ):
        super().__init__(
            budget=budget,
            backward_operator=backward_operator,
            kspace_shape=kspace_shape,
            sampler_detach_mask=sampler_detach_mask,
            kspace_sampler=kspace_sampler,
            st_slope=st_slope,
            st_clamp=st_clamp,
            fix_sign_leakage=fix_sign_leakage,
            sampler_chans=sampler_chans,
            sampler_num_pool_layers=sampler_num_pool_layers,
            sampler_fc_size=sampler_fc_size,
            sampler_drop_prob=sampler_drop_prob,
            sampler_slope=sampler_slope,
            sampler_use_softplus=sampler_use_softplus,
            sampler_num_fc_layers=sampler_num_fc_layers,
            sampler_activation=sampler_activation,
            sampler_cwn_conv=sampler_cwn_conv,
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
        padding: Optional[torch.Tensor] = None,
    ):
        masked_kspace = self.pad_time_dimension(masked_kspace)
        sensitivity_map = self.pad_time_dimension(sensitivity_map)

        batch_size, _, time, height, width, _ = masked_kspace.shape  # batch, coils, time, height, width, complex

        if self.PolicySamplingDimension.ONE_D:
            mask = mask[:, :, :, 0, :, :].reshape(batch_size, width)
            if padding:
                # This assumes padding has same pattern as mask
                padding = padding[:, :, :, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask.reshape(batch_size, height * width)
            if padding:
                padding = padding.reshape(batch_size, height * width)

        # BMHWC --> BCHW
        image = self.sens_reduce(masked_kspace, sensitivity_map)

        acquisitions, flat_prob_mask = self(mask, image, masked_kspace, padding)

        if self.PolicySamplingDimension.ONE_D:
            acquisitions = acquisitions.reshape(batch_size, 1, time, 1, width, 1).expand(
                batch_size, 1, time, height, width, 1
            )
            prob_mask = flat_prob_mask.reshape(batch_size, 1, time, 1, width, 1).expand(
                batch_size, 1, time, height, width, 1
            )
            mask = mask.reshape(batch_size, 1, time, 1, width, 1).expand(batch_size, 1, time, height, width, 1)
        else:
            acquisitions = acquisitions.reshape(batch_size, 1, time, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, time, height, width, 1)
            mask = mask.reshape(batch_size, 1, time, height, width, 1)

        # B11HW1
        mask = mask + acquisitions

        # BMHWC
        with torch.no_grad():
            masked_kspace = mask * kspace

        if self.sampler_detach_mask:
            mask = mask.detach()
        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, prob_mask

    def pad_time_dimension(self, kspace: torch.Tensor) -> torch.Tensor:
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


class StraightThroughPolicy(nn.Module):
    """Multi layer Straight through policy model."""

    def __init__(
        self,
        acceleration: float,
        center_fraction: float,
        backward_operator: Callable,
        kspace_shape: tuple[int, int],
        num_layers: int = 1,
        sampling_dimension: PolicySamplingDimension = PolicySamplingDimension.ONE_D,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        sampler_chans: int = 16,
        sampler_num_pool_layers: int = 4,
        sampler_fc_size: int = 256,
        sampler_drop_prob: float = 0,
        sampler_slope: float = 10,
        sampler_use_softplus: bool = True,
        sampler_num_fc_layers: int = 3,
        sampler_activation: ActivationType = ActivationType.LEAKYRELU,
        sampler_cwn_conv: bool = False,
        sampling_type: PolicySamplingType = PolicySamplingType.NON_DYNAMIC,
        num_time_steps: Optional[int] = None,
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

        if sampling_type == PolicySamplingType.DYNAMIC:
            if len(kspace_shape) == 3:
                raise NotImplementedError(f"Dynamic sampling is only implemented for 2D data.")
            self.num_time_steps = num_time_steps
            kspace_shape = (num_time_steps, *kspace_shape)
            self.num_actions *= num_time_steps

        self.kspace_shape = kspace_shape
        self.sampling_dimension = sampling_dimension
        self.sampling_type = sampling_type

        budget = int(self.num_actions / acceleration - self.num_actions * center_fraction)
        layer_budget = budget // num_layers

        self.layers = nn.ModuleList()
        st_policy_block = StraightThroughPolicy2dBlock if len(kspace_shape) == 2 else StraightThroughPolicy3dBlock
        for i in range(num_layers):
            if i == (num_layers - 1):
                layer_budget = budget - (num_layers - 1) * layer_budget

            self.layers.append(
                st_policy_block(
                    budget=layer_budget,
                    backward_operator=backward_operator,
                    kspace_shape=kspace_shape,
                    sampler_detach_mask=sampler_detach_mask,
                    kspace_sampler=kspace_sampler,
                    st_slope=st_slope,
                    st_clamp=st_clamp,
                    fix_sign_leakage=fix_sign_leakage,
                    sampler_chans=sampler_chans,
                    sampler_num_pool_layers=sampler_num_pool_layers,
                    sampler_fc_size=sampler_fc_size,
                    sampler_drop_prob=sampler_drop_prob,
                    sampler_slope=sampler_slope,
                    sampler_use_softplus=sampler_use_softplus,
                    sampler_num_fc_layers=sampler_num_fc_layers,
                    sampler_activation=sampler_activation,
                    sampler_cwn_conv=sampler_cwn_conv,
                )
            )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        kspace: torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
        masks = [mask]
        prob_masks = []

        for _, layer in enumerate(self.layers):
            mask, masked_kspace, prob_mask = layer.do_acquisition(
                kspace, masked_kspace, mask, sensitivity_map, padding
            )
            masks.append(mask)
            prob_masks.append(prob_mask)

        return masked_kspace, masks, prob_masks
