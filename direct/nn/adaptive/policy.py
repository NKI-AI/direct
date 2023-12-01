# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.policy module."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import direct.data.transforms as T
from direct.nn.adaptive.binarizer import ThresholdSigmoidMask
from direct.nn.adaptive.sampler import ImageLineConvSampler, KSpaceLineConvSampler
from direct.nn.types import ActivationType

__all__ = ["LOUPEPolicy", "MultiStraightThroughPolicy"]


class LOUPEPolicy(nn.Module):
    """LOUPE policy model."""

    def __init__(
        self,
        acceleration: float,
        center_fraction: float,
        kspace_shape: tuple[int, ...],
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
    ):
        super().__init__()
        if len(kspace_shape) not in [1, 2]:
            raise ValueError(
                f"Input dimension of LOUPEPolicy should have length of 1 or 2. Received `input_dim`={kspace_shape}."
            )
        self.sampling_ndim = len(kspace_shape)  # dimension of sampling
        num_actions = np.prod(kspace_shape)
        self.kspace_shape = kspace_shape
        self.use_softplus = use_softplus
        self.slope = slope
        self.st_slope = st_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp

        if use_softplus:
            # Softplus will be applied
            self.sampler = nn.Parameter(torch.normal(torch.ones((1, num_actions)), torch.ones((1, num_actions)) / 10))
        else:
            # Sigmoid will be applied
            self.sampler = nn.Parameter(torch.zeros((1, num_actions)))

        budget = int(num_actions / acceleration - num_actions * center_fraction)

        self.binarizer = ThresholdSigmoidMask(self.st_slope, self.st_clamp)

        self.budget = budget

    def forward(self, mask: torch.Tensor, kspace: torch.Tensor, padding: Optional[torch.Tensor] = None):
        batch_size, _, height, width, _ = kspace.shape  # batch, coils, height, width, complex
        masks = [mask]
        if self.sampling_ndim == 1:
            mask = mask[:, :, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask.reshape(batch_size, height * width)

        # Reshape to [B, num_actions]
        sampler_out = self.sampler.expand(batch_size, -1)
        if self.use_softplus:
            # Softplus to make positive
            prob_mask = F.softplus(sampler_out, beta=self.slope)
            prob_mask = prob_mask / torch.max(
                (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1])) * prob_mask,
                dim=1,
            )[0].reshape(-1, 1)
        else:
            # Sigmoid to make positive
            prob_mask = torch.sigmoid(self.slope * sampler_out)
        # Mask out already sampled rows
        masked_prob_mask = prob_mask * (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1]))
        # Mask out padded areas
        if padding is not None:
            if self.sampling_ndim == 1:
                padding = padding[:, :, 0, :, :].reshape(batch_size, width)
            else:
                padding = padding.reshape(batch_size, height * width)
            masked_prob_mask = masked_prob_mask * (1 - padding)
        # Take out zero (masked) probabilities, since we don't want to include those in the normalisation
        nonzero_idcs = (mask == 0).nonzero(as_tuple=True)
        probs_to_norm = masked_prob_mask[nonzero_idcs].reshape(batch_size, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = rescale_probs(probs_to_norm, self.budget)
        # Reassign to original array
        masked_prob_mask[nonzero_idcs] = normed_probs.flatten()

        # Binarize the mask
        flat_bin_mask = self.binarizer(masked_prob_mask)

        if self.sampling_ndim == 1:
            # BCHW --> BW --> B11W1 --> B1HW1
            acquisitions = flat_bin_mask.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
            final_prob_mask = masked_prob_mask.reshape(batch_size, 1, 1, width, 1).expand(
                batch_size, 1, height, width, 1
            )
            mask = mask.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
        else:
            # BCHW --> BH*W --> B1HW1
            acquisitions = flat_bin_mask.reshape(batch_size, 1, height, width, 1)
            final_prob_mask = masked_prob_mask.reshape(batch_size, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, height, width, 1)
        mask = mask + acquisitions
        masks.append(mask)
        # BMHWC
        with torch.no_grad():
            masked_kspace = mask * kspace

        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return masked_kspace, masks, [final_prob_mask]


class LOUPE3dPolicy(nn.Module):
    """LOUPE policy model."""

    def __init__(
        self,
        acceleration: float,
        center_fraction: float,
        kspace_shape: tuple[int, ...],
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
    ):
        super().__init__()
        if len(kspace_shape) not in [1, 2]:
            raise ValueError(
                f"Input dimension of LOUPEPolicy should have lenght of 1 or 2. Received `input_dim`={kspace_shape}."
            )
        self.sampling_ndim = len(kspace_shape)
        num_actions = np.prod(kspace_shape)
        self.kspace_shape = kspace_shape
        self.use_softplus = use_softplus
        self.slope = slope
        self.st_slope = st_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp

        if use_softplus:
            # Softplus will be applied
            self.sampler = nn.Parameter(torch.normal(torch.ones((1, num_actions)), torch.ones((1, num_actions)) / 10))
        else:
            # Sigmoid will be applied
            self.sampler = nn.Parameter(torch.zeros((1, num_actions)))

        budget = int(num_actions / acceleration - num_actions * center_fraction)

        self.binarizer = ThresholdSigmoidMask(self.st_slope, self.st_clamp)

        self.budget = budget

    def forward(self, mask: torch.Tensor, kspace: torch.Tensor, padding: Optional[torch.Tensor] = None):
        batch_size, _, slc, height, width, _ = kspace.shape  # batch, coils, slice, height, width, complex
        masks = [mask]
        if self.sampling_ndim == 1:
            mask = mask[:, :, 0, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask[:, :, 0].reshape(batch_size, height * width)

        sampler_out = self.sampler.expand(batch_size, -1)
        if self.use_softplus:
            # Softplus to make positive
            prob_mask = F.softplus(sampler_out, beta=self.slope)
            prob_mask = prob_mask / torch.max(
                (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1])) * prob_mask,
                dim=1,
            )[0].reshape(-1, 1)
        else:
            # Sigmoid to make positive
            prob_mask = torch.sigmoid(self.slope * sampler_out)
        # Mask out already sampled rows
        masked_prob_mask = prob_mask * (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1]))
        # Mask out padded areas
        if padding is not None:
            if self.sampling_ndim == 1:
                padding = padding[:, :, 0, 0, :, :].reshape(batch_size, width)
            else:
                padding = padding[:, :, 0].reshape(batch_size, height * width)
            masked_prob_mask = masked_prob_mask * (1 - padding)
        # Take out zero (masked) probabilities, since we don't want to include those in the normalisation
        nonzero_idcs = (mask == 0).nonzero(as_tuple=True)
        probs_to_norm = masked_prob_mask[nonzero_idcs].reshape(batch_size, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = rescale_probs(probs_to_norm, self.budget)
        # Reassign to original array
        masked_prob_mask[nonzero_idcs] = normed_probs.flatten()

        # Binarize the mask
        flat_bin_mask = self.binarizer(masked_prob_mask)

        if self.sampling_ndim == 1:
            # BCSHW --> BW --> B111W1 --> B1SHW1
            acquisitions = flat_bin_mask.reshape(batch_size, 1, 1, 1, width, 1).expand(
                batch_size, 1, 1, height, width, 1
            )
            final_prob_mask = masked_prob_mask.reshape(batch_size, 1, 1, 1, width, 1).expand(
                batch_size, 1, 1, height, width, 1
            )
            mask = mask.reshape(batch_size, 1, 1, 1, width, 1).expand(batch_size, 1, 1, height, width, 1)
        else:
            # BCSHW --> BH*W --> B1SHW1
            acquisitions = flat_bin_mask.reshape(batch_size, 1, 1, height, width, 1)
            final_prob_mask = masked_prob_mask.reshape(batch_size, 1, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, 1, height, width, 1)

        mask = mask + acquisitions
        masks.append(mask)
        # BMHWC
        with torch.no_grad():
            masked_kspace = mask * kspace

        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return masked_kspace, masks, [final_prob_mask]


class LOUPEDynamicPolicy(nn.Module):
    """LOUPE policy model."""

    def __init__(
        self,
        acceleration: float,
        center_fraction: float,
        kspace_shape: tuple[int, ...],
        time_steps: int,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
    ):
        super().__init__()
        if len(kspace_shape) not in [1, 2]:
            raise ValueError(
                f"Input dimension of LOUPEPolicy should have lenght of 1 or 2. Received `input_dim`={kspace_shape}."
            )
        self.sampling_ndim = len(kspace_shape)
        num_actions = np.prod(kspace_shape)
        self.kspace_shape = kspace_shape
        self.time_steps = time_steps

        self.use_softplus = use_softplus
        self.slope = slope
        self.st_slope = st_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp

        if use_softplus:
            # Softplus will be applied
            self.sampler = nn.Parameter(
                torch.normal(torch.ones((1, time_steps, num_actions)), torch.ones((1, time_steps, num_actions)) / 10)
            )
        else:
            # Sigmoid will be applied
            self.sampler = nn.Parameter(torch.zeros((1, time_steps, num_actions)))

        budget = int(num_actions / acceleration - num_actions * center_fraction)

        self.binarizer = ThresholdSigmoidMask(self.st_slope, self.st_clamp)

        self.budget = budget

    def forward(self, mask: torch.Tensor, kspace: torch.Tensor, padding: Optional[torch.Tensor] = None):
        batch_size, _, steps, height, width, _ = kspace.shape  # batch, coils, slice, height, width, complex
        masks = [mask.expand(batch_size, 1, steps, height, width, 1)]

        out_mask = []
        final_prob_mask = []
        for _ in range(steps):
            if self.sampling_ndim == 1:
                mask_step = mask[:, :, 0, 0, :, :].reshape(batch_size, width)
            else:
                mask_step = mask[:, :, 0].reshape(batch_size, height * width)

            sampler_out = self.sampler[:, _].expand(batch_size, -1)
            if self.use_softplus:
                # Softplus to make positive
                prob_mask = F.softplus(sampler_out, beta=self.slope)
                prob_mask = prob_mask / torch.max(
                    (1 - mask_step.reshape(prob_mask.shape[0], prob_mask.shape[1])) * prob_mask,
                    dim=1,
                )[0].reshape(-1, 1)
            else:
                # Sigmoid to make positive
                prob_mask = torch.sigmoid(self.slope * sampler_out)
            # Mask out already sampled rows
            masked_prob_mask = prob_mask * (1 - mask_step.reshape(prob_mask.shape[0], prob_mask.shape[1]))
            # Mask out padded areas
            if padding is not None:
                if self.sampling_ndim == 1:
                    padding = padding[:, :, 0, 0, :, :].reshape(batch_size, width)
                else:
                    padding = padding[:, :, 0].reshape(batch_size, height * width)
                masked_prob_mask = masked_prob_mask * (1 - padding)
            # Take out zero (masked) probabilities, since we don't want to include those in the normalisation
            nonzero_idcs = (mask_step == 0).nonzero(as_tuple=True)
            probs_to_norm = masked_prob_mask[nonzero_idcs].reshape(batch_size, -1)
            # Rescale probabilities to desired sparsity.
            normed_probs = rescale_probs(probs_to_norm, self.budget)
            # Reassign to original array
            masked_prob_mask[nonzero_idcs] = normed_probs.flatten()

            # Binarize the mask
            flat_bin_mask = self.binarizer(masked_prob_mask)

            if self.sampling_ndim == 1:
                # BCSHW --> BW --> B111W1 --> B1SHW1
                acquisitions = flat_bin_mask.reshape(batch_size, 1, 1, width, 1).expand(
                    batch_size, 1, height, width, 1
                )
                final_prob_mask = masked_prob_mask.reshape(batch_size, 1, 1, width, 1).expand(
                    batch_size, 1, height, width, 1
                )
                mask_step = mask_step.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
            else:
                # BCSHW --> BH*W --> B1SHW1
                acquisitions = flat_bin_mask.reshape(batch_size, 1, height, width, 1)
                final_prob_mask = masked_prob_mask.reshape(batch_size, 1, height, width, 1)
                mask_step = mask_step.reshape(batch_size, 1, height, width, 1)

            mask_step = mask_step + acquisitions
            out_mask.append(mask_step)
        out_mask = torch.stack(out_mask, 2)
        masks.append(out_mask)
        # BMHWC
        with torch.no_grad():
            masked_kspace = out_mask * kspace

        # Note that since masked_kspace = out_mask * kspace, this masked_kspace will leak sign information
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, out_mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return masked_kspace, masks, [final_prob_mask]


class StraightThroughPolicy(nn.Module):
    """
    Straight through policy model block.
    """

    def __init__(
        self,
        budget: int,
        backward_operator: Callable,
        kspace_shape: tuple[int, int] = (128, 128),
        slope: float = 10,
        sampler_detach_mask: bool = False,
        kspace_sampler: bool = False,
        use_softplus: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        fc_size: int = 256,
        drop_prob: float = 0.0,
        num_fc_layers: int = 3,
        activation: ActivationType = ActivationType.LEAKYRELU,
    ):
        super().__init__()

        self.sampler = (KSpaceLineConvSampler if kspace_sampler else ImageLineConvSampler)(
            input_dim=(2, *kspace_shape),
            slope=slope,
            use_softplus=use_softplus,
            fc_size=fc_size,
            num_fc_layers=num_fc_layers,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.kspace_sampler = kspace_sampler

        self.binarizer = ThresholdSigmoidMask(st_slope, st_clamp)

        self.slope = slope
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask
        self.use_softplus = use_softplus
        self.fix_sign_leakage = fix_sign_leakage
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.num_fc_layers = num_fc_layers
        self.activation = activation

        self.backward_operator = backward_operator

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        masked_kspace: torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
        batch_size, _, _, width = image.shape

        flat_prob_mask = self.sampler(masked_kspace.permute(0, 1, 4, 2, 3) if self.kspace_sampler else image, mask)

        # Mask out padded areas
        if padding is not None:
            mask = mask * (1 - padding.reshape(*mask.shape))
        # Take out zero (masked) probabilities, since we don't want to include those in the normalisation
        nonzero_idcs = (mask.view(batch_size, width) == 0).nonzero(as_tuple=True)
        probs_to_norm = flat_prob_mask[nonzero_idcs].reshape(batch_size, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = rescale_probs(probs_to_norm, self.budget)
        # Reassign to original array
        flat_prob_mask[nonzero_idcs] = normed_probs.flatten()
        # Binarize the mask
        flat_bin_mask = self.binarizer(flat_prob_mask)
        return flat_bin_mask, flat_prob_mask

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ):
        batch_size, _, _, width, _ = kspace.shape  # batch, coils, height, width, complex
        # BMHWC --> BHWC --> BCHW
        current_recon = self.sens_reduce(masked_kspace, sensitivity_map).permute(0, 3, 1, 2)

        # BCHW --> BW --> B11W1
        acquisitions, flat_prob_mask = self(current_recon, mask, masked_kspace, padding)
        acquisitions = acquisitions.reshape(batch_size, 1, 1, width, 1)
        prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, width, 1)

        # B11W1
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

    def sens_reduce(self, x: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        x = self.backward_operator(x, dim=(2, 3))
        return T.reduce_operator(x, sensitivity_map, 1)


class MultiStraightThroughPolicy(nn.Module):
    """Multi layer Straight through policy model."""

    def __init__(
        self,
        acceleration: float,
        center_fraction: float,
        backward_operator: Callable,
        num_layers: int = 1,
        kspace_shape: tuple[int, int] = (128, 128),
        slope: float = 10,
        kspace_sampler: bool = False,
        sampler_detach_mask: bool = False,
        use_softplus: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        fix_sign_leakage: bool = True,
        fc_size: int = 256,
        drop_prob: float = 0.0,
        num_fc_layers: int = 3,
        activation: ActivationType = ActivationType.LEAKYRELU,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        num_cols = kspace_shape[-1]
        budget = int(num_cols / acceleration - num_cols * center_fraction)
        layer_budget = budget // num_layers

        for i in range(num_layers):
            if i == (num_layers - 1):
                layer_budget = budget - (num_layers - 1) * layer_budget

            self.layers.append(
                StraightThroughPolicy(
                    budget=layer_budget,
                    backward_operator=backward_operator,
                    kspace_shape=kspace_shape,
                    slope=slope,
                    sampler_detach_mask=sampler_detach_mask,
                    kspace_sampler=kspace_sampler,
                    use_softplus=use_softplus,
                    st_slope=st_slope,
                    st_clamp=st_clamp,
                    fix_sign_leakage=fix_sign_leakage,
                    fc_size=fc_size,
                    drop_prob=drop_prob,
                    num_fc_layers=num_fc_layers,
                    activation=activation,
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
        batch_size, _, height, width, _ = kspace.shape  # batch, coils, height, width, complex
        mask = mask[:, :, 0, :, :].reshape(batch_size, 1, 1, width, 1)
        masks = [mask]
        prob_masks = []
        if padding is not None:
            padding = padding[:, :, 0, :, :].reshape(batch_size, 1, 1, width, 1)

        for _, layer in enumerate(self.layers):
            mask, masked_kspace, prob_mask = layer.do_acquisition(
                kspace, masked_kspace, mask, sensitivity_map, padding
            )
            masks.append(mask.expand(batch_size, 1, height, width, 1))
            prob_masks.append(prob_mask.expand(batch_size, 1, height, width, 1))

        return masked_kspace, masks, prob_masks


def rescale_probs(batch_x: torch.Tensor, budget: int):
    """
    Rescale Probability Map
    given a prob map x, rescales it so that it obtains the desired sparsity,
    specified by budget and the image size.

    if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    if mean(x) < sparsity, one can basically do the same thing by rescaling
                            (1-x) appropriately, then taking 1 minus the result.

    Parameters
    ----------
    batch_x : torch.Tensor
        Input batch of probabilities.
    budget : int
        Number of budget lines.

    Returns
    -------
    torch.Tensor
        Rescaled probabilities.
    """

    batch_size, width = batch_x.shape
    sparsity = budget / width
    ret = []
    for i in range(batch_size):
        x = batch_x[i : i + 1]
        xbar = torch.mean(x)
        r = sparsity / xbar
        beta = (1 - sparsity) / (1 - xbar)

        # compute adjustment
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)
