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

"""Utility functions for adaptive k-space sampling policies."""

from typing import Any

import torch

from direct.nn.adaptive.types import PolicySamplingDimension
from direct.types import TensorOrNone


def reshape_acquisitions_post_sampling(
    sampling_dimension: PolicySamplingDimension,
    acquisitions: torch.Tensor,
    flat_prob_mask: torch.Tensor,
    mask: torch.Tensor,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape flat acquisition tensors back to k-space layout after sampling.

    Args:
        sampling_dimension: Sampling dimension, either 1D (lines) or 2D (pixels).
        acquisitions: Flat acquisition mask of shape ``(batch, num_actions)``.
        flat_prob_mask: Flat probability mask of shape ``(batch, num_actions)``.
        mask: Flat or partially reshaped mask tensor.
        shape: Target k-space shape: 5D for 2D data or 6D for 3D data.

    Returns:
        Reshaped ``(acquisitions, prob_mask, mask)`` tensors.
    """
    if len(shape) == 5:
        batch_size, _, height, width, _ = shape
        if sampling_dimension == PolicySamplingDimension.ONE_D:
            acquisitions = acquisitions.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
        else:
            acquisitions = acquisitions.reshape(batch_size, 1, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, height, width, 1)
    elif len(shape) == 6:
        batch_size, _, _, height, width, _ = shape
        if sampling_dimension == PolicySamplingDimension.ONE_D:
            acquisitions = acquisitions.reshape(batch_size, 1, 1, 1, width, 1).expand(
                batch_size, 1, 1, height, width, 1
            )
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, 1, width, 1).expand(batch_size, 1, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, 1, 1, width, 1).expand(batch_size, 1, 1, height, width, 1)
        else:
            acquisitions = acquisitions.reshape(batch_size, 1, 1, height, width, 1)
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, height, width, 1)
            mask = mask.reshape(batch_size, 1, 1, height, width, 1)
    else:
        raise ValueError(
            f"Incorrect k-space shape. Should be a 5D tuple for 2D data or a 6D tuple for 3D data. "
            f"Received shape={shape}."
        )

    return acquisitions, prob_mask, mask


def reshape_mask_pre_sampling(
    sampling_dimension: PolicySamplingDimension,
    mask: torch.Tensor,
    padding: TensorOrNone,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, TensorOrNone]:
    """Flatten k-space masks to action vectors before adaptive sampling.

    Args:
        sampling_dimension: Sampling dimension, either 1D (lines) or 2D (pixels).
        mask: Sampling mask in k-space layout.
        padding: Optional padding mask in k-space layout.
        shape: K-space shape: 5D for 2D data or 6D for 3D data.

    Returns:
        Flattened ``(mask, padding)`` tensors of shape ``(batch, num_actions)``.
    """
    if len(shape) == 5:
        (
            batch_size,
            _,
            height,
            width,
            _,
        ) = shape  # [batch, coils, height, width, complex]

        # Reshape initial mask to [batch, num_actions]
        if sampling_dimension == PolicySamplingDimension.ONE_D:
            mask = mask[:, :, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask.reshape(batch_size, height * width)

        if padding is not None:
            if sampling_dimension == PolicySamplingDimension.ONE_D:
                padding = padding[:, :, 0, :, :].reshape(batch_size, width)
            else:
                padding = padding.reshape(batch_size, height * width)

    elif len(shape) == 6:
        (
            batch_size,
            _,
            _slc,
            height,
            width,
            _,
        ) = shape  # [batch, coils, slc, height, width, complex]

        # Reshape initial mask to [batch, num_actions]
        if sampling_dimension == PolicySamplingDimension.ONE_D:
            mask = mask[:, :, 0, 0, :, :].reshape(batch_size, width)
        else:
            mask = mask[:, :, 0].reshape(batch_size, height * width)

        if padding is not None:
            if sampling_dimension == PolicySamplingDimension.ONE_D:
                padding = padding[:, :, 0, 0, :, :].reshape(batch_size, width)
            else:
                padding = padding[:, :, 0].reshape(batch_size, height * width)
    else:
        raise ValueError(
            f"Incorrect k-space shape. Should be a 5D tuple for 2D data or a 6D tuple for 3D data. "
            f"Received shape={shape}."
        )

    return mask, padding


def rescale_probs(batch_x: torch.Tensor, budget: int | torch.Tensor) -> torch.Tensor:
    """Rescale Probability Map.

     Given a prob map x, rescales it so that it obtains the desired sparsity, specified by budget and the image size.

    * if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    * if mean(x) < sparsity, one can basically do the same thing by rescaling (1-x) appropriately,
    then taking 1 minus the result.

    Args:
        batch_x: Input batch of probabilities.
        budget: Number of budget lines.

    Returns:
        Rescaled probabilities.
    """

    batch_size, width = batch_x.shape

    sparsity = budget / width
    if isinstance(sparsity, float):
        sparsity = torch.tensor([sparsity] * batch_size)

    ret = []
    for i in range(batch_size):
        x = batch_x[i : i + 1]
        xbar = torch.mean(x)
        r = sparsity[i] / xbar  # ty: ignore[not-subscriptable]
        beta = (1 - sparsity[i]) / (1 - xbar)  # ty: ignore[not-subscriptable]

        # compute adjustment
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)


def normalize_masked_probabilities(
    mask: torch.Tensor, masked_prob_mask: torch.Tensor, budget: torch.Tensor
) -> torch.Tensor:  # ty: ignore[invalid-return-type]
    """Rescale masked probability maps to match the sampling budget per batch element.

    Args:
        mask: Binary mask indicating already sampled locations.
        masked_prob_mask: Probability map with sampled locations zeroed out.
        budget: Remaining sampling budget per batch element.

    Returns:
        Normalized probability map with the same shape as ``masked_prob_mask``.
    """
    # Have to iterate through batch as nonzero_idcs might defer across batch
    for batch_idx in range(mask.shape[0]):
        # Take out zero (masked) probabilities, since we don't want to include those in the normalisation
        nonzero_idcs = (mask[batch_idx] == 0).nonzero(as_tuple=True)
        probs_to_norm = masked_prob_mask[batch_idx][nonzero_idcs].reshape(1, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = rescale_probs(probs_to_norm, budget[batch_idx : batch_idx + 1])
        # Reassign to original tensor
        masked_prob_mask[batch_idx][nonzero_idcs] = normed_probs.flatten()

    return masked_prob_mask


def export_sampling_mask(data: dict[str, Any]) -> torch.Tensor | None:
    """Export sampling mask(s) for logging / inference without the complex dim.

    When adaptive sampling history is present (``data["masks"]``), returns a stack
    with shape ``(..., num_masks)`` where index ``0`` is the initial (ACS/init) mask
    and ``-1`` is the final predicted mask. Otherwise returns the single
    ``sampling_mask`` tensor.

    Args:
        data: Data.

    Returns:
        The result.
    """
    masks = data.get("masks")
    if masks:
        return torch.stack([m.detach().float().squeeze(-1) for m in masks], dim=-1)

    sampling_mask = data.get("sampling_mask")
    if sampling_mask is None:
        return None
    return sampling_mask.detach().float().squeeze(-1)


def sampling_mask_rgb_overlay(initial: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
    """Build an RGB overlay: blue = initial samples, red = newly predicted samples.

    Args:
        initial: Initial (ACS / init) mask, shape ``(*spatial,)`` or ``(1, *spatial)``.
        final: Final predicted mask, same shape as ``initial``.

    Returns:
        RGB image of shape ``(3, *spatial)``. Pure initial → blue, pure new → red, overlap → magenta.
    """
    initial = initial.detach().float().cpu()
    final = final.detach().float().cpu()
    if initial.ndim >= 1 and initial.shape[0] == 1:
        initial = initial.squeeze(0)
    if final.ndim >= 1 and final.shape[0] == 1:
        final = final.squeeze(0)

    initial = initial.clamp(0, 1)
    final = final.clamp(0, 1)
    acquired = (final - initial).clamp(min=0, max=1)

    rgb = torch.zeros(3, *final.shape, dtype=torch.float32)
    rgb[0] = acquired
    rgb[2] = initial
    return rgb


def split_sampling_mask_history(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Split a stacked ADS mask history into ``(initial, final)``.

    Expects ``mask`` shaped ``(..., num_masks)`` with ``num_masks >= 2``. Returns
    ``None`` when the tensor does not look like a history stack (caller should
    only invoke this when adaptive sampling produced ``data["masks"]``).

    Args:
        mask: Mask.

    Returns:
        The result.
    """
    if mask.ndim < 1 or mask.shape[-1] < 2:
        return None
    return mask[..., 0], mask[..., -1]
