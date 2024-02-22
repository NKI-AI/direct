# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.utils module."""

from __future__ import annotations

import torch

from direct.nn.adaptive.types import PolicySamplingDimension
from direct.types import TensorOrNone


def reshape_acquisitions_post_sampling(
    sampling_dimension: PolicySamplingDimension,
    acquisitions: torch.Tensor,
    flat_prob_mask: torch.Tensor,
    mask: torch.Tensor,
    shape: tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            prob_mask = flat_prob_mask.reshape(batch_size, 1, 1, 1, width, 1).expand(
                batch_size, 1, 1, height, width, 1
            )
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
    shape: tuple,
) -> tuple[torch.Tensor, TensorOrNone]:
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
            slc,
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


def rescale_probs(batch_x: torch.Tensor, budget: int | torch.Tensor):
    """Rescale Probability Map.

     Given a prob map x, rescales it so that it obtains the desired sparsity, specified by budget and the image size.

    * if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    * if mean(x) < sparsity, one can basically do the same thing by rescaling (1-x) appropriately,
    then taking 1 minus the result.

    Parameters
    ----------
    batch_x : torch.Tensor
        Input batch of probabilities.
    budget : int or torch.Tensor
        Number of budget lines.

    Returns
    -------
    torch.Tensor
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
        r = sparsity[i] / xbar
        beta = (1 - sparsity[i]) / (1 - xbar)

        # compute adjustment
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)


def normalize_masked_probabilities(
    mask: torch.Tensor, masked_prob_mask: torch.Tensor, budget: torch.Tensor
) -> torch.Tensor:
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
