# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.utils module."""

import torch


def rescale_probs(batch_x: torch.Tensor, budget: int):
    """Rescale Probability Map.

     Given a prob map x, rescales it so that it obtains the desired sparsity, specified by budget and the image size.

    * if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    * if mean(x) < sparsity, one can basically do the same thing by rescaling (1-x) appropriately,
    then taking 1 minus the result.

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
