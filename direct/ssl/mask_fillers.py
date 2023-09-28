# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import torch

from direct.ssl._gaussian_fill import gaussian_fill as _gaussian_fill

__all__ = ["gaussian_fill", "uniform_fill"]


def gaussian_fill(
    nonzero_mask_count: int,
    nrow: int,
    ncol: int,
    center_x: int,
    center_y: int,
    std_scale: float,
    mask: np.ndarray,
    output_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generates a binary mask filled with randomly sampled positions following a 2D Gaussian distribution.

    Makes a call to the cython function `_gaussian_fill`.

    Parameters
    ----------

    nonzero_mask_count : int
        Number of non-zero entries in the output mask.
    nrow : int
        Number of rows of the output mask.
    ncol : int
        Number of columns of the output mask.
    center_x : int
        X coordinate of the center of the Gaussian distribution.
    center_y : int
        Y coordinate of the center of the Gaussian distribution.
    std_scale : float
        Scaling factor for the standard deviation of the Gaussian distribution. The standard deviation of the Gaussian
        distribution will be (nrow-1)/std_scale and (ncol-1)/std_scale along the X and Y axes, respectively.
    mask : np.ndarray
        A binary integer 2D array representing the input mask.
    output_mask : np.ndarray
        A binary integer 2D array representing the output mask.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        A 2D array representing the output mask filled with randomly sampled positions following
        a 2D Gaussian distribution.
    """
    return _gaussian_fill(nonzero_mask_count, nrow, ncol, center_x, center_y, std_scale, mask, output_mask, seed)


def uniform_fill(
    nonzero_mask_count: int, nrow: int, ncol: int, mask: torch.Tensor, rng: np.random.RandomState
) -> torch.Tensor:
    """Fills a binary `torch.Tensor` mask with the specified number of ones in a uniform random manner.

    Parameters
    ----------
    nonzero_mask_count : int
        The number of 1s to place in the mask.
    nrow : int
        The number of rows in the mask.
    ncol : int
        The number of columns in the mask.
    mask : torch.Tensor
        A binary mask with zeros and ones.
    rng : np.random.RandomState
        A NumPy random state object for reproducibility.

    Returns
    -------
    torch.Tensor
        A binary mask with the specified number of 1s placed in a uniform random manner.
    """
    prob = mask.flatten().numpy()
    ind_flattened = rng.choice(
        torch.arange(nrow * ncol),
        size=nonzero_mask_count,
        replace=False,
        p=prob / prob.sum(),
    )
    (ind_x, ind_y) = np.unravel_index(ind_flattened, (nrow, ncol))

    output_mask = torch.zeros_like(mask, dtype=mask.dtype)
    output_mask[ind_x, ind_y] = True

    return output_mask
