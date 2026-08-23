# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
"""SSL Mask Fillers.

This module contains functions for splitting binary masks into ``(disjoint)`` subsets to be used for
self-supervised learning MRI reconstruction tasks.
"""

import numpy as np
import torch

from direct.ssl._gaussian_fill import gaussian_fill as _gaussian_fill  # pylint: disable=no-name-in-module

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

    Makes a call to the native ``(C++/nanobind)`` function `_gaussian_fill`.

    Args:
        nonzero_mask_count: Number of non-zero entries in the output mask.
        nrow: Number of rows of the output mask.
        ncol: Number of columns of the output mask.
        center_x: X coordinate of the center of the Gaussian distribution.
        center_y: Y coordinate of the center of the Gaussian distribution.
        std_scale: Scaling factor for the standard deviation of the Gaussian distribution. The standard deviation of the
            Gaussian distribution will be (nrow- ``1`` )/std_scale and (ncol- ``1`` )/std_scale along the X and Y axes,
            respectively.
        mask: A binary integer 2D array representing the input mask.
        output_mask: A binary integer 2D array representing the output mask.
        seed: Seed for the random number generator.

    Returns:
        A 2D array representing the output mask filled with randomly sampled positions following a 2D Gaussian
        distribution.
    """
    return _gaussian_fill(nonzero_mask_count, nrow, ncol, center_x, center_y, std_scale, mask, output_mask, seed)


def uniform_fill(
    nonzero_mask_count: int, nrow: int, ncol: int, mask: torch.Tensor, rng: np.random.RandomState
) -> torch.Tensor:
    """Fills a binary `torch.Tensor` mask with the specified number of ones in a uniform random manner.

    Args:
        nonzero_mask_count: The number of 1s to place in the mask.
        nrow: The number of rows in the mask.
        ncol: The number of columns in the mask.
        mask: A binary mask with zeros and ones.
        rng: A NumPy random state object for reproducibility.

    Returns:
        A binary mask with the specified number of 1s placed in a uniform random manner.
    """
    prob = mask.flatten().numpy()
    ind_flattened = rng.choice(
        torch.arange(nrow * ncol),
        size=nonzero_mask_count,
        replace=False,
        p=prob / prob.sum(),
    )
    (ind_x, ind_y) = np.unravel_index(ind_flattened, (nrow, ncol))  # pylint: disable=unbalanced-tuple-unpacking

    output_mask = torch.zeros_like(mask, dtype=mask.dtype)
    output_mask[ind_x, ind_y] = True

    return output_mask
