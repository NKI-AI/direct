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

"""Visualization helpers for registration / displacement fields."""

import numpy as np
import torch
from scipy.ndimage import map_coordinates

# Matplotlib plasma control points (RGB in [0, 1]).
_PLASMA_STOPS = np.asarray(
    [
        [0.050383, 0.029803, 0.527975],
        [0.254971, 0.013666, 0.615419],
        [0.417642, 0.000564, 0.658390],
        [0.563812, 0.031785, 0.643928],
        [0.691949, 0.112549, 0.571812],
        [0.798216, 0.219943, 0.479208],
        [0.877402, 0.340904, 0.379832],
        [0.935556, 0.467295, 0.275687],
        [0.973833, 0.598533, 0.164697],
        [0.988362, 0.743458, 0.078403],
        [0.940015, 0.975158, 0.131326],
    ],
    dtype=np.float64,
)


def _plasma_lut(n: int = 256) -> np.ndarray:
    """Build a discrete plasma colormap lookup table.

    Args:
        n: Number of discrete color entries. Default is ``256``.

    Returns:
        Lookup table of shape ``(n, 3)`` with RGB values in ``[0, 1]``.
    """
    xs = np.linspace(0.0, 1.0, len(_PLASMA_STOPS))
    x = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x, xs, _PLASMA_STOPS[:, i]) for i in range(3)], axis=1)


_PLASMA_LUT = _plasma_lut(256)


def apply_plasma(values: np.ndarray) -> np.ndarray:
    """Map scalar values in ``[0, 1]`` to plasma RGB.

    Args:
        values: Scalar values to colorize.

    Returns:
        RGB colors of shape ``values.shape + (3,)`` with values in ``[0, 1]``.
    """
    idx = np.clip((values * 255.0).astype(np.int64), 0, 255)
    return _PLASMA_LUT[idx]


def displacement_field_to_warped_grid(
    displacement_field: torch.Tensor,
    *,
    spacing: int = 12,
    background: tuple[float, float, float] = (0.05, 0.0, 0.08),
) -> torch.Tensor:
    """Render a displacement field as a plasma-colored warped grid.

    Args:
        displacement_field: Displacement of shape ``(2, height, width)`` with ``(dx, dy)`` channels in pixel units (image
            domain).
        spacing: Grid line spacing in pixels. Default is ````12`` (coarse grid)``.
        background: RGB background for zero-grid regions.

    Returns:
        RGB image of shape ``(3, height, width)`` in ``[0, 1]``.
    """
    if displacement_field.ndim != 3 or displacement_field.shape[0] != 2:
        raise ValueError(f"Expected displacement_field of shape (2, H, W), got {tuple(displacement_field.shape)}.")
    if spacing < 1:
        raise ValueError(f"spacing must be >= 1, got {spacing}.")

    dx = displacement_field[0].detach().float().cpu().numpy()
    dy = displacement_field[1].detach().float().cpu().numpy()
    height, width = dx.shape

    grid = np.zeros((height, width), dtype=np.float32)
    grid[::spacing, :] = 1.0
    grid[:, ::spacing] = 1.0

    # Amplify small fields so the warp is visible; cap for stability.
    max_mag = float(np.sqrt(dx * dx + dy * dy).max())
    amp = min(max(5.0, 8.0 / max(max_mag, 1e-6)), 40.0)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    warped = map_coordinates(grid, [yy - dy * amp, xx - dx * amp], order=1, mode="constant")

    # Color warped lines with plasma; keep a dark background elsewhere.
    norm = warped / max(float(warped.max()), 1e-8)
    rgb = np.empty((height, width, 3), dtype=np.float64)
    rgb[:] = background
    mask = warped > 0.05
    rgb[mask] = apply_plasma(norm[mask])

    return torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).contiguous()
