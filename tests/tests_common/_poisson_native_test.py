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
"""Smoke tests for the nanobind `_poisson` extension."""

from __future__ import annotations

import numpy as np

from direct.common._poisson import poisson  # pylint: disable=no-name-in-module


def _radius_grid(nx: int, ny: int, r: float) -> tuple[np.ndarray, np.ndarray]:
    radius_x = np.full((nx, ny), r, dtype=np.float64)
    radius_y = np.full((nx, ny), r, dtype=np.float64)
    return radius_x, radius_y


def test_poisson_produces_at_least_one_sample() -> None:
    nx, ny = 32, 32
    mask = np.zeros((nx, ny), dtype=np.int64)
    rx, ry = _radius_grid(nx, ny, 4.0)

    poisson(nx, ny, max_attempts=30, mask=mask, radius_x=rx, radius_y=ry, seed=2024)

    rows, cols = np.nonzero(mask)
    assert mask.sum() >= 1
    assert np.all((rows >= 0) & (rows < nx))
    assert np.all((cols >= 0) & (cols < ny))


def test_poisson_seed_determinism() -> None:
    nx, ny = 24, 24
    rx, ry = _radius_grid(nx, ny, 3.0)

    a = np.zeros((nx, ny), dtype=np.int64)
    b = np.zeros((nx, ny), dtype=np.int64)
    poisson(nx, ny, 30, a, rx, ry, 11)
    poisson(nx, ny, 30, b, rx, ry, 11)
    assert np.array_equal(a, b)


def test_poisson_approximate_minimum_distance() -> None:
    """Stored mask cells should respect the local radius up to the truncation slack.

    The algorithm performs rejection in continuous (float) space but stores
    each accepted candidate at the integer truncation of its float position,
    so two stored cells can sit at most ``sqrt(2)`` units closer than the
    radius. We assert the relaxed bound here.
    """
    nx, ny = 40, 40
    radius = 3.5
    rx, ry = _radius_grid(nx, ny, radius)
    mask = np.zeros((nx, ny), dtype=np.int64)

    poisson(nx, ny, 30, mask, rx, ry, 7)
    pts = np.column_stack(np.nonzero(mask)).astype(np.float64)
    tolerance = np.sqrt(2.0)

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            assert d >= radius - tolerance, f"points {pts[i]} and {pts[j]} are closer than {radius - tolerance:.3f}"
