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
"""Smoke tests for the nanobind `_gaussian` extension."""

from __future__ import annotations

import numpy as np
import pytest
from direct.common._gaussian import gaussian_mask_1d, gaussian_mask_2d  # pylint: disable=no-name-in-module


def test_gaussian_mask_1d_count_and_bounds() -> None:
    n = 256
    nonzero_count = 32
    mask = np.zeros(n, dtype=np.int64)

    gaussian_mask_1d(nonzero_count, n, n // 2, 6 * np.sqrt(n // 2), mask, 1234)

    set_idx = np.flatnonzero(mask == 1)
    assert mask.sum() == nonzero_count + 1
    assert np.all((set_idx >= 0) & (set_idx < n))


def test_gaussian_mask_1d_seed_determinism() -> None:
    n = 128
    a = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    gaussian_mask_1d(20, n, n // 2, 6 * np.sqrt(n // 2), a, 7)
    gaussian_mask_1d(20, n, n // 2, 6 * np.sqrt(n // 2), b, 7)
    assert np.array_equal(a, b)


def test_gaussian_mask_2d_count_and_bounds() -> None:
    nrow, ncol = 64, 64
    nonzero_count = 80
    mask = np.zeros((nrow, ncol), dtype=np.int64)
    std = 6 * np.array([np.sqrt(nrow // 2), np.sqrt(ncol // 2)], dtype=float)

    gaussian_mask_2d(nonzero_count, nrow, ncol, nrow // 2, ncol // 2, std, mask, 42)

    assert mask.sum() == nonzero_count + 1
    rows, cols = np.nonzero(mask)
    assert np.all((rows >= 0) & (rows < nrow))
    assert np.all((cols >= 0) & (cols < ncol))


def test_gaussian_mask_2d_seed_determinism() -> None:
    nrow, ncol = 32, 32
    std = 6 * np.array([np.sqrt(nrow // 2), np.sqrt(ncol // 2)], dtype=float)
    a = np.zeros((nrow, ncol), dtype=np.int64)
    b = np.zeros((nrow, ncol), dtype=np.int64)
    gaussian_mask_2d(50, nrow, ncol, nrow // 2, ncol // 2, std, a, 99)
    gaussian_mask_2d(50, nrow, ncol, nrow // 2, ncol // 2, std, b, 99)
    assert np.array_equal(a, b)


def test_gaussian_mask_1d_rejects_wrong_dtype() -> None:
    """Passing a float mask must fail loudly because of `noconvert()`."""
    mask = np.zeros(32, dtype=np.float64)
    with pytest.raises(TypeError):
        gaussian_mask_1d(5, 32, 16, 4.0, mask, 0)


def test_gaussian_mask_2d_rejects_wrong_shape() -> None:
    mask = np.zeros((8, 8), dtype=np.int64)
    std = np.array([2.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        gaussian_mask_2d(3, 4, 4, 2, 2, std, mask, 0)
