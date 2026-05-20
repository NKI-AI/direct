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
"""Smoke tests for the nanobind `_gaussian_fill` extension."""

from __future__ import annotations

import numpy as np
from direct.ssl._gaussian_fill import gaussian_fill  # pylint: disable=no-name-in-module


def test_gaussian_fill_only_sets_positions_inside_input_mask() -> None:
    nrow, ncol = 64, 64
    nonzero_mask_count = 80

    rng = np.random.default_rng(42)
    mask = (rng.random((nrow, ncol)) < 0.6).astype(np.int64)
    mask[nrow // 2 - 8 : nrow // 2 + 8, ncol // 2 - 8 : ncol // 2 + 8] = 1
    output_mask = np.zeros_like(mask)

    returned = gaussian_fill(nonzero_mask_count, nrow, ncol, nrow // 2, ncol // 2, 4.0, mask, output_mask, 7)

    assert returned is output_mask
    assert output_mask.sum() == nonzero_mask_count + 1
    assert np.all((mask == 1) | (output_mask == 0))


def test_gaussian_fill_seed_determinism() -> None:
    nrow, ncol = 32, 32
    rng = np.random.default_rng(0)
    mask = (rng.random((nrow, ncol)) < 0.7).astype(np.int64)

    a = np.zeros_like(mask)
    b = np.zeros_like(mask)
    gaussian_fill(40, nrow, ncol, nrow // 2, ncol // 2, 4.0, mask, a, 13)
    gaussian_fill(40, nrow, ncol, nrow // 2, ncol // 2, 4.0, mask, b, 13)
    assert np.array_equal(a, b)
