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
"""Tests for the direct.algorithms module."""

import pytest
import torch

from direct.algorithms.optimization import MaximumEigenvaluePowerMethod


@pytest.mark.parametrize("size", [20, 30])
def test_power_method(size):
    mat = torch.rand((size, size)) + torch.rand((size, size)) * 1j
    x0 = torch.ones(size) + 0 * 1j

    def A(x):
        return mat @ x

    algo = MaximumEigenvaluePowerMethod(A)
    algo.fit(x0)

    all_eigenvalues = torch.linalg.eig(mat).eigenvalues
    max_eig_torch = all_eigenvalues[all_eigenvalues.abs().argmax()]

    assert torch.allclose(algo.max_eig, max_eig_torch, 0.001)
