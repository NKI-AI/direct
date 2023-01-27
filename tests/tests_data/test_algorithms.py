# coding=utf-8
# Copyright (c) DIRECT Contributors

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

    algo = MaximumEigenvaluePowerMethod(A, x0)
    algo()

    all_eigenvalues = torch.linalg.eig(mat).eigenvalues
    max_eig_torch = all_eigenvalues[all_eigenvalues.abs().argmax()]

    assert torch.allclose(algo.max_eig, max_eig_torch, 0.001)
