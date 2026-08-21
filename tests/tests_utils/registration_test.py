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
"""Tests for classical registration utilities."""

import pytest
import torch

from direct.registration.demons import (
    DemonsFilterType,
    create_demons_filter,
    multiscale_demons_displacement,
)
from direct.registration.optical_flow import (
    OpticalFlowEstimatorType,
    optical_flow_displacement,
)
from direct.registration.warp import create_grid, integrate_vector_field, warp, warp_tensor


def test_create_grid_shape() -> None:
    shape = torch.Size([2, 1, 8, 10])
    grid = create_grid(shape, torch.device("cpu"))
    assert grid.shape == (2, 2, 8, 10)


def test_warp_identity() -> None:
    image = torch.randn(2, 1, 16, 16)
    vector = torch.zeros(2, 2, 16, 16)
    warped = warp(image, vector, num_integration_steps=0)
    assert warped.shape == image.shape
    assert torch.allclose(warped, image, atol=1e-5)


def test_warp_tensor_shape_mismatch() -> None:
    image = torch.randn(1, 1, 8, 8)
    vector = torch.zeros(1, 2, 8, 10)
    with pytest.raises(ValueError, match="spatial dimensions"):
        warp_tensor(image, vector)


def test_integrate_vector_field_preserves_shape() -> None:
    vector = torch.randn(1, 2, 12, 12) * 0.1
    out = integrate_vector_field(vector, num_steps=2)
    assert out.shape == vector.shape


def test_optical_flow_ilk_displacement() -> None:
    height, width = 24, 24
    reference = torch.rand(height, width)
    # Slightly shifted moving frame so flow is non-trivial but cheap.
    moving = torch.roll(reference, shifts=1, dims=1).unsqueeze(0)
    flow = optical_flow_displacement(
        reference,
        moving,
        estimator_type=OpticalFlowEstimatorType.ILK,
        radius=3,
        num_warp=2,
    )
    assert flow.shape == (1, 2, height, width)


def test_optical_flow_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="one less dimension"):
        optical_flow_displacement(torch.rand(8, 8), torch.rand(8, 8))


@pytest.mark.parametrize(
    "filter_type",
    [
        DemonsFilterType.DEMONS,
        DemonsFilterType.FAST_SYMMETRIC_FORCES,
        DemonsFilterType.SYMMETRIC_FORCES,
        DemonsFilterType.DIFFEOMORPHIC,
    ],
)
def test_create_demons_filter(filter_type: DemonsFilterType) -> None:
    demons = create_demons_filter(filter_type=filter_type, num_iterations=5)
    assert demons.GetNumberOfIterations() == 5


def test_multiscale_demons_displacement() -> None:
    height, width = 32, 32
    reference = torch.rand(height, width)
    moving = torch.roll(reference, shifts=2, dims=0).unsqueeze(0)
    flow = multiscale_demons_displacement(
        reference,
        moving,
        filter_type=DemonsFilterType.DEMONS,
        num_iterations=5,
    )
    assert flow.shape == (1, 2, height, width)


def test_multiscale_demons_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="one less dimension"):
        multiscale_demons_displacement(torch.rand(8, 8), torch.rand(8, 8))
