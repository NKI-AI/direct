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
"""Tests for the direct.nn.medl module."""

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.medl.medl import MEDL2D, MEDL3D


def create_input(shape: list[int]) -> torch.Tensor:
    return torch.rand(shape).float()


@pytest.mark.parametrize("shape", [[2, 4, 32, 32]])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("iterations", [2, (1, 2)])
def test_medl2d(shape: list[int], num_layers: int, iterations: int | tuple[int, ...]) -> None:
    if isinstance(iterations, tuple) and len(iterations) != num_layers:
        pytest.skip("iterations length must match num_layers")

    model = MEDL2D(
        forward_operator=fft2,
        backward_operator=ifft2,
        iterations=iterations,
        num_layers=num_layers,
        unet_num_filters=8,
        unet_num_pool_layers=2,
    )
    kspace = create_input(shape + [2])
    sens = create_input(shape + [2])
    mask = create_input([shape[0], 1] + shape[2:] + [1]).round()

    out = model(kspace, mask, sens)
    assert len(out) == num_layers
    assert list(out[-1].shape) == [shape[0], shape[2], shape[3], 2]


def test_medl2d_rejects_mismatched_iterations() -> None:
    with pytest.raises(ValueError, match="Number of iterations"):
        MEDL2D(
            forward_operator=fft2,
            backward_operator=ifft2,
            iterations=(1, 2, 3),
            num_layers=2,
            unet_num_filters=4,
            unet_num_pool_layers=2,
        )


@pytest.mark.parametrize("shape", [[1, 2, 2, 16, 16]])
def test_medl3d(shape: list[int]) -> None:
    model = MEDL3D(
        forward_operator=fft2,
        backward_operator=ifft2,
        iterations=1,
        num_layers=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
    )
    kspace = create_input(shape + [2])
    sens = create_input(shape + [2])
    mask = create_input([shape[0], 1, 1] + shape[3:] + [1]).round()

    out = model(kspace, mask, sens)
    assert len(out) == 1
    assert list(out[0].shape) == [shape[0], shape[2], shape[3], shape[4], 2]
