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
"""Tests for registration transform modules and visualization."""

import pytest
import torch

from direct.registration.demons import DemonsFilterType
from direct.registration.elastic_deformation import (
    RandomElasticDeformationModule,
    random_elastic_deformation,
)
from direct.registration.registration import (
    DisplacementModule,
    DisplacementTransformType,
    WarpModule,
)
from direct.registration.visualize import apply_plasma, displacement_field_to_warped_grid
from direct.types import TransformKey


def test_random_elastic_deformation_shape() -> None:
    image = torch.rand(2, 24, 24)
    out = random_elastic_deformation(image, sigma=1.0, points=3, order=1, seed=0)
    assert out.shape == image.shape


def test_random_elastic_deformation_module() -> None:
    module = RandomElasticDeformationModule(
        image_key=TransformKey.TARGET,
        target_key=TransformKey.REFERENCE_IMAGE,
        sigma=1.0,
        points=3,
        order=1,
        use_seed=None,
    )
    data = {TransformKey.TARGET: torch.rand(1, 20, 20)}
    out = module(data)
    assert TransformKey.REFERENCE_IMAGE in out
    assert out[TransformKey.REFERENCE_IMAGE].shape == data[TransformKey.TARGET].shape


def test_displacement_and_warp_modules() -> None:
    height, width = 24, 24
    reference = torch.rand(1, height, width)
    moving = torch.roll(reference, shifts=1, dims=-1).unsqueeze(1)  # (B, T, H, W)
    sample = {
        TransformKey.REFERENCE_IMAGE: reference,
        TransformKey.MOVING_IMAGE: moving,
    }
    displacement_module = DisplacementModule(
        transform_type=DisplacementTransformType.MULTISCALE_DEMONS,
        demons_filter_type=DemonsFilterType.DEMONS,
        demons_num_iterations=3,
    )
    sample = displacement_module(sample)
    assert TransformKey.DISPLACEMENT_FIELD in sample
    assert sample[TransformKey.DISPLACEMENT_FIELD].shape[0] == 1

    sample = WarpModule()(sample)
    assert TransformKey.WARPED_IMAGE in sample
    # WarpModule keeps a singleton channel axis after reshaping.
    assert sample[TransformKey.WARPED_IMAGE].shape == (moving.shape[0], moving.shape[1], 1, height, width)


def test_displacement_module_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="MULTISCALE_DEMONS"):
        DisplacementModule(transform_type=DisplacementTransformType.OPTICAL_FLOW)


def test_apply_plasma_and_warped_grid() -> None:
    values = torch.linspace(0, 1, 16).numpy()
    rgb = apply_plasma(values)
    assert rgb.shape == (16, 3)

    field = torch.zeros(2, 32, 32)
    field[0, 10:20, 10:20] = 1.5
    grid = displacement_field_to_warped_grid(field, spacing=8)
    assert grid.shape == (3, 32, 32)
    assert grid.min() >= 0 and grid.max() <= 1


def test_displacement_field_to_warped_grid_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="Expected displacement_field"):
        displacement_field_to_warped_grid(torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="spacing"):
        displacement_field_to_warped_grid(torch.zeros(2, 8, 8), spacing=0)
