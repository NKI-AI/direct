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
"""Tests for direct.nn.unet.unet_3d module."""

import numpy as np
import pytest
import torch

from direct.nn.unet.unet_3d import NormUnetModel3d, UnetModel3d


def create_input(shape):
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3, 20, 16, 16],
        [4, 2, 30, 20, 30],
        [4, 2, 21, 24, 20],
    ],
)
@pytest.mark.parametrize(
    "num_filters",
    [4, 6, 8],
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [2, 3],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_unet_3d(shape, num_filters, num_pool_layers, normalized):
    model_architecture = NormUnetModel3d if normalized else UnetModel3d
    model = model_architecture(
        in_channels=shape[1],
        out_channels=shape[1],
        num_filters=num_filters,
        num_pool_layers=num_pool_layers,
        dropout_probability=0.05,
    ).cpu()

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == shape
