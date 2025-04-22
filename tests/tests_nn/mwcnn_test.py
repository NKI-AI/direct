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
import pytest
import torch
import torch.nn as nn

from direct.nn.mwcnn.mwcnn import MWCNN


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 20, 34],
    ],
)
@pytest.mark.parametrize(
    "first_conv_hidden_channels",
    [4, 8],
)
@pytest.mark.parametrize(
    "n_scales",
    [2, 3],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
@pytest.mark.parametrize(
    "act",
    [nn.ReLU(), nn.PReLU()],
)
def test_mwcnn(shape, first_conv_hidden_channels, n_scales, bias, batchnorm, act):
    model = MWCNN(shape[1], first_conv_hidden_channels, n_scales, bias, batchnorm, act)

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == shape
