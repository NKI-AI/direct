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

from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 16, 16],
        [3, 2, 15, 17],
    ],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [4, 8],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_conv2dgru(shape, hidden_channels, normalized):
    model = (
        NormConv2dGRU(shape[1], hidden_channels, shape[1])
        if normalized
        else Conv2dGRU(shape[1], hidden_channels, shape[1])
    )
    data = create_input(shape).cpu()

    out = model(data, None)[0]

    assert list(out.shape) == shape
