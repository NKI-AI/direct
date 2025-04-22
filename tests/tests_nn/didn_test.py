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

from direct.nn.didn.didn import DIDN


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 16, 16],
    ],
)
@pytest.mark.parametrize(
    "out_channels",
    [3, 5],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [16, 8],
)
@pytest.mark.parametrize(
    "n_dubs",
    [3, 4],
)
@pytest.mark.parametrize(
    "num_convs_recon",
    [3, 4],
)
@pytest.mark.parametrize(
    "skip",
    [True, False],
)
def test_didn(shape, out_channels, hidden_channels, n_dubs, num_convs_recon, skip):
    model = DIDN(shape[1], out_channels, hidden_channels, n_dubs, num_convs_recon, skip)

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == [shape[0]] + [out_channels] + shape[2:]
