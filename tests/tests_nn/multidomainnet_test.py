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

from direct.data.transforms import fft2, ifft2
from direct.nn.multidomainnet.multidomain import MultiDomainUnet2d
from direct.nn.multidomainnet.multidomainnet import MultiDomainNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [2, 2, 16, 16],
        [4, 2, 16, 32],
        [3, 2, 32, 32],
        [3, 2, 40, 20],
    ],
)
@pytest.mark.parametrize(
    "num_filters",
    [4, 8, 16],  # powers of 2
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [2, 3],
)
def test_multidomainunet2d(shape, num_filters, num_pool_layers):
    model = MultiDomainUnet2d(
        fft2,
        ifft2,
        shape[1],
        shape[1],
        num_filters=num_filters,
        num_pool_layers=num_pool_layers,
        dropout_probability=0.05,
    ).cpu()

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == shape


@pytest.mark.parametrize(
    "shape",
    [
        [2, 2, 16, 16],
        [4, 2, 16, 32],
        [3, 2, 32, 32],
        [3, 2, 40, 20],
    ],
)
@pytest.mark.parametrize("standardization", [True, False])
@pytest.mark.parametrize(
    "num_filters",
    [4, 8],  # powers of 2
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [2, 3],
)
def test_multidomainnet(shape, standardization, num_filters, num_pool_layers):
    model = MultiDomainNet(fft2, ifft2, standardization, num_filters, num_pool_layers)

    shape = shape + [2]

    kspace = create_input(shape).cpu()
    sens = create_input(shape).cpu()

    out = model(kspace, sens)

    assert list(out.shape) == shape
