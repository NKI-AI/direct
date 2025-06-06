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
from direct.nn.kikinet.kikinet import KIKINet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "num_iter",
    [1, 3],
)
@pytest.mark.parametrize(
    "image_model_architecture",
    ["MWCNN", "UNET", "NORMUNET"],
)
@pytest.mark.parametrize(
    "kspace_model_architecture",
    ["CONV", "DIDN", "UNET", "NORMUNET"],
)
@pytest.mark.parametrize(
    "normalize",
    [True, False],
)
def test_kikinet(shape, num_iter, image_model_architecture, kspace_model_architecture, normalize):
    model = KIKINet(
        fft2,
        ifft2,
        num_iter=num_iter,
        image_model_architecture=image_model_architecture,
        kspace_model_architecture=kspace_model_architecture,
        normalize=normalize,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()

    out = model(kspace, mask, sens)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
