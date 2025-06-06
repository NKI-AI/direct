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
from direct.nn.iterdualnet.iterdualnet import IterDualNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize("shape", [[3, 3, 32, 32], [5, 11, 20, 22]])
@pytest.mark.parametrize("num_iter", [3, 5])
@pytest.mark.parametrize("image_no_parameter_sharing", [True, False])
@pytest.mark.parametrize("kspace_no_parameter_sharing", [True, False])
@pytest.mark.parametrize("compute_per_coil", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_iterdualnet(
    shape, num_iter, image_no_parameter_sharing, kspace_no_parameter_sharing, compute_per_coil, normalized
):
    model = IterDualNet(
        fft2,
        ifft2,
        num_iter=num_iter,
        image_normunet=normalized,
        kspace_normunet=normalized,
        image_no_parameter_sharing=image_no_parameter_sharing,
        kspace_no_parameter_sharing=kspace_no_parameter_sharing,
        compute_per_coil=compute_per_coil,
        image_unet_num_filters=4,
        image_unet_num_pool_layers=3,
        kspace_unet_num_filters=4,
        kspace_unet_num_pool_layers=3,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()

    out = model(kspace, mask, sens)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
