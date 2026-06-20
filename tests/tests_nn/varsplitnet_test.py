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
"""Tests for the direct.nn.varsplitnet module."""

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.get_nn_model_config import ModelName
from direct.nn.types import ActivationType, InitType
from direct.nn.varsplitnet.varsplitnet import MRIVarSplitNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize("shape", [[4, 3, 32, 32], [4, 5, 40, 20]])
@pytest.mark.parametrize("num_steps_reg", [2, 3])
@pytest.mark.parametrize("num_steps_dc", [1, 4])
@pytest.mark.parametrize("image_init", [InitType.SENSE, InitType.ZERO_FILLED])
@pytest.mark.parametrize("no_parameter_sharing", [True, False])
@pytest.mark.parametrize(
    "image_model_architecture, image_model_kwargs",
    [
        [ModelName.UNET, {"image_unet_num_filters": 4, "image_unet_num_pool_layers": 2}],
        [ModelName.DIDN, {"image_didn_hidden_channels": 4, "image_didn_num_dubs": 2, "image_didn_num_convs_recon": 2}],
        [
            ModelName.CONV,
            {
                "image_conv_hidden_channels": 8,
                "image_conv_n_convs": 3,
                "image_conv_activation": ActivationType.LEAKY_RELU,
            },
        ],
    ],
)
@pytest.mark.parametrize("kspace_no_parameter_sharing", [True, False])
@pytest.mark.parametrize(
    "kspace_model_architecture, kspace_model_kwargs",
    [
        [
            ModelName.CONV,
            {
                "kspace_conv_hidden_channels": 8,
                "kspace_conv_n_convs": 3,
                "kspace_conv_activation": ActivationType.RELU,
            },
        ],
        [None, {}],
    ],
)
@pytest.mark.parametrize("pass_scaling_factor", [True, False])
def test_varsplitnet(
    shape,
    num_steps_reg,
    num_steps_dc,
    image_init,
    no_parameter_sharing,
    image_model_architecture,
    image_model_kwargs,
    kspace_no_parameter_sharing,
    kspace_model_architecture,
    kspace_model_kwargs,
    pass_scaling_factor,
):
    model = MRIVarSplitNet(
        fft2,
        ifft2,
        num_steps_reg,
        num_steps_dc,
        image_init,
        no_parameter_sharing,
        image_model_architecture,
        kspace_no_parameter_sharing,
        kspace_model_architecture,
        **image_model_kwargs,
        **kspace_model_kwargs,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()
    scaling_factor = None if not pass_scaling_factor else create_input(shape[0]).cpu()
    out = model(kspace, sens, mask, scaling_factor)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
