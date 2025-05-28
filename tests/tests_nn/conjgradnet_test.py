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
"""Tests for the direct.nn.conjgradnet module."""

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.conjgradnet.conjgrad import CGUpdateType
from direct.nn.conjgradnet.conjgradnet import ConjGradNet
from direct.nn.get_nn_model_config import ModelName
from direct.nn.types import InitType


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [[3, 3, 32, 32]],
)
@pytest.mark.parametrize("nums_steps", [3])
@pytest.mark.parametrize(
    "denoiser_architecture, kwargs",
    [
        [
            ModelName.RESNET,
            {"resnet_hidden_channels": 8, "resnet_num_blocks": 4, "resnet_batchnorm": True, "resnet_scale": None},
        ],
    ],
)
@pytest.mark.parametrize(
    "cg_param_update_type", [CGUpdateType.FR, CGUpdateType.PRP, CGUpdateType.DY, CGUpdateType.BAN]
)
@pytest.mark.parametrize("image_init", [InitType.SENSE, InitType.ZERO_FILLED, InitType.ZEROS, "invalid"])
@pytest.mark.parametrize("no_parameter_sharing", [True, False])
@pytest.mark.parametrize("cg_iters", [5, 20])
@pytest.mark.parametrize("cg_tol", [1e-2, 1e-8])
def test_conjgradnet(
    shape,
    nums_steps,
    denoiser_architecture,
    kwargs,
    cg_param_update_type,
    image_init,
    no_parameter_sharing,
    cg_iters,
    cg_tol,
):
    kwargs = {
        "forward_operator": fft2,
        "backward_operator": ifft2,
        "num_steps": nums_steps,
        "denoiser_architecture": denoiser_architecture,
        "image_init": image_init,
        "no_parameter_sharing": no_parameter_sharing,
        "cg_iters": cg_iters,
        "cg_tol": cg_tol,
        "cg_param_update_type": cg_param_update_type,
        **kwargs,
    }

    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    if image_init == "invalid":
        with pytest.raises(ValueError):
            model = ConjGradNet(**kwargs)
    else:
        model = ConjGradNet(**kwargs)
        out = model(kspace, sens, mask)
        assert list(out.shape) == [shape[0]] + shape[2:] + [2]
