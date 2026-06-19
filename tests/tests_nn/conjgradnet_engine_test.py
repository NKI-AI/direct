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
import functools

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.conjgradnet.conjgrad import CGUpdateType
from direct.nn.conjgradnet.conjgradnet import ConjGradNet
from direct.nn.conjgradnet.conjgradnet_engine import ConjGradNetEngine


def create_sample(shape, **kwargs):
    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["scaling_factor"] = torch.tensor([1.0])
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    for k, v in sample.items():
        sample[k] = v.cpu()
    return sample


@pytest.mark.parametrize("shape", [(4, 3, 10, 16, 2)])
@pytest.mark.parametrize("loss_fns", [["l1_loss"]])
@pytest.mark.parametrize("nums_steps", [2])
@pytest.mark.parametrize(
    "denoiser_architecture, resnet_hidden_channels, resnet_num_blocks, resnet_batchnorm, resnet_scale",
    [["resnet", 8, 4, True, 1.0]],
)
@pytest.mark.parametrize("cg_param_update_type", [CGUpdateType.FR])
def test_resnetconjgrad_engine(
    shape,
    loss_fns,
    nums_steps,
    denoiser_architecture,
    resnet_hidden_channels,
    resnet_num_blocks,
    resnet_batchnorm,
    resnet_scale,
    cg_param_update_type,
):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    kwargs = {
        "forward_operator": forward_operator,
        "backward_operator": backward_operator,
        "num_steps": nums_steps,
        "denoiser_architecture": denoiser_architecture,
        "resnet_batchnorm": resnet_batchnorm,
        "resnet_scale": resnet_scale,
        "resnet_num_blocks": resnet_num_blocks,
        "resnet_hidden_channels": resnet_hidden_channels,
    }
    model = ConjGradNet(**kwargs).cpu()
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1).cpu()
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    config = DefaultConfig(training=training_config, validation=validation_config)
    # Define engine
    engine = ConjGradNetEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).float(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image.shape == (shape[0],) + tuple(shape[2:-1])
