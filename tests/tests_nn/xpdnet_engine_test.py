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
from direct.nn.xpdnet.xpdnet import XPDNet
from direct.nn.xpdnet.xpdnet_engine import XPDNetEngine


def create_sample(shape, **kwargs):
    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sampling_mask"] = torch.from_numpy(np.random.randn(1, shape[1], shape[2], 1)).float()
    sample["target"] = torch.from_numpy(np.random.randn(shape[1], shape[2])).float()
    sample["scaling_factor"] = torch.tensor([1.0])
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


@pytest.mark.parametrize("shape", [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)])
@pytest.mark.parametrize("loss_fns", [["l1_loss", "ssim_loss", "l2_loss"]])
@pytest.mark.parametrize("num_iter", [2])
@pytest.mark.parametrize("num_primal", [3])
@pytest.mark.parametrize("image_model_architecture", ["MWCNN"])
@pytest.mark.parametrize("primal_only, kspace_model_architecture, num_dual", [[True, None, 1]])
def test_xpdnet_engine(
    shape,
    loss_fns,
    num_iter,
    num_primal,
    image_model_architecture,
    primal_only,
    kspace_model_architecture,
    num_dual,
):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    kwargs = {
        "forward_operator": forward_operator,
        "backward_operator": backward_operator,
        "num_iter": num_iter,
        "num_primal": num_primal,
        "num_dual": num_dual,
        "image_model_architecture": image_model_architecture,
        "kspace_model_architecture": kspace_model_architecture,
        "use_primal_only": primal_only,
    }
    model = XPDNet(**kwargs)
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    config = DefaultConfig(training=training_config, validation=validation_config)
    # Define engine
    engine = XPDNetEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
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
