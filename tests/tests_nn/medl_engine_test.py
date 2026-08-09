# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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
"""Tests for the direct.nn.medl.medl_engine module."""

import functools

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.medl.config import MEDL2DConfig, MEDL3DConfig
from direct.nn.medl.medl import MEDL2D, MEDL3D
from direct.nn.medl.medl_engine import MEDL3DEngine, MEDLEngine


def create_sample(shape, **kwargs):
    sample = {
        "masked_kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "sensitivity_map": torch.from_numpy(np.random.randn(*shape)).float(),
    }
    sample.update(kwargs)
    return sample


@pytest.mark.parametrize("shape", [(2, 3, 16, 16, 2)])
def test_medl_engine(shape):
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    loss_config = LossConfig(losses=[FunctionConfig("l1_loss")])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = MEDL2DConfig(
        iterations=1,
        num_layers=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
    )
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    model = MEDL2D(
        forward_operator,
        backward_operator,
        iterations=model_config.iterations,
        num_layers=model_config.num_layers,
        unet_num_filters=model_config.unet_num_filters,
        unet_num_pool_layers=model_config.unet_num_pool_layers,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    engine = MEDLEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2

    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(shape[0], 1, shape[2], shape[3], 1)).round().bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image.shape == (shape[0], shape[2], shape[3])


@pytest.mark.parametrize("shape", [(1, 2, 2, 16, 16, 2)])
def test_medl3d_engine(shape):
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    loss_config = LossConfig(losses=[FunctionConfig("l1_loss")])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = MEDL3DConfig(
        iterations=1,
        num_layers=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
    )
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    model = MEDL3D(
        forward_operator,
        backward_operator,
        iterations=model_config.iterations,
        num_layers=model_config.num_layers,
        unet_num_filters=model_config.unet_num_filters,
        unet_num_pool_layers=model_config.unet_num_pool_layers,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    engine = MEDL3DEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 3

    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(shape[0], 1, 1, shape[3], shape[4], 1)).round().bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3], shape[4])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image.shape == (shape[0], shape[2], shape[3], shape[4])
