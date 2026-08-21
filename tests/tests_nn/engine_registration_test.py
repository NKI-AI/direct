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
"""Engine tests covering adaptive sampling and registration branches."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.medl.config import MEDL2DConfig
from direct.nn.medl.medl import MEDL2D
from direct.nn.medl.medl_engine import MEDLEngine
from direct.nn.registration.config import UnetRegistration2dModelConfig
from direct.nn.registration.registration import UnetRegistration2dModel
from direct.nn.vsharp.config import VSharpNetConfig
from direct.nn.vsharp.vsharp import VSharpNet
from direct.nn.vsharp.vsharp_engine import VSharpNetEngine


def _make_reg_cfg() -> UnetRegistration2dModelConfig:
    return UnetRegistration2dModelConfig(
        max_seq_len=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
        reg_loss_factor=1.0,
        rec_loss_factor=1.0,
        train_end_to_end=True,
        decoupled_training=False,
        reg_loss_on_target=False,
    )


def test_vsharp_engine_with_registration() -> None:
    shape = (2, 3, 16, 16, 2)
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    loss_config = LossConfig(losses=[FunctionConfig("l1_loss")])
    config = DefaultConfig(
        training=TrainingConfig(loss=loss_config),
        validation=ValidationConfig(crop=None),
        model=VSharpNetConfig(
            num_steps=2,
            num_steps_dc_gd=1,
            image_unet_num_filters=8,
            image_unet_num_pool_layers=2,
            auxiliary_steps=-1,
        ),
    )
    reg_cfg = _make_reg_cfg()
    config.additional_models = SimpleNamespace(registration_model=reg_cfg)

    model = VSharpNet(
        forward_operator,
        backward_operator,
        num_steps=2,
        num_steps_dc_gd=1,
        image_unet_num_filters=8,
        image_unet_num_pool_layers=2,
        auxiliary_steps=-1,
    )
    registration_model = UnetRegistration2dModel(
        max_seq_len=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    engine = VSharpNetEngine(
        config,
        model,
        "cpu",
        fft2,
        ifft2,
        sensitivity_model=sensitivity_model,
        registration_model=registration_model,
    )
    engine.ndim = 2
    engine.model.train()

    data = {
        "masked_kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "sensitivity_map": torch.from_numpy(np.random.randn(*shape)).float(),
        "sampling_mask": torch.from_numpy(np.random.randn(shape[0], 1, shape[2], shape[3], 1)).round().bool(),
        "target": torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        "reference_image": torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        "scaling_factor": torch.ones(shape[0]),
    }
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image is not None


def test_medl_engine_with_registration() -> None:
    shape = (2, 3, 16, 16, 2)
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    config = DefaultConfig(
        training=TrainingConfig(loss=LossConfig(losses=[FunctionConfig("l1_loss")])),
        validation=ValidationConfig(crop=None),
        model=MEDL2DConfig(iterations=1, num_layers=1, unet_num_filters=4, unet_num_pool_layers=2),
    )
    reg_cfg = _make_reg_cfg()
    config.additional_models = SimpleNamespace(registration_model=reg_cfg)

    model = MEDL2D(
        forward_operator,
        backward_operator,
        iterations=1,
        num_layers=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
    )
    registration_model = UnetRegistration2dModel(
        max_seq_len=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    engine = MEDLEngine(
        config,
        model,
        "cpu",
        fft2,
        ifft2,
        sensitivity_model=sensitivity_model,
        registration_model=registration_model,
    )
    engine.ndim = 2
    engine.model.train()

    data = {
        "masked_kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "sensitivity_map": torch.from_numpy(np.random.randn(*shape)).float(),
        "sampling_mask": torch.from_numpy(np.random.randn(shape[0], 1, shape[2], shape[3], 1)).round().bool(),
        "target": torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        "reference_image": torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        "scaling_factor": torch.ones(shape[0]),
    }
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image is not None


def test_medl_engine_decoupled_registration() -> None:
    shape = (1, 2, 16, 16, 2)
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    config = DefaultConfig(
        training=TrainingConfig(loss=LossConfig(losses=[FunctionConfig("l1_loss")])),
        validation=ValidationConfig(crop=None),
        model=MEDL2DConfig(iterations=1, num_layers=1, unet_num_filters=4, unet_num_pool_layers=2),
    )
    reg_cfg = _make_reg_cfg()
    reg_cfg.decoupled_training = True
    reg_cfg.train_end_to_end = False
    config.additional_models = SimpleNamespace(registration_model=reg_cfg)

    model = MEDL2D(
        forward_operator,
        backward_operator,
        iterations=1,
        num_layers=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
    )
    registration_model = UnetRegistration2dModel(
        max_seq_len=1,
        unet_num_filters=4,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
    )
    engine = MEDLEngine(
        config,
        model,
        "cpu",
        fft2,
        ifft2,
        sensitivity_model=torch.nn.Conv2d(2, 2, kernel_size=1),
        registration_model=registration_model,
    )
    engine.ndim = 2
    engine.model.train()
    data = {
        "masked_kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "sensitivity_map": torch.from_numpy(np.random.randn(*shape)).float(),
        "sampling_mask": torch.from_numpy(np.random.randn(shape[0], 1, shape[2], shape[3], 1)).round().bool(),
        "target": torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        "reference_image": torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        "scaling_factor": torch.ones(shape[0]),
    }
    out = engine._do_iteration(data, engine.build_loss())
    assert out.output_image is not None
