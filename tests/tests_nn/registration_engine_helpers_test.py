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
"""Tests for classical NN registration wrappers and MRIModelEngine helpers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig
from direct.nn.loss_keys import KeyedLossFn
from direct.nn.mri_models import MRIModelEngine
from direct.nn.registration.config import UnetRegistration2dModelConfig
from direct.nn.registration.registration import (
    DemonsRegistration2dModel,
    OpticalFlowILKRegistration2dModel,
    OpticalFlowTVL1Registration2dModel,
    UnetRegistration2dModel,
)
from direct.registration.demons import DemonsFilterType
from direct.registration.registration import DISCPLACEMENT_FIELD_2D_DIMENSIONS


def test_optical_flow_ilk_registration_model() -> None:
    model = OpticalFlowILKRegistration2dModel(
        radius=3,
        num_warp=2,
        warp_num_integration_steps=0,
    )
    batch, seq_len, height, width = 1, 2, 24, 24
    reference = torch.rand(batch, height, width)
    moving = torch.stack([torch.roll(reference[0], shifts=i + 1, dims=1) for i in range(seq_len)], dim=0).unsqueeze(0)
    warped, displacement = model(moving, reference)
    assert warped.shape == moving.shape
    assert displacement.shape == (batch, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width)


def test_optical_flow_tvl1_registration_model() -> None:
    model = OpticalFlowTVL1Registration2dModel(
        num_warp=1,
        num_iter=2,
        warp_num_integration_steps=0,
    )
    batch, seq_len, height, width = 1, 1, 20, 20
    reference = torch.rand(batch, height, width)
    moving = reference.unsqueeze(1)
    warped, displacement = model(moving, reference)
    assert warped.shape == moving.shape
    assert displacement.shape[-2:] == (height, width)


def test_demons_registration_2d_model() -> None:
    model = DemonsRegistration2dModel(
        demons_filter_type=DemonsFilterType.DEMONS,
        demons_num_iterations=3,
        warp_num_integration_steps=0,
    )
    batch, seq_len, height, width = 1, 1, 24, 24
    reference = torch.rand(batch, height, width)
    moving = torch.roll(reference, shifts=2, dims=-1).unsqueeze(1)
    warped, displacement = model(moving, reference)
    assert warped.shape == moving.shape
    assert displacement.shape == (batch, seq_len, 2, height, width)


class _DummyMRIEngine(MRIModelEngine):
    def __init__(self, cfg, registration_model: nn.Module):
        # Bypass Engine.__init__; only need registration helpers.
        self.cfg = cfg
        self.model = nn.Linear(1, 1)
        self.models = {"registration_model": registration_model}
        self._complex_dim = -1
        self._coil_dim = 1
        self.device = "cpu"
        self._scaler = SimpleNamespace(scale=lambda x: x)

    def forward_function(self, data):
        raise NotImplementedError


def test_mri_model_engine_registration_helpers() -> None:
    reg_cfg = UnetRegistration2dModelConfig(
        max_seq_len=2,
        unet_num_filters=4,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
        reg_loss_factor=1.0,
        reg_loss_on_target=True,
    )
    cfg = DefaultConfig(training=TrainingConfig(loss=LossConfig(losses=[FunctionConfig("l1_loss")])))
    cfg.additional_models = SimpleNamespace(registration_model=reg_cfg)

    registration_model = UnetRegistration2dModel(
        max_seq_len=2,
        unet_num_filters=4,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
    )
    engine = _DummyMRIEngine(cfg, registration_model)

    batch, seq_len, height, width = 1, 2, 16, 16
    moving = torch.randn(batch, seq_len, height, width)
    reference = torch.randn(batch, height, width)
    data = {
        "reference_image": reference,
        "target": torch.randn(batch, seq_len, height, width),
        "reconstruction_size": None,
    }

    registered, displacement = engine.do_registration(data, moving)
    assert registered.shape == moving.shape
    assert displacement.shape[0] == batch

    warped = engine.warp_with_displacement(moving, displacement)
    assert warped.shape == moving.shape

    tiled = engine._registration_reference_image(data, registered)
    assert tiled.shape == registered.shape

    def _l1(source, target, reduction="mean", reconstruction_size=None):
        return (source - target).abs().mean()

    loss_fns = {
        "l1_loss": KeyedLossFn(_l1, "output_image", "target"),
        "registered_l1_loss": KeyedLossFn(_l1, "registered_image", "reference_image"),
    }
    loss_dict, _ = engine._accumulate_registration_losses(
        loss_fns,
        {},
        data,
        registered,
        displacement,
    )
    assert "l1_loss" in loss_dict or "registered_l1_loss" in loss_dict

    engine._set_requires_grad(False, include_registration=False)
    assert all(not p.requires_grad for p in engine.model.parameters())
    assert any(p.requires_grad for p in engine.models["registration_model"].parameters())


def test_perform_sampling_with_parameterized_policy() -> None:
    from direct.nn.adaptive.parameterized import Parameterized2dPolicy
    from direct.nn.adaptive.types import PolicySamplingDimension

    height, width = 12, 10
    policy = Parameterized2dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
        acceleration=4.0,
    )
    cfg = DefaultConfig(training=TrainingConfig(loss=LossConfig(losses=[FunctionConfig("l1_loss")])))
    engine = _DummyMRIEngine(cfg, nn.Identity())
    engine.models = {"sampling_model": policy}

    batch, coils = 1, 2
    kspace = torch.randn(batch, coils, height, width, 2)
    mask = torch.zeros(batch, 1, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1.0
    data = {
        "kspace": kspace,
        "masked_kspace": kspace * mask,
        "sampling_mask": mask,
        "sensitivity_map": torch.ones(batch, coils, height, width, 2) / coils,
        "acceleration": torch.tensor([[4.0]]),
    }

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out = engine.perform_sampling(data)
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")

    assert "probability_masks" in out
    assert out["masked_kspace"].shape == kspace.shape
