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
"""Tests for learned registration models."""

import torch

from direct.nn.registration.registration import UnetRegistration2dModel
from direct.nn.registration.voxelmorph import SpatialTransformer, VecInt, VxmDense
from direct.registration.registration import DISCPLACEMENT_FIELD_2D_DIMENSIONS


def test_spatial_transformer_identity() -> None:
    height, width = 16, 20
    src = torch.randn(2, 1, height, width)
    flow = torch.zeros(2, 2, height, width)
    warped = SpatialTransformer((height, width))(src, flow)
    assert warped.shape == src.shape
    assert torch.allclose(warped, src, atol=1e-5)


def test_vec_int_zero_field() -> None:
    height, width = 12, 12
    vec = torch.zeros(1, 2, height, width)
    out = VecInt((height, width), nsteps=2)(vec)
    assert out.shape == vec.shape
    assert torch.allclose(out, torch.zeros_like(out))


def test_unet_registration_2d_forward() -> None:
    batch, seq_len, height, width = 2, 3, 32, 32
    model = UnetRegistration2dModel(
        max_seq_len=4,
        unet_num_filters=8,
        unet_num_pool_layers=2,
        warp_num_integration_steps=0,
    )
    moving = torch.randn(batch, seq_len, height, width)
    reference = torch.randn(batch, height, width)

    warped, displacement = model(moving, reference)
    assert warped.shape == moving.shape
    assert displacement.shape == (batch, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width)


def test_vxm_dense_forward() -> None:
    batch, seq_len, height, width = 1, 2, 32, 32
    model = VxmDense(
        inshape=(height, width),
        nb_unet_features=4,
        nb_unet_levels=2,
        nb_unet_conv_per_level=1,
        warp_num_integration_steps=0,
        int_downsize=1,
    )
    moving = torch.randn(batch, seq_len, height, width)
    reference = torch.randn(batch, height, width)

    warped, displacement = model(moving, reference)
    assert warped.shape == moving.shape
    assert displacement.shape == (batch, seq_len, 2, height, width)
