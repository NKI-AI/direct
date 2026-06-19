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

from direct.nn.conv.conv import Conv2d
from direct.nn.conv.modulated import (
    ModConv2d,
    ModConv2dBias,
    ModConv3d,
    ModConvActivation,
    ModConvTranspose2d,
    ModConvTranspose3d,
    ModConvType,
    ModulationParams,
    mod_conv2d,
    mod_conv3d,
    mod_conv_transpose2d,
    mod_conv_transpose3d,
)
from direct.nn.didn.didn import DIDN
from direct.nn.mwcnn.mwcnn import MWCNN
from direct.nn.unet.unet_2d import UnetModel2d

MODULATION_TYPES = [
    ModConvType.NONE,
    ModConvType.FEATURES,
    ModConvType.PARTIAL_IN,
    ModConvType.PARTIAL_OUT,
    ModConvType.SUM,
]


@pytest.mark.parametrize("modulation", MODULATION_TYPES)
@pytest.mark.parametrize("bias", [ModConv2dBias.PARAM, ModConv2dBias.NONE])
def test_modconv2d(modulation, bias):
    batch, in_ch, out_ch, h, w = 2, 4, 8, 16, 16
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        padding=1,
        bias=bias,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(16,), fc_groups=1)
    if modulation == ModConvType.SUM:
        kwargs["num_weights"] = 3

    model = ModConv2d(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, out_ch, h, w)


@pytest.mark.parametrize("modulation", MODULATION_TYPES)
def test_modconv_transpose2d(modulation):
    batch, in_ch, out_ch, h, w = 2, 8, 4, 8, 8
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=2,
        stride=2,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(16,), fc_groups=1)
    if modulation == ModConvType.SUM:
        kwargs["num_weights"] = 3

    model = ModConvTranspose2d(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, out_ch, h * 2, w * 2)


@pytest.mark.parametrize("modulation", [ModConvType.NONE, ModConvType.FEATURES, ModConvType.SUM])
def test_modconv3d(modulation):
    batch, in_ch, out_ch, d, h, w = 2, 4, 8, 4, 8, 8
    aux_feat = 3
    x = torch.randn(batch, in_ch, d, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        padding=1,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(16,), fc_groups=1)
    if modulation == ModConvType.SUM:
        kwargs["num_weights"] = 3

    model = ModConv3d(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, out_ch, d, h, w)


@pytest.mark.parametrize("modulation", [ModConvType.NONE, ModConvType.FEATURES, ModConvType.SUM])
def test_modconv_transpose3d(modulation):
    batch, in_ch, out_ch, d, h, w = 2, 8, 4, 4, 4, 4
    aux_feat = 3
    x = torch.randn(batch, in_ch, d, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=2,
        stride=2,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(16,), fc_groups=1)
    if modulation == ModConvType.SUM:
        kwargs["num_weights"] = 3

    model = ModConvTranspose3d(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, out_ch, d * 2, h * 2, w * 2)


@pytest.mark.parametrize("modulation", [ModConvType.NONE, ModConvType.FEATURES])
def test_conv2d_cascade(modulation):
    batch, in_ch, out_ch, h, w = 2, 2, 4, 16, 16
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_channels=8,
        n_convs=3,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(16,), fc_groups=1)

    model = Conv2d(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, out_ch, h, w)


@pytest.mark.parametrize("modulation", [ModConvType.NONE, ModConvType.FEATURES])
def test_didn(modulation):
    batch, in_ch, out_ch, h, w = 1, 2, 2, 32, 32
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_channels=8,
        num_dubs=2,
        num_convs_recon=3,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(8,), fc_groups=1)

    model = DIDN(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, out_ch, h, w)


@pytest.mark.parametrize("modulation", [ModConvType.NONE, ModConvType.FEATURES])
def test_mwcnn(modulation):
    batch, in_ch, h, w = 1, 2, 32, 32
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    kwargs = dict(
        input_channels=in_ch,
        first_conv_hidden_channels=8,
        num_scales=2,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(8,), fc_groups=1)

    model = MWCNN(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    assert out.shape == (batch, in_ch, h, w)


@pytest.mark.parametrize("modulation", [ModConvType.NONE, ModConvType.FEATURES])
def test_modconv2d_gradient_flow(modulation):
    batch, in_ch, out_ch, h, w = 2, 4, 8, 8, 8
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w, requires_grad=True)
    y = torch.randn(batch, aux_feat, requires_grad=True)

    kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        padding=1,
        modulation=modulation,
    )
    if modulation != ModConvType.NONE:
        kwargs.update(aux_in_features=aux_feat, fc_hidden_features=(16,), fc_groups=1)

    model = ModConv2d(**kwargs)
    if modulation == ModConvType.NONE:
        out = model(x)
    else:
        out = model(x, y)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    if modulation != ModConvType.NONE:
        assert y.grad is not None


@pytest.mark.parametrize("fc_groups", [1, 2])
def test_modconv2d_fc_groups(fc_groups):
    batch, in_ch, out_ch, h, w = 2, 4, 8, 8, 8
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    model = ModConv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        padding=1,
        modulation=ModConvType.FEATURES,
        aux_in_features=aux_feat,
        fc_hidden_features=(16,),
        fc_groups=fc_groups,
    )
    out = model(x, y)
    assert out.shape == (batch, out_ch, h, w)


def test_modconv2d_learned_bias():
    batch, in_ch, out_ch, h, w = 2, 4, 8, 8, 8
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    model = ModConv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        padding=1,
        modulation=ModConvType.FEATURES,
        bias=ModConv2dBias.LEARNED,
        aux_in_features=aux_feat,
        fc_hidden_features=(16,),
        fc_groups=1,
    )
    out = model(x, y)
    assert out.shape == (batch, out_ch, h, w)


@pytest.mark.parametrize(
    "modulation",
    [ModConvType.FULL, ModConvType.PARTIAL_IN, ModConvType.PARTIAL_OUT],
)
@pytest.mark.parametrize("fc_groups", [1, 2])
def test_modconv2d_weight_modulation_types(modulation, fc_groups):
    batch, in_ch, out_ch, h, w = 2, 4, 8, 16, 16
    aux_feat = 3
    x = torch.randn(batch, in_ch, h, w)
    y = torch.randn(batch, aux_feat)

    model = ModConv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        padding=1,
        modulation=modulation,
        aux_in_features=aux_feat,
        fc_hidden_features=(16, 8),
        fc_groups=fc_groups,
        fc_activation=ModConvActivation.SOFTPLUS,
    )
    out = model(x, y)
    assert out.shape == (batch, out_ch, h, w)


def test_mod_conv_factories():
    params = ModulationParams(
        modulation=ModConvType.FEATURES,
        aux_in_features=2,
        fc_hidden_features=(8,),
        fc_activation=ModConvActivation.SOFTPLUS,
    )
    x2d = torch.randn(1, 4, 8, 8)
    y = torch.randn(1, 2)
    x3d = torch.randn(1, 4, 4, 8, 8)

    conv2d = mod_conv2d(4, 8, kernel_size=3, padding=1, modulation_params=params)
    assert conv2d(x2d, y).shape == (1, 8, 8, 8)

    conv_t2d = mod_conv_transpose2d(8, 4, kernel_size=2, stride=2, modulation_params=params)
    assert conv_t2d(torch.randn(1, 8, 8, 8), y).shape == (1, 4, 16, 16)

    conv3d = mod_conv3d(4, 8, kernel_size=3, padding=1, modulation_params=params)
    assert conv3d(x3d, y).shape == (1, 8, 4, 8, 8)

    conv_t3d = mod_conv_transpose3d(8, 4, kernel_size=2, stride=2, modulation_params=params)
    assert conv_t3d(torch.randn(1, 8, 4, 8, 8), y).shape == (1, 4, 8, 16, 16)


def test_modconv2d_init_validation():
    with pytest.raises(ValueError, match="aux_in_features"):
        ModConv2d(4, 8, 3, modulation=ModConvType.FEATURES)
    with pytest.raises(ValueError, match="fc_hidden_features"):
        ModConv2d(4, 8, 3, modulation=ModConvType.FEATURES, aux_in_features=2)
    with pytest.raises(ValueError, match="num_weights"):
        ModConv2d(
            4,
            8,
            3,
            modulation=ModConvType.SUM,
            aux_in_features=2,
            fc_hidden_features=(8,),
        )
    with pytest.raises(ValueError, match="LEARNED requires modulation"):
        ModConv2d(4, 8, 3, bias=ModConv2dBias.LEARNED)


def test_unet2d_modulated_forward():
    model = UnetModel2d(
        in_channels=2,
        out_channels=2,
        num_filters=8,
        num_pool_layers=2,
        dropout_probability=0.0,
        modulation=ModConvType.FEATURES,
        aux_in_features=2,
        fc_hidden_features=(8,),
    )
    x = torch.randn(1, 2, 32, 32)
    y = torch.randn(1, 2)
    out = model(x, y)
    assert out.shape == x.shape


def test_didn_modulation_params():
    params = ModulationParams(
        modulation=ModConvType.FEATURES,
        aux_in_features=2,
        fc_hidden_features=(8,),
    )
    model = DIDN(
        in_channels=2,
        out_channels=2,
        hidden_channels=8,
        num_dubs=2,
        num_convs_recon=3,
        modulation_params=params,
    )
    x = torch.randn(1, 2, 32, 32)
    y = torch.randn(1, 2)
    out = model(x, y)
    assert out.shape == x.shape
