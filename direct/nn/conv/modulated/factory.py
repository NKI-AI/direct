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
"""Typed helpers for constructing modulated convolution layers."""

from dataclasses import dataclass

from direct.nn.conv.modulated.modulated_conv import (
    ModConv2d,
    ModConv2dBias,
    ModConv3d,
    ModConvActivation,
    ModConvTranspose2d,
    ModConvTranspose3d,
    ModConvType,
)
from direct.types import IntOrTuple

__all__ = [
    "ModulationParams",
    "mod_conv2d",
    "mod_conv3d",
    "mod_conv_transpose2d",
    "mod_conv_transpose3d",
]


def mod_conv2d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: IntOrTuple,
    modulation_params: ModulationParams | None = None,
    stride: IntOrTuple = 1,
    padding: IntOrTuple = 0,
    dilation: IntOrTuple = 1,
    bias: ModConv2dBias = ModConv2dBias.PARAM,
) -> ModConv2d:
    """See :class:`ModConv2d`."""
    params = modulation_params or ModulationParams()
    return ModConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        modulation=params.modulation,
        aux_in_features=params.aux_in_features,
        fc_hidden_features=params.fc_hidden_features,
        fc_bias=params.fc_bias,
        fc_groups=params.fc_groups,
        fc_activation=params.fc_activation,
        num_weights=params.num_weights,
    )


def mod_conv_transpose2d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: IntOrTuple,
    modulation_params: ModulationParams | None = None,
    stride: IntOrTuple = 1,
    padding: IntOrTuple = 0,
    dilation: IntOrTuple = 1,
    bias: ModConv2dBias = ModConv2dBias.PARAM,
) -> ModConvTranspose2d:
    """See :class:`ModConvTranspose2d`."""
    params = modulation_params or ModulationParams()
    return ModConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        modulation=params.modulation,
        aux_in_features=params.aux_in_features,
        fc_hidden_features=params.fc_hidden_features,
        fc_bias=params.fc_bias,
        fc_groups=params.fc_groups,
        fc_activation=params.fc_activation,
        num_weights=params.num_weights,
    )


def mod_conv3d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: IntOrTuple,
    modulation_params: ModulationParams | None = None,
    stride: IntOrTuple = 1,
    padding: IntOrTuple = 0,
    dilation: IntOrTuple = 1,
    bias: ModConv2dBias = ModConv2dBias.PARAM,
) -> ModConv3d:
    """See :class:`ModConv3d`."""
    params = modulation_params or ModulationParams()
    return ModConv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        modulation=params.modulation,
        aux_in_features=params.aux_in_features,
        fc_hidden_features=params.fc_hidden_features,
        fc_bias=params.fc_bias,
        fc_groups=params.fc_groups,
        fc_activation=params.fc_activation,
        num_weights=params.num_weights,
    )


def mod_conv_transpose3d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: IntOrTuple,
    modulation_params: ModulationParams | None = None,
    stride: IntOrTuple = 1,
    padding: IntOrTuple = 0,
    dilation: IntOrTuple = 1,
    bias: ModConv2dBias = ModConv2dBias.PARAM,
) -> ModConvTranspose3d:
    """See :class:`ModConvTranspose3d`."""
    params = modulation_params or ModulationParams()
    return ModConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        modulation=params.modulation,
        aux_in_features=params.aux_in_features,
        fc_hidden_features=params.fc_hidden_features,
        fc_bias=params.fc_bias,
        fc_groups=params.fc_groups,
        fc_activation=params.fc_activation,
        num_weights=params.num_weights,
    )


@dataclass(frozen=True)
class ModulationParams:
    """Shared modulation settings for modulated convolution layers."""

    modulation: ModConvType = ModConvType.NONE
    aux_in_features: int | None = None
    fc_hidden_features: int | tuple[int, ...] | None = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: int | None = None
    fc_bias: bool | None = True
