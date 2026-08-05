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
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from direct.nn.conv.modulated import (
    ModConv2dBias,
    ModConvActivation,
    ModConvType,
    ModulationParams,
    mod_conv2d,
)


class Conv2d(nn.Module):
    """Implementation of a simple cascade of 2D convolutions.

    If `batchnorm` is set to True, batch normalization layer is applied after each convolution.
    Supports modulated convolutions when `modulation` is not ModConvType.NONE.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_convs: int = 3,
        activation: nn.Module = nn.PReLU(),
        batchnorm: bool = False,
        modulation: ModConvType = ModConvType.NONE,
        modulation_params: Optional[ModulationParams] = None,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`Conv2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels.
        n_convs: int
            Number of convolutional layers.
        activation: nn.Module
            Activation function.
        batchnorm: bool
            If True a batch normalization layer is applied after every convolution.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of features in the auxiliary input for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features in the modulation MLP.
        fc_groups : int
            Groups for modulation MLP output interpolation. Default: 1.
        fc_activation : ModConvActivation
            Activation after modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()

        if modulation_params is None:
            modulation_params = ModulationParams(
                modulation=modulation,
                aux_in_features=aux_in_features,
                fc_hidden_features=fc_hidden_features,
                fc_groups=fc_groups,
                fc_activation=fc_activation,
                num_weights=num_weights,
            )
        self.modulation = modulation_params.modulation

        conv_layers: List[nn.Module] = []
        norm_layers: List[Optional[nn.Module]] = []
        act_layers: List[Optional[nn.Module]] = []

        for idx in range(n_convs):
            ic = in_channels if idx == 0 else hidden_channels
            oc = hidden_channels if idx != n_convs - 1 else out_channels

            conv_layers.append(
                mod_conv2d(
                    ic,
                    oc,
                    kernel_size=3,
                    padding=1,
                    bias=ModConv2dBias.PARAM,
                    modulation_params=modulation_params,
                )
            )
            if batchnorm:
                norm_layers.append(nn.BatchNorm2d(oc, eps=1e-4))
            else:
                norm_layers.append(None)
            if idx != n_convs - 1:
                act_layers.append(activation)
            else:
                act_layers.append(None)

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layers = nn.ModuleList([m for m in norm_layers if m is not None]) if batchnorm else None
        self.act_layers = act_layers
        self.n_convs = n_convs
        self.batchnorm = batchnorm

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the forward pass of :class:`Conv2d`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        y: torch.Tensor, optional
            Auxiliary signal for modulation of shape (N, aux_in_features).

        Returns
        -------
        out: torch.Tensor
            Convoluted output.
        """
        norm_idx = 0
        for idx in range(self.n_convs):
            if self.modulation != ModConvType.NONE:
                x = self.conv_layers[idx](x, y)
            else:
                x = self.conv_layers[idx](x)
            if self.batchnorm and self.norm_layers is not None:
                x = self.norm_layers[norm_idx](x)
                norm_idx += 1
            act_layer = self.act_layers[idx]
            if act_layer is not None:
                x = act_layer(x)
        return x


class CWNorm(nn.Module):
    """Centered Weight Normalization module for Conv layer weights."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_cwn = weight_ / norm
        return weight_cwn.view(weight.size())


class CWNConv2d(nn.Conv2d):
    """``Conv2d`` with centered weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        n_scale: float = 1.414,
        adjust_scale: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **kwargs)
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(n_scale)
        if adjust_scale:
            self.WnScale = Parameter(self.scale_)
        else:
            self.register_buffer("WnScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight) * self.WnScale
        return F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CWNConvTranspose2d(nn.ConvTranspose2d):
    """``ConvTranspose2d`` with centered weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, ...] = 1,
        n_scale: float = 1.414,
        adjust_scale: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            **kwargs,
        )
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(in_channels, 1, 1, 1).fill_(n_scale)
        if adjust_scale:
            self.WnScale = Parameter(self.scale_)
        else:
            self.register_buffer("WnScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight) * self.WnScale
        return F.conv_transpose2d(
            input_f, weight_q, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )


class CWNConv3d(nn.Conv3d):
    """``Conv3d`` with centered weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        n_scale: float = 1.414,
        adjust_scale: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **kwargs)
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1, 1).fill_(n_scale)
        if adjust_scale:
            self.WnScale = Parameter(self.scale_)
        else:
            self.register_buffer("WnScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight) * self.WnScale
        return F.conv3d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CWNConvTranspose3d(nn.ConvTranspose3d):
    """``ConvTranspose3d`` with centered weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, ...] = 1,
        n_scale: float = 1.414,
        adjust_scale: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            **kwargs,
        )
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(in_channels, 1, 1, 1, 1).fill_(n_scale)
        if adjust_scale:
            self.WnScale = Parameter(self.scale_)
        else:
            self.register_buffer("WnScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight) * self.WnScale
        return F.conv_transpose3d(
            input_f, weight_q, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
