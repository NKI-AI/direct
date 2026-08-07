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

import torch
from torch import nn

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
        modulation_params: ModulationParams | None = None,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
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

        conv_layers: list[nn.Module] = []
        norm_layers: list[nn.Module | None] = []
        act_layers: list[nn.Module | None] = []

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

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
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
