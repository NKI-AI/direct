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
"""Modulated convolution layers based on [1]_.

These layers extend standard convolutions with input-dependent weight modulation,
allowing the network to dynamically adjust its convolutional filters based on an
auxiliary signal (e.g., acceleration factor, coil information).

References
----------

.. [1] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
    Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
    PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from direct.types import DirectEnum, IntOrTuple

__all__ = [
    "ModConv2d",
    "ModConv2dBias",
    "ModConv3d",
    "ModConvActivation",
    "ModConvTranspose2d",
    "ModConvTranspose3d",
    "ModConvType",
]


class ModConv2dBias(DirectEnum):
    LEARNED = "learned"
    PARAM = "param"
    NONE = "none"


class ModConvActivation(DirectEnum):
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"


class ModConvType(DirectEnum):
    FEATURES = "features"
    PARTIAL_IN = "partial_in"
    PARTIAL_OUT = "partial_out"
    FULL = "full"
    SUM = "sum"
    NONE = "none"


def _as_hidden_tuple(features: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(features, int):
        return (features,)
    return tuple(features)


def _resolved_fc_bias(fc_bias: bool | None) -> bool:
    return True if fc_bias is None else fc_bias


def _build_modulation_mlp(
    aux_in_features: int,
    hidden_features: tuple[int, ...],
    fc_activation: ModConvActivation,
    fc_bias: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(aux_in_features, hidden_features[0], bias=fc_bias)]
    for i in range(len(hidden_features) - 1):
        layers.append(nn.PReLU())
        layers.append(nn.Linear(hidden_features[i], hidden_features[i + 1]))
    if fc_activation == ModConvActivation.SIGMOID:
        layers.append(nn.Sigmoid())
    elif fc_activation == ModConvActivation.SOFTPLUS:
        layers.append(nn.Softplus())
    return nn.Sequential(*layers)


def _build_learned_bias_mlp(
    aux_in_features: int,
    hidden_features: tuple[int, ...],
    out_channels: int,
    fc_bias: bool,
) -> nn.Sequential:
    bias_layers: list[nn.Module] = [nn.Linear(aux_in_features, hidden_features[0], bias=fc_bias)]
    for i in range(len(hidden_features) - 1):
        bias_layers.append(nn.PReLU())
        bias_layers.append(
            nn.Linear(
                hidden_features[i],
                hidden_features[i + 1] if i != (len(hidden_features) - 2) else out_channels,
            )
        )
    return nn.Sequential(*bias_layers)


class ModConv2d(nn.Module):
    """Modulated 2D convolution based on [1]_.

    When ``modulation`` is :attr:`ModConvType.NONE` and ``bias`` is :attr:`ModConv2dBias.PARAM`,
    this behaves identically to :class:`torch.nn.Conv2d`.

    When modulation is enabled, the convolutional weights are element-wise scaled by
    an MLP-derived signal conditioned on an auxiliary input ``y``.

    References
    ----------

    .. [1] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrTuple,
        modulation: ModConvType = ModConvType.NONE,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: int | None = None,
        fc_hidden_features: int | tuple[int, ...] | None = None,
        fc_bias: bool | None = True,
        fc_groups: int | None = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        """Inits :class:`ModConv2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple of int
            Size of the convolutional kernel.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        stride : int or tuple of int
            Stride of the convolution. Default: 1.
        padding : int or tuple of int
            Padding added to all sides of the input. Default: 0.
        dilation : int or tuple of int
            Spacing between kernel elements. Default: 1.
        bias : ModConv2dBias
            Type of bias. Default: ModConv2dBias.PARAM.
        aux_in_features : int, optional
            Number of features in the auxiliary input ``y``.
        fc_hidden_features : int or tuple of int, optional
            Hidden features in the modulation MLP.
        fc_bias : bool, optional
            Whether the modulation MLP uses bias. Default: True.
        fc_groups : int or None
            If > 1, the MLP output is divided by fc_groups^2 and expanded via nearest interpolation.
            Default: 1.
        fc_activation : ModConvActivation
            Activation after the MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias
        self.fc_groups = 1 if fc_groups is None else fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        mod_hidden: tuple[int, ...] | None = None
        if modulation != ModConvType.NONE:
            if aux_in_features is None:
                raise ValueError("aux_in_features cannot be None when modulation is enabled.")
            if fc_hidden_features is None:
                raise ValueError("fc_hidden_features cannot be None when modulation is enabled.")
            if fc_groups is None:
                raise ValueError("fc_groups cannot be None when modulation is enabled.")
            if fc_groups < 1:
                raise ValueError("fc_groups must be >= 1.")

            hidden = _as_hidden_tuple(fc_hidden_features)

            if modulation == ModConvType.FEATURES:
                mod_out_features = (out_channels // fc_groups) * (in_channels // fc_groups)
            elif modulation == ModConvType.FULL:
                mod_out_features = (
                    (out_channels // fc_groups) * (in_channels // fc_groups) * self.kernel_size[0] * self.kernel_size[1]
                )
            elif modulation == ModConvType.PARTIAL_OUT:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (out_channels // fc_groups)
            elif modulation == ModConvType.PARTIAL_IN:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (in_channels // fc_groups)
            else:
                if (num_weights is None) or (num_weights < 1):
                    raise ValueError(f"ModConvType.SUM requires num_weights >= 1, got {num_weights}.")
                mod_out_features = num_weights

            mod_hidden = hidden + (mod_out_features,)
            self.fc = _build_modulation_mlp(
                aux_in_features,
                mod_hidden,
                fc_activation,
                _resolved_fc_bias(fc_bias),
            )

        weight_shape = (out_channels, in_channels, *self.kernel_size)
        if modulation == ModConvType.SUM:
            weight_shape = (num_weights,) + weight_shape
        k = math.sqrt(1 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = nn.Parameter(torch.FloatTensor(*weight_shape).uniform_(-k, k))

        self.bias_type = bias
        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if mod_hidden is None or aux_in_features is None:
                raise ValueError("ModConv2dBias.LEARNED requires modulation to be enabled.")
            self.bias = _build_learned_bias_mlp(
                aux_in_features,
                mod_hidden,
                out_channels,
                _resolved_fc_bias(fc_bias),
            )
        else:
            self.bias = None

    def __repr__(self):
        return (
            f"ModConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, bias={self.bias_type})"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, in_channels, H, W).
        y : torch.Tensor, optional
            Auxiliary signal of shape (N, aux_in_features).

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, H_out, W_out).
        """
        if self.modulation == ModConvType.NONE:
            out = F.conv2d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            fc_out = self.fc(y)

            if self.modulation == ModConvType.SUM:
                weight = (fc_out.view(x.shape[0], -1, 1, 1, 1, 1) * self.weight).sum(1)
            else:
                if self.modulation == ModConvType.FEATURES:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            1,
                            self.out_channels // self.fc_groups,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.out_channels, self.in_channels),
                            mode="nearest",
                        )
                    fc_out = fc_out.view(x.shape[0], self.out_channels, self.in_channels, 1, 1)

                elif self.modulation == ModConvType.FULL:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.out_channels // self.fc_groups,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(
                                self.kernel_size[1],
                                self.out_channels,
                                self.in_channels,
                            ),
                            mode="nearest",
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0],
                        self.out_channels,
                        self.in_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )

                elif self.modulation == ModConvType.PARTIAL_OUT:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.out_channels // self.fc_groups,
                            1,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.kernel_size[1], self.out_channels, 1),
                            mode="nearest",
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0],
                        self.out_channels,
                        1,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )

                else:  # PARTIAL_IN
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            1,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.kernel_size[1], 1, self.in_channels),
                            mode="nearest",
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0],
                        1,
                        self.in_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )

                weight = fc_out * self.weight

            out = torch.cat(
                [
                    F.conv2d(
                        x[i : i + 1],
                        weight[i],
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )

        if self.bias is not None:
            if isinstance(self.bias, nn.parameter.Parameter):
                bias = self.bias.view(1, -1, 1, 1)
            else:
                bias = self.bias(y).view(x.shape[0], -1, 1, 1)
            out = out + bias

        return out


class ModConvTranspose2d(nn.Module):
    """Modulated 2D transposed convolution.

    Transpose variant of :class:`ModConv2d` supporting the same modulation types.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrTuple,
        modulation: ModConvType = ModConvType.NONE,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: int | None = None,
        fc_hidden_features: int | tuple[int, ...] | None = None,
        fc_bias: bool | None = True,
        fc_groups: int | None = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        super().__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias
        self.fc_groups = 1 if fc_groups is None else fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        mod_hidden: tuple[int, ...] | None = None
        if modulation != ModConvType.NONE:
            if aux_in_features is None:
                raise ValueError("aux_in_features cannot be None when modulation is enabled.")
            if fc_hidden_features is None:
                raise ValueError("fc_hidden_features cannot be None when modulation is enabled.")
            if fc_groups is None:
                raise ValueError("fc_groups cannot be None when modulation is enabled.")
            if fc_groups < 1:
                raise ValueError("fc_groups must be >= 1.")

            hidden = _as_hidden_tuple(fc_hidden_features)

            if modulation == ModConvType.FEATURES:
                mod_out_features = (in_channels // fc_groups) * (out_channels // fc_groups)
            elif modulation == ModConvType.FULL:
                mod_out_features = (
                    (in_channels // fc_groups) * (out_channels // fc_groups) * self.kernel_size[0] * self.kernel_size[1]
                )
            elif modulation == ModConvType.PARTIAL_OUT:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (out_channels // fc_groups)
            elif modulation == ModConvType.PARTIAL_IN:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (in_channels // fc_groups)
            else:
                if (num_weights is None) or (num_weights < 1):
                    raise ValueError(f"ModConvType.SUM requires num_weights >= 1, got {num_weights}.")
                mod_out_features = num_weights

            mod_hidden = hidden + (mod_out_features,)
            self.fc = _build_modulation_mlp(
                aux_in_features,
                mod_hidden,
                fc_activation,
                _resolved_fc_bias(fc_bias),
            )

        weight_shape = (in_channels, out_channels, *self.kernel_size)
        if modulation == ModConvType.SUM:
            weight_shape = (num_weights,) + weight_shape
        k = math.sqrt(1 / (out_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = nn.Parameter(torch.FloatTensor(*weight_shape).uniform_(-k, k))

        self.bias_type = bias
        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if mod_hidden is None or aux_in_features is None:
                raise ValueError("ModConv2dBias.LEARNED requires modulation to be enabled.")
            self.bias = _build_learned_bias_mlp(
                aux_in_features,
                mod_hidden,
                out_channels,
                _resolved_fc_bias(fc_bias),
            )
        else:
            self.bias = None

    def __repr__(self):
        return (
            f"ModConvTranspose2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias_type})"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, in_channels, H, W).
        y : torch.Tensor, optional
            Auxiliary signal of shape (N, aux_in_features).

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, H_out, W_out).
        """
        if self.modulation == ModConvType.NONE:
            out = F.conv_transpose2d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            fc_out = self.fc(y)

            if self.modulation == ModConvType.SUM:
                weight = (fc_out.view(x.shape[0], -1, 1, 1, 1, 1) * self.weight).sum(1)
            else:
                if self.modulation == ModConvType.FEATURES:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            1,
                            self.in_channels // self.fc_groups,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.in_channels, self.out_channels),
                            mode="nearest",
                        )
                    fc_out = fc_out.view(x.shape[0], self.in_channels, self.out_channels, 1, 1)

                elif self.modulation == ModConvType.FULL:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.in_channels // self.fc_groups,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(
                                self.kernel_size[1],
                                self.in_channels,
                                self.out_channels,
                            ),
                            mode="nearest",
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0],
                        self.in_channels,
                        self.out_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )

                elif self.modulation == ModConvType.PARTIAL_OUT:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            1,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.kernel_size[1], 1, self.out_channels),
                            mode="nearest",
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0],
                        1,
                        self.out_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )

                else:  # PARTIAL_IN
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.in_channels // self.fc_groups,
                            1,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.kernel_size[1], self.in_channels, 1),
                            mode="nearest",
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0],
                        self.in_channels,
                        1,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )

                weight = fc_out * self.weight

            out = torch.cat(
                [
                    F.conv_transpose2d(
                        x[i : i + 1],
                        weight[i],
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )

        if self.bias is not None:
            if isinstance(self.bias, nn.parameter.Parameter):
                bias = self.bias.view(1, -1, 1, 1)
            else:
                bias = self.bias(y).view(x.shape[0], -1, 1, 1)
            out = out + bias

        return out


class ModConv3d(nn.Module):
    """Modulated 3D convolution.

    3D extension of :class:`ModConv2d` supporting the same modulation types.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrTuple,
        modulation: ModConvType = ModConvType.NONE,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: int | None = None,
        fc_hidden_features: int | tuple[int, ...] | None = None,
        fc_bias: bool | None = True,
        fc_groups: int | None = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        elif len(kernel_size) == 2:
            self.kernel_size = (kernel_size[0], kernel_size[0], kernel_size[1])
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias
        self.fc_groups = 1 if fc_groups is None else fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        mod_hidden: tuple[int, ...] | None = None
        if modulation != ModConvType.NONE:
            if aux_in_features is None:
                raise ValueError("aux_in_features cannot be None when modulation is enabled.")
            if fc_hidden_features is None:
                raise ValueError("fc_hidden_features cannot be None when modulation is enabled.")
            if fc_groups is None:
                raise ValueError("fc_groups cannot be None when modulation is enabled.")
            if fc_groups < 1:
                raise ValueError("fc_groups must be >= 1.")

            hidden = _as_hidden_tuple(fc_hidden_features)

            if modulation == ModConvType.FEATURES:
                mod_out_features = (out_channels // fc_groups) * (in_channels // fc_groups)
            elif modulation == ModConvType.FULL:
                mod_out_features = (
                    (out_channels // fc_groups)
                    * (in_channels // fc_groups)
                    * self.kernel_size[0]
                    * self.kernel_size[1]
                    * self.kernel_size[2]
                )
            elif modulation == ModConvType.PARTIAL_OUT:
                mod_out_features = (
                    self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (out_channels // fc_groups)
                )
            elif modulation == ModConvType.PARTIAL_IN:
                mod_out_features = (
                    self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (in_channels // fc_groups)
                )
            else:
                if (num_weights is None) or (num_weights < 1):
                    raise ValueError(f"ModConvType.SUM requires num_weights >= 1, got {num_weights}.")
                mod_out_features = num_weights

            mod_hidden = hidden + (mod_out_features,)
            self.fc = _build_modulation_mlp(
                aux_in_features,
                mod_hidden,
                fc_activation,
                _resolved_fc_bias(fc_bias),
            )

        weight_shape = (out_channels, in_channels, *self.kernel_size)
        if modulation == ModConvType.SUM:
            weight_shape = (num_weights,) + weight_shape
        k = math.sqrt(1 / (in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]))
        self.weight = nn.Parameter(torch.FloatTensor(*weight_shape).uniform_(-k, k))

        self.bias_type = bias
        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if mod_hidden is None or aux_in_features is None:
                raise ValueError("ModConv2dBias.LEARNED requires modulation to be enabled.")
            self.bias = _build_learned_bias_mlp(
                aux_in_features,
                mod_hidden,
                out_channels,
                _resolved_fc_bias(fc_bias),
            )
        else:
            self.bias = None

    def __repr__(self):
        return (
            f"ModConv3d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias_type})"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, in_channels, D, H, W).
        y : torch.Tensor, optional
            Auxiliary signal of shape (N, aux_in_features).

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, D_out, H_out, W_out).
        """
        if self.modulation == ModConvType.NONE:
            out = F.conv3d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            fc_out = self.fc(y)

            if self.modulation == ModConvType.SUM:
                weight = (fc_out.view(x.shape[0], -1, 1, 1, 1, 1, 1) * self.weight).sum(1)
            else:
                if self.modulation == ModConvType.FEATURES:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            1,
                            self.out_channels // self.fc_groups,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.out_channels, self.in_channels),
                            mode="nearest",
                        )
                    fc_out = fc_out.view(x.shape[0], self.out_channels, self.in_channels, 1, 1, 1)

                elif self.modulation == ModConvType.FULL:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            self.out_channels // self.fc_groups,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = fc_out.permute(0, 4, 5, 1, 2, 3)
                        out_ch_expand = self.out_channels // self.fc_groups
                        in_ch_expand = self.in_channels // self.fc_groups
                        if out_ch_expand < self.out_channels or in_ch_expand < self.in_channels:
                            fc_out = fc_out.repeat(1, self.fc_groups, self.fc_groups, 1, 1, 1)
                            fc_out = fc_out[:, : self.out_channels, : self.in_channels, :, :, :]
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.out_channels,
                            self.in_channels,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                        )
                    else:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.out_channels,
                            self.in_channels,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                        )

                elif self.modulation == ModConvType.PARTIAL_OUT:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            self.out_channels // self.fc_groups,
                            1,
                        )
                        fc_out = fc_out.permute(0, 4, 5, 1, 2, 3)
                    fc_out = fc_out.view(
                        x.shape[0],
                        self.out_channels,
                        1,
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.kernel_size[2],
                    )

                else:  # PARTIAL_IN
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            1,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = fc_out.permute(0, 4, 5, 1, 2, 3)
                    fc_out = fc_out.view(
                        x.shape[0],
                        1,
                        self.in_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.kernel_size[2],
                    )

                weight = fc_out * self.weight

            out = torch.cat(
                [
                    F.conv3d(
                        x[i : i + 1],
                        weight[i],
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )

        if self.bias is not None:
            if isinstance(self.bias, nn.parameter.Parameter):
                bias = self.bias.view(1, -1, 1, 1, 1)
            else:
                bias = self.bias(y).view(x.shape[0], -1, 1, 1, 1)
            out = out + bias

        return out


class ModConvTranspose3d(nn.Module):
    """Modulated 3D transposed convolution.

    3D transpose variant of :class:`ModConv3d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrTuple,
        modulation: ModConvType = ModConvType.NONE,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: int | None = None,
        fc_hidden_features: int | tuple[int, ...] | None = None,
        fc_bias: bool | None = True,
        fc_groups: int | None = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        elif len(kernel_size) == 2:
            self.kernel_size = (kernel_size[0], kernel_size[0], kernel_size[1])
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias
        self.fc_groups = 1 if fc_groups is None else fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        mod_hidden: tuple[int, ...] | None = None
        if modulation != ModConvType.NONE:
            if aux_in_features is None:
                raise ValueError("aux_in_features cannot be None when modulation is enabled.")
            if fc_hidden_features is None:
                raise ValueError("fc_hidden_features cannot be None when modulation is enabled.")
            if fc_groups is None:
                raise ValueError("fc_groups cannot be None when modulation is enabled.")
            if fc_groups < 1:
                raise ValueError("fc_groups must be >= 1.")

            hidden = _as_hidden_tuple(fc_hidden_features)

            if modulation == ModConvType.FEATURES:
                mod_out_features = (in_channels // fc_groups) * (out_channels // fc_groups)
            elif modulation == ModConvType.FULL:
                mod_out_features = (
                    (in_channels // fc_groups)
                    * (out_channels // fc_groups)
                    * self.kernel_size[0]
                    * self.kernel_size[1]
                    * self.kernel_size[2]
                )
            elif modulation == ModConvType.PARTIAL_OUT:
                mod_out_features = (
                    self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (out_channels // fc_groups)
                )
            elif modulation == ModConvType.PARTIAL_IN:
                mod_out_features = (
                    self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (in_channels // fc_groups)
                )
            else:
                if (num_weights is None) or (num_weights < 1):
                    raise ValueError(f"ModConvType.SUM requires num_weights >= 1, got {num_weights}.")
                mod_out_features = num_weights

            mod_hidden = hidden + (mod_out_features,)
            self.fc = _build_modulation_mlp(
                aux_in_features,
                mod_hidden,
                fc_activation,
                _resolved_fc_bias(fc_bias),
            )

        weight_shape = (in_channels, out_channels, *self.kernel_size)
        if modulation == ModConvType.SUM:
            weight_shape = (num_weights,) + weight_shape
        k = math.sqrt(1 / (out_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]))
        self.weight = nn.Parameter(torch.FloatTensor(*weight_shape).uniform_(-k, k))

        self.bias_type = bias
        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if mod_hidden is None or aux_in_features is None:
                raise ValueError("ModConv2dBias.LEARNED requires modulation to be enabled.")
            self.bias = _build_learned_bias_mlp(
                aux_in_features,
                mod_hidden,
                out_channels,
                _resolved_fc_bias(fc_bias),
            )
        else:
            self.bias = None

    def __repr__(self):
        return (
            f"ModConvTranspose3d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias_type})"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, in_channels, D, H, W).
        y : torch.Tensor, optional
            Auxiliary signal of shape (N, aux_in_features).

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, D_out, H_out, W_out).
        """
        if self.modulation == ModConvType.NONE:
            out = F.conv_transpose3d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            fc_out = self.fc(y)

            if self.modulation == ModConvType.SUM:
                weight = (fc_out.view(x.shape[0], -1, 1, 1, 1, 1, 1) * self.weight).sum(1)
            else:
                if self.modulation == ModConvType.FEATURES:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            1,
                            self.in_channels // self.fc_groups,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = F.interpolate(
                            fc_out,
                            size=(self.in_channels, self.out_channels),
                            mode="nearest",
                        )
                    fc_out = fc_out.view(x.shape[0], self.in_channels, self.out_channels, 1, 1, 1)

                elif self.modulation == ModConvType.FULL:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            self.in_channels // self.fc_groups,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = fc_out.permute(0, 4, 5, 1, 2, 3)
                        in_ch_expand = self.in_channels // self.fc_groups
                        out_ch_expand = self.out_channels // self.fc_groups
                        if out_ch_expand < self.out_channels or in_ch_expand < self.in_channels:
                            fc_out = fc_out.repeat(1, self.fc_groups, self.fc_groups, 1, 1, 1)
                            fc_out = fc_out[:, : self.in_channels, : self.out_channels, :, :, :]
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.in_channels,
                            self.out_channels,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                        )
                    else:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.in_channels,
                            self.out_channels,
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                        )

                elif self.modulation == ModConvType.PARTIAL_OUT:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            1,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = fc_out.permute(0, 4, 5, 1, 2, 3)
                        if self.out_channels // self.fc_groups < self.out_channels:
                            fc_out = fc_out.repeat(1, 1, self.fc_groups, 1, 1, 1)
                            fc_out = fc_out[:, :, : self.out_channels, :, :, :]
                    fc_out = fc_out.view(
                        x.shape[0],
                        1,
                        self.out_channels,
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.kernel_size[2],
                    )

                else:  # PARTIAL_IN
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.kernel_size[2],
                            self.in_channels // self.fc_groups,
                            1,
                        )
                        fc_out = fc_out.permute(0, 4, 5, 1, 2, 3)
                        if self.in_channels // self.fc_groups < self.in_channels:
                            fc_out = fc_out.repeat(1, self.fc_groups, 1, 1, 1, 1)
                            fc_out = fc_out[:, : self.in_channels, :, :, :, :]
                    fc_out = fc_out.view(
                        x.shape[0],
                        self.in_channels,
                        1,
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.kernel_size[2],
                    )

                weight = fc_out * self.weight

            out = torch.cat(
                [
                    F.conv_transpose3d(
                        x[i : i + 1],
                        weight[i],
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )

        if self.bias is not None:
            if isinstance(self.bias, nn.parameter.Parameter):
                bias = self.bias.view(1, -1, 1, 1, 1)
            else:
                bias = self.bias(y).view(x.shape[0], -1, 1, 1, 1)
            out = out + bias

        return out
