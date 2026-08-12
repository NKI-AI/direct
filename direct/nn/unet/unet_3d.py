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

"""Code for three-dimensional U-Net adapted from the 2D variant."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from direct.nn.adain.adain import AdaIN3d, NormType
from direct.nn.conv.modulated import ModConv2dBias, ModConv3d, ModConvActivation, ModConvTranspose3d, ModConvType


class ConvModule3D(nn.Module):
    """Single 3D convolution + norm + activation + dropout module supporting modulated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dropout_probability: float,
        modulation: ModConvType = ModConvType.NONE,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: tuple[int] | int | None = None,
    ):
        super().__init__()

        self.modulation = modulation
        self.norm_type = norm_type

        self.conv = ModConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )
        if norm_type == NormType.ADAIN:
            if adain_hidden_features is None:
                raise ValueError("AdaIN hidden features must be provided if norm_type is NormType.ADAIN.")
            if aux_in_features is None:
                raise ValueError("aux_in_features must be provided if norm_type is NormType.ADAIN.")
            self.instance_norm = AdaIN3d(
                num_channels=out_channels,
                aux_in_features=aux_in_features,
                hidden_features=adain_hidden_features,
            )
        else:
            self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(dropout_probability)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if self.modulation != ModConvType.NONE:
            x = self.conv(x, y)
        else:
            x = self.conv(x)

        if self.norm_type == NormType.ADAIN:
            if y is None:
                raise ValueError("AdaIN requires aux vector y, but got None.")
            x = self.instance_norm(x, y)
        else:
            x = self.instance_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


class ConvBlock3D(nn.Module):
    """3D U-Net convolutional block with optional modulation and AdaIN support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_probability: float,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: tuple[int] | int | None = None,
    ) -> None:
        """Inits :class:`ConvBlock3D`.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolutional layers.
        dropout_probability : float
            Dropout probability applied after convolutional layers.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of auxiliary input features.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for the modulation MLP.
        fc_groups : int
            Groups for the modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for the modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        norm_type : NormType
            Normalization type. Default: NormType.INSTANCE.
        adain_hidden_features : int or tuple of int, optional
            Hidden features for AdaIN.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability
        self.modulation = modulation
        self.norm_type = norm_type

        self.layer_1 = ConvModule3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dropout_probability=dropout_probability,
            bias=(ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED),
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            norm_type=norm_type,
            adain_hidden_features=adain_hidden_features,
        )
        self.layer_2 = ConvModule3D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dropout_probability=dropout_probability,
            bias=(ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED),
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            norm_type=norm_type,
            adain_hidden_features=adain_hidden_features,
        )

    def forward(self, input_data: torch.Tensor, aux_data: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock3D`.

        Parameters
        ----------
        input_data : torch.Tensor
        aux_data : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
            return self.layer_2(self.layer_1(input_data, aux_data), aux_data)
        return self.layer_2(self.layer_1(input_data))


class TransposeConvBlock3D(nn.Module):
    """3D U-Net Transpose Convolutional Block with optional modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: tuple[int] | int | None = None,
    ) -> None:
        """Inits :class:`TransposeConvBlock3D`.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolutional layers.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of auxiliary input features.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for the modulation MLP.
        fc_groups : int
            Groups for the modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for the modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        norm_type : NormType
            Normalization type. Default: NormType.INSTANCE.
        adain_hidden_features : int or tuple of int, optional
            Hidden features for AdaIN.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modulation = modulation
        self.norm_type = norm_type

        self.conv = ModConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            modulation=modulation,
            bias=(ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED),
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )
        if norm_type == NormType.ADAIN:
            if adain_hidden_features is None:
                raise ValueError("AdaIN hidden features must be provided if norm_type is NormType.ADAIN.")
            if aux_in_features is None:
                raise ValueError("aux_in_features must be provided if norm_type is NormType.ADAIN.")
            self.instance_norm = AdaIN3d(
                num_channels=out_channels,
                aux_in_features=aux_in_features,
                hidden_features=adain_hidden_features,
            )
        else:
            self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_data: torch.Tensor, aux_data: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass of :class:`TransposeConvBlock3D`.

        Parameters
        ----------
        input_data : torch.Tensor
        aux_data : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        if self.modulation != ModConvType.NONE:
            x = self.conv(input_data, aux_data)
        else:
            x = self.conv(input_data)
        if self.norm_type == NormType.ADAIN:
            x = self.instance_norm(x, aux_data)
        else:
            x = self.instance_norm(x)
        return self.leaky_relu(x)


class UnetModel3d(nn.Module):
    """PyTorch implementation of a 3D U-Net model with optional modulated convolutions.

    This class defines a 3D U-Net architecture consisting of down-sampling and up-sampling layers with 3D convolutional
    blocks. This is an extension to 3D volumes of :class:`direct.nn.unet.unet_2d.UnetModel2d`.
    Modulated convolutions are based on [1]_.

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
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        modulation_at_input: bool = False,
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: tuple[int] | int | None = None,
    ) -> None:
        """Inits :class:`UnetModel3d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_filters : int
            Number of output channels of the first convolutional layer.
        num_pool_layers : int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability : float
            Dropout probability.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of auxiliary input features.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for the modulation MLP.
        fc_groups : int
            Groups for the modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for the modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        modulation_at_input : bool
            If True, only the first conv block uses modulation. Default: False.
        norm_type : NormType
            Normalization type. Default: NormType.INSTANCE.
        adain_hidden_features : int or tuple of int, optional
            Hidden features for AdaIN.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability
        self.modulation = modulation
        self.norm_type = norm_type

        self.down_sample_layers = nn.ModuleList(
            [
                ConvBlock3D(
                    in_channels,
                    num_filters,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                    norm_type,
                    adain_hidden_features,
                )
            ]
        )
        ch = num_filters

        if modulation != ModConvType.NONE and modulation_at_input:
            modulation = ModConvType.NONE

        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [
                ConvBlock3D(
                    ch,
                    ch * 2,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                    norm_type,
                    adain_hidden_features,
                )
            ]
            ch *= 2
        self.conv = ConvBlock3D(
            ch,
            ch * 2,
            dropout_probability,
            modulation,
            aux_in_features,
            fc_hidden_features,
            fc_groups,
            fc_activation,
            num_weights,
            norm_type,
            adain_hidden_features,
        )

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [
                TransposeConvBlock3D(
                    ch * 2,
                    ch,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                    norm_type,
                    adain_hidden_features,
                )
            ]
            self.up_conv += [
                ConvBlock3D(
                    ch * 2,
                    ch,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                    norm_type,
                    adain_hidden_features,
                )
            ]
            ch //= 2

        if modulation != ModConvType.NONE and modulation_at_input:
            modulation = ModConvType.NONE
        self.up_transpose_conv += [
            TransposeConvBlock3D(
                ch * 2,
                ch,
                modulation,
                aux_in_features,
                fc_hidden_features,
                fc_groups,
                fc_activation,
                num_weights,
                norm_type,
                adain_hidden_features,
            )
        ]
        self.up_conv += [
            nn.Sequential(
                ConvBlock3D(
                    ch * 2,
                    ch,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                    norm_type,
                    adain_hidden_features,
                ),
                nn.Conv3d(ch, out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor, aux_data: torch.Tensor | None = None) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel3d`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor of shape (N, in_channels, slice/time, height, width).
        aux_data : torch.Tensor, optional
            Auxiliary data for modulation/AdaIN.

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, slice/time, height, width).
        """
        stack = []
        output, inp_pad = pad_to_pow_of_2(input_data, self.num_pool_layers)

        for layer in self.down_sample_layers:
            if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
                output = layer(output, aux_data)
            else:
                output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
            output = self.conv(output, aux_data)
        else:
            output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
                output = transpose_conv(output, aux_data)
            else:
                output = transpose_conv(output)

            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
                if isinstance(conv, nn.Sequential):
                    output = conv[0](output, aux_data)
                    output = conv[1](output)
                else:
                    output = conv(output, aux_data)
            else:
                output = conv(output)

        if sum(inp_pad) != 0:
            output = output[
                :,
                :,
                inp_pad[4] : output.shape[2] - inp_pad[5],
                inp_pad[2] : output.shape[3] - inp_pad[3],
                inp_pad[0] : output.shape[4] - inp_pad[1],
            ]

        return output


class NormUnetModel3d(nn.Module):
    """Implementation of a Normalized U-Net model for 3D data with optional modulation.

    This is an extension to 3D volumes of :class:`direct.nn.unet.unet_2d.NormUnetModel2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        norm_groups: int = 2,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        modulation_at_input: bool = False,
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: tuple[int] | int | None = None,
    ) -> None:
        """Inits :class:`NormUnetModel3d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_filters : int
            Number of output channels of the first convolutional layer.
        num_pool_layers : int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability : float
            Dropout probability.
        norm_groups: int
            Number of normalization groups.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of auxiliary input features.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for the modulation MLP.
        fc_groups : int
            Groups for the modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for the modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        modulation_at_input : bool
            If True, only the first conv block uses modulation. Default: False.
        norm_type : NormType
            Normalization type. Default: NormType.INSTANCE.
        adain_hidden_features : int or tuple of int, optional
            Hidden features for AdaIN.
        """
        super().__init__()

        self.unet3d = UnetModel3d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            modulation_at_input=modulation_at_input,
            norm_type=norm_type,
            adain_hidden_features=adain_hidden_features,
        )

        self.norm_groups = norm_groups
        self.modulation = modulation
        self.norm_type = norm_type

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        b, c, z, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, z, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, z, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, z, h, w)

    @staticmethod
    def pad(
        input_data: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[list[int], list[int], list[int], int, int, int]]:
        _, _, z, h, w = input_data.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        z_mult = ((z - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        z_pad = [math.floor((z_mult - z) / 2), math.ceil((z_mult - z) / 2)]

        output = F.pad(input_data, w_pad + h_pad + z_pad)
        return output, (h_pad, w_pad, z_pad, h_mult, w_mult, z_mult)

    @staticmethod
    def unpad(
        input_data: torch.Tensor,
        h_pad: list[int],
        w_pad: list[int],
        z_pad: list[int],
        h_mult: int,
        w_mult: int,
        z_mult: int,
    ) -> torch.Tensor:
        return input_data[
            ...,
            z_pad[0] : z_mult - z_pad[1],
            h_pad[0] : h_mult - h_pad[1],
            w_pad[0] : w_mult - w_pad[1],
        ]

    def forward(self, input_data: torch.Tensor, aux_data: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass of :class:`NormUnetModel3d`.

        Parameters
        ----------
        input_data : torch.Tensor
        aux_data : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)

        if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
            output = self.unet3d(output, aux_data)
        else:
            output = self.unet3d(output)

        h_pad, w_pad, z_pad, h_mult, w_mult, z_mult = pad_sizes
        output = self.unpad(output, h_pad, w_pad, z_pad, h_mult, w_mult, z_mult)
        output = self.unnorm(output, mean, std, self.norm_groups)

        return output


def pad_to_pow_of_2(inp: torch.Tensor, k: int) -> tuple[torch.Tensor, list[int]]:
    """Pads the input tensor along the spatial dimensions to the nearest power of 2.

    Parameters
    ----------
    inp : torch.Tensor
        The input tensor to be padded.
    k : int
        The exponent to which 2 is raised to determine target dimension size.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        A tuple containing the padded tensor and the padding list.
    """
    diffs = [_ - 2**k for _ in inp.shape[2:]]
    padding = [0, 0, 0, 0, 0, 0]
    for i, diff in enumerate(diffs[::-1]):
        if diff < 1:
            padding[2 * i] = abs(diff) // 2
            padding[2 * i + 1] = abs(diff) - padding[2 * i]

    if sum(padding) > 0:
        inp = F.pad(inp, padding)

    return inp, padding
