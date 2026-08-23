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
"""direct.nn.didn.didn module."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from direct.nn.conv.modulated import (
    ModConv2d,
    ModConv2dBias,
    ModConvActivation,
    ModConvType,
    ModulationParams,
    mod_conv2d,
)


class Subpixel(nn.Module):
    """Subpixel convolution layer for up-scaling of low resolution features at super-resolution as implemented in [#]_.

    References:
        .. [#] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising."
            CVPRW, 2019. https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        kernel_size: int | tuple[int, int],
        padding: int = 0,
        modulation: ModConvType = ModConvType.NONE,
        modulation_params: ModulationParams | None = None,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        """Inits :class:`Subpixel`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            upscale_factor: Subpixel upscale factor.
            kernel_size: Convolution kernel size.
            padding: Padding size. Default is ``0``.
            modulation: Modulation type. Default is ``ModConvType.NONE``.
            aux_in_features: Auxiliary input features for modulation.
            fc_hidden_features: Hidden features for modulation MLP.
            fc_groups: Groups for modulation MLP. Default is ``1``.
            fc_activation: Activation for modulation MLP. Default is ``ModConvActivation.SIGMOID``.
            num_weights: Number of weight bases for ModConvType.SUM.
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
        self.conv = mod_conv2d(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=kernel_size,
            padding=padding,
            bias=ModConv2dBias.PARAM,
            modulation_params=modulation_params,
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Computes :class:`Subpixel` convolution on input torch.Tensor ``x``.

        Args:
            x: Input tensor.
            y: Auxiliary signal for modulation.
        """
        if self.modulation != ModConvType.NONE:
            return self.pixelshuffle(self.conv(x, y))
        return self.pixelshuffle(self.conv(x))


class ReconBlock(nn.Module):
    """Reconstruction Block of :class:`DIDN` model as implemented in [#]_.

    References:
        .. [#] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising."
            CVPRW, 2019. https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        num_convs: int,
        modulation: ModConvType = ModConvType.NONE,
        modulation_params: ModulationParams | None = None,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        """Inits :class:`ReconBlock`.

        Args:
            in_channels: Number of input channels.
            num_convs: Number of convolution blocks.
            modulation: Modulation type. Default is ``ModConvType.NONE``.
            aux_in_features: Auxiliary input features for modulation.
            fc_hidden_features: Hidden features for modulation MLP.
            fc_groups: Groups for modulation MLP. Default is ``1``.
            fc_activation: Activation for modulation MLP. Default is ``ModConvActivation.SIGMOID``.
            num_weights: Number of weight bases for ModConvType.SUM.
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
        self.num_convs = num_convs
        self.activations = nn.ModuleList([nn.PReLU() for _ in range(num_convs - 1)])

        self.convs = nn.ModuleList()
        for _ in range(num_convs):
            self.convs.append(
                mod_conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=ModConv2dBias.PARAM,
                    modulation_params=modulation_params,
                )
            )

    def forward(self, input_data: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Computes num_convs convolutions followed by PReLU activation on `input_data`.

        Args:
            input_data: Input tensor.
            y: Auxiliary signal for modulation.
        """
        output = input_data.clone()
        for idx in range(self.num_convs):
            if self.modulation != ModConvType.NONE:
                output = self.convs[idx](output, y)
            else:
                output = self.convs[idx](output)
            if idx < self.num_convs - 1:
                output = self.activations[idx](output)

        return input_data + output


class DUB(nn.Module):
    """Down-up block (DUB) for :class:`DIDN` model as implemented in [#]_.

    References:
        .. [#] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising."
            CVPRW, 2019. https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modulation: ModConvType = ModConvType.NONE,
        modulation_params: ModulationParams | None = None,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        """Inits :class:`DUB`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modulation: Modulation type. Default is ``ModConvType.NONE``.
            aux_in_features: Auxiliary input features for modulation.
            fc_hidden_features: Hidden features for modulation MLP.
            fc_groups: Groups for modulation MLP. Default is ``1``.
            fc_activation: Activation for modulation MLP. Default is ``ModConvActivation.SIGMOID``.
            num_weights: Number of weight bases for ModConvType.SUM.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
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

        # Scale 1 - down path
        self.conv1_1_a = mod_conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, modulation_params=modulation_params
        )
        self.conv1_1_a_act = nn.PReLU()
        self.conv1_1_b = mod_conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, modulation_params=modulation_params
        )
        self.conv1_1_b_act = nn.PReLU()
        self.down1 = mod_conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            modulation_params=modulation_params,
        )

        # Scale 2 - down path
        self.conv2_1 = mod_conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            modulation_params=modulation_params,
        )
        self.conv2_1_act = nn.PReLU()
        self.down2 = mod_conv2d(
            in_channels * 2,
            in_channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            modulation_params=modulation_params,
        )

        # Scale 3 - bottom
        self.conv3_1 = mod_conv2d(
            in_channels * 4,
            in_channels * 4,
            kernel_size=3,
            padding=1,
            modulation_params=modulation_params,
        )
        self.conv3_1_act = nn.PReLU()
        self.up1 = Subpixel(in_channels * 4, in_channels * 2, 2, 1, 0, modulation_params=modulation_params)

        # Scale 2 - up path
        self.conv_agg_1 = mod_conv2d(
            in_channels * 4,
            in_channels * 2,
            kernel_size=1,
            padding=0,
            modulation_params=modulation_params,
        )
        self.conv2_2 = mod_conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            modulation_params=modulation_params,
        )
        self.conv2_2_act = nn.PReLU()
        self.up2 = Subpixel(in_channels * 2, in_channels, 2, 1, 0, modulation_params=modulation_params)

        # Scale 1 - up path
        self.conv_agg_2 = mod_conv2d(
            in_channels * 2, in_channels, kernel_size=1, padding=0, modulation_params=modulation_params
        )
        self.conv1_2_a = mod_conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, modulation_params=modulation_params
        )
        self.conv1_2_a_act = nn.PReLU()
        self.conv1_2_b = mod_conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, modulation_params=modulation_params
        )
        self.conv1_2_b_act = nn.PReLU()
        self.conv_out = mod_conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, modulation_params=modulation_params
        )
        self.conv_out_act = nn.PReLU()

    def _conv(self, layer: ModConv2d, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """Conv.

        Args:
            layer: Layer.
            x: X.
            y: Y.

        Returns:
            The result.
        """
        if self.modulation != ModConvType.NONE:
            return layer(x, y)
        return layer(x)

    @staticmethod
    def pad(x: torch.Tensor) -> torch.Tensor:
        """Pads input to height and width dimensions if odd."""
        padding = [0, 0, 0, 0]
        if x.shape[-2] % 2 != 0:
            padding[3] = 1
        if x.shape[-1] % 2 != 0:
            padding[1] = 1
        if sum(padding) != 0:
            x = F.pad(x, padding, "reflect")
        return x

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        r"""Crops ``x`` to specified shape.

        Args:
            x: Input tensor with shape (\*, H, W).
            shape: Crop shape corresponding to H, W.

        Returns:
            Cropped tensor.
        """
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            y: Auxiliary signal for modulation.

        Returns:
            DUB output.
        """
        x1 = self.pad(x.clone())
        x1 = x1 + self.conv1_1_b_act(
            self._conv(self.conv1_1_b, self.conv1_1_a_act(self._conv(self.conv1_1_a, x1, y)), y)
        )
        x2 = self._conv(self.down1, x1, y)
        x2 = x2 + self.conv2_1_act(self._conv(self.conv2_1, x2, y))
        out = self._conv(self.down2, x2, y)
        out = out + self.conv3_1_act(self._conv(self.conv3_1, out, y))

        if self.modulation != ModConvType.NONE:
            out = self.up1(out, y)
        else:
            out = self.up1(out)

        out = torch.cat([x2, self.crop_to_shape(out, (x2.shape[-2], x2.shape[-1]))], dim=1)
        out = self._conv(self.conv_agg_1, out, y)
        out = out + self.conv2_2_act(self._conv(self.conv2_2, out, y))

        if self.modulation != ModConvType.NONE:
            out = self.up2(out, y)
        else:
            out = self.up2(out)

        out = torch.cat([x1, self.crop_to_shape(out, (x1.shape[-2], x1.shape[-1]))], dim=1)
        out = self._conv(self.conv_agg_2, out, y)
        out = out + self.conv1_2_b_act(
            self._conv(
                self.conv1_2_b,
                self.conv1_2_a_act(self._conv(self.conv1_2_a, out, y)),
                y,
            )
        )
        out = x + self.crop_to_shape(
            self.conv_out_act(self._conv(self.conv_out, out, y)),
            (x.shape[-2], x.shape[-1]),
        )
        return out


class DIDN(nn.Module):
    """Deep Iterative Down-up convolutional Neural network (DIDN) implementation as in [#]_.

    References:
        .. [#] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising."
            CVPRW, 2019. https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_dubs: int = 6,
        num_convs_recon: int = 9,
        skip_connection: bool = False,
        modulation: ModConvType = ModConvType.NONE,
        modulation_params: ModulationParams | None = None,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
    ):
        """Inits :class:`DIDN`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of hidden channels. First convolution out_channels. Default is ``128``.
            num_dubs: Number of DUB networks. Default is ``6``.
            num_convs_recon: Number of ReconBlock convolutions. Default is ``9``.
            skip_connection: Use skip connection. Default is ``False``.
            modulation: Modulation type. Default is ``ModConvType.NONE``.
            aux_in_features: Auxiliary input features for modulation.
            fc_hidden_features: Hidden features for modulation MLP.
            fc_groups: Groups for modulation MLP. Default is ``1``.
            fc_activation: Activation for modulation MLP. Default is ``ModConvActivation.SIGMOID``.
            num_weights: Number of weight bases for ModConvType.SUM.
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

        self.conv_in = mod_conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            modulation_params=modulation_params,
        )
        self.conv_in_act = nn.PReLU()
        self.down = mod_conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            modulation_params=modulation_params,
        )
        self.dubs = nn.ModuleList(
            [
                DUB(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    modulation_params=modulation_params,
                )
                for _ in range(num_dubs)
            ]
        )
        self.recon_blocks = nn.ModuleList(
            [
                ReconBlock(in_channels=hidden_channels, num_convs=num_convs_recon, modulation_params=modulation_params)
                for _ in range(num_dubs)
            ]
        )
        self.recon_agg = mod_conv2d(
            in_channels=hidden_channels * num_dubs,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            modulation_params=modulation_params,
        )
        self.conv = mod_conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            modulation_params=modulation_params,
        )
        self.conv_act = nn.PReLU()
        self.up2 = Subpixel(hidden_channels, hidden_channels, 2, 1, modulation_params=modulation_params)
        self.conv_out = mod_conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            modulation_params=modulation_params,
        )
        self.num_dubs = num_dubs
        self.skip_connection = (in_channels == out_channels) and skip_connection

    def _conv(self, layer: ModConv2d, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """Conv.

        Args:
            layer: Layer.
            x: X.
            y: Y.

        Returns:
            The result.
        """
        if self.modulation != ModConvType.NONE:
            return layer(x, y)
        return layer(x)

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        r"""Crops ``x`` to specified shape.

        Args:
            x: Input tensor with shape (\*, H, W).
            shape: Crop shape corresponding to H, W.

        Returns:
            Cropped tensor.
        """
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, channel_dim: int = 1) -> torch.Tensor:
        """Takes as input a torch.Tensor `x` and computes DIDN(x).

        Args:
            x: Input tensor.
            y: Auxiliary signal for modulation of shape (N, aux_in_features).
            channel_dim: Channel dimension. Default is ``1``.

        Returns:
            DIDN output tensor.
        """
        out = self.conv_in_act(self._conv(self.conv_in, x, y))
        out = self._conv(self.down, out, y)

        dub_outs = []
        for dub in self.dubs:
            out = dub(out, y)
            dub_outs.append(out)

        out = [self.recon_blocks[i](dub_outs[i], y) for i in range(self.num_dubs)]
        out = self._conv(self.recon_agg, torch.cat(out, dim=channel_dim), y)
        out = self.conv_act(self._conv(self.conv, out, y))

        if self.modulation != ModConvType.NONE:
            out = self.up2(out, y)
        else:
            out = self.up2(out)

        out = self._conv(self.conv_out, out, y)
        out = self.crop_to_shape(out, (x.shape[-2], x.shape[-1]))

        if self.skip_connection:
            out = x + out
        return out
