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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.nn.conv.modulated_conv import (ModConv2d, ModConv2dBias,
                                           ModConvActivation, ModConvType)


class Subpixel(nn.Module):
    """Subpixel convolution layer for up-scaling of low resolution features at super-resolution as implemented in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising." 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: int = 0,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`Subpixel`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        upscale_factor: int
            Subpixel upscale factor.
        kernel_size: int or (int, int)
            Convolution kernel size.
        padding: int
            Padding size. Default: 0.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Auxiliary input features for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for modulation MLP.
        fc_groups : int
            Groups for modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()
        self.modulation = modulation
        self.conv = ModConv2d(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=kernel_size,
            padding=padding,
            modulation=modulation,
            bias=ModConv2dBias.PARAM,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes :class:`Subpixel` convolution on input torch.Tensor ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        y: torch.Tensor, optional
            Auxiliary signal for modulation.
        """
        if self.modulation != ModConvType.NONE:
            return self.pixelshuffle(self.conv(x, y))
        return self.pixelshuffle(self.conv(x))


class ReconBlock(nn.Module):
    """Reconstruction Block of :class:`DIDN` model as implemented in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising." 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        num_convs: int,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`ReconBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        num_convs: int
            Number of convolution blocks.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Auxiliary input features for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for modulation MLP.
        fc_groups : int
            Groups for modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()
        self.modulation = modulation
        self.num_convs = num_convs
        self.activations = nn.ModuleList([nn.PReLU() for _ in range(num_convs - 1)])

        self.convs = nn.ModuleList()
        for idx in range(num_convs):
            self.convs.append(
                ModConv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    padding=1,
                    modulation=modulation,
                    bias=ModConv2dBias.PARAM,
                    aux_in_features=aux_in_features,
                    fc_hidden_features=fc_hidden_features,
                    fc_groups=fc_groups,
                    fc_activation=fc_activation,
                    num_weights=num_weights,
                )
            )

    def forward(
        self, input_data: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes num_convs convolutions followed by PReLU activation on `input_data`.

        Parameters
        ----------
        input_data: torch.Tensor
            Input tensor.
        y: torch.Tensor, optional
            Auxiliary signal for modulation.
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
    """Down-up block (DUB) for :class:`DIDN` model as implemented in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising." 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`DUB`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Auxiliary input features for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for modulation MLP.
        fc_groups : int
            Groups for modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modulation = modulation

        conv_mod_kwargs = dict(
            modulation=modulation,
            bias=ModConv2dBias.PARAM,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )
        subpixel_kwargs = dict(
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        # Scale 1 - down path
        self.conv1_1_a = ModConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, **conv_mod_kwargs
        )
        self.conv1_1_a_act = nn.PReLU()
        self.conv1_1_b = ModConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, **conv_mod_kwargs
        )
        self.conv1_1_b_act = nn.PReLU()
        self.down1 = ModConv2d(
            in_channels,
            in_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            **conv_mod_kwargs,
        )

        # Scale 2 - down path
        self.conv2_1 = ModConv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            **conv_mod_kwargs,
        )
        self.conv2_1_act = nn.PReLU()
        self.down2 = ModConv2d(
            in_channels * 2,
            in_channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            **conv_mod_kwargs,
        )

        # Scale 3 - bottom
        self.conv3_1 = ModConv2d(
            in_channels * 4,
            in_channels * 4,
            kernel_size=3,
            padding=1,
            **conv_mod_kwargs,
        )
        self.conv3_1_act = nn.PReLU()
        self.up1 = Subpixel(
            in_channels * 4, in_channels * 2, 2, 1, 0, **subpixel_kwargs
        )

        # Scale 2 - up path
        self.conv_agg_1 = ModConv2d(
            in_channels * 4,
            in_channels * 2,
            kernel_size=1,
            padding=0,
            **conv_mod_kwargs,
        )
        self.conv2_2 = ModConv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            **conv_mod_kwargs,
        )
        self.conv2_2_act = nn.PReLU()
        self.up2 = Subpixel(in_channels * 2, in_channels, 2, 1, 0, **subpixel_kwargs)

        # Scale 1 - up path
        self.conv_agg_2 = ModConv2d(
            in_channels * 2, in_channels, kernel_size=1, padding=0, **conv_mod_kwargs
        )
        self.conv1_2_a = ModConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, **conv_mod_kwargs
        )
        self.conv1_2_a_act = nn.PReLU()
        self.conv1_2_b = ModConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, **conv_mod_kwargs
        )
        self.conv1_2_b_act = nn.PReLU()
        self.conv_out = ModConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, **conv_mod_kwargs
        )
        self.conv_out_act = nn.PReLU()

    def _conv(
        self, layer: ModConv2d, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> torch.Tensor:
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
    def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Crops ``x`` to specified shape.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape (\\*, H, W).
        shape: Tuple(int, int)
            Crop shape corresponding to H, W.

        Returns
        -------
        cropped_output: torch.Tensor
            Cropped tensor.
        """
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        y: torch.Tensor, optional
            Auxiliary signal for modulation.

        Returns
        -------
        out: torch.Tensor
            DUB output.
        """
        x1 = self.pad(x.clone())
        x1 = x1 + self.conv1_1_b_act(
            self._conv(
                self.conv1_1_b, self.conv1_1_a_act(self._conv(self.conv1_1_a, x1, y)), y
            )
        )
        x2 = self._conv(self.down1, x1, y)
        x2 = x2 + self.conv2_1_act(self._conv(self.conv2_1, x2, y))
        out = self._conv(self.down2, x2, y)
        out = out + self.conv3_1_act(self._conv(self.conv3_1, out, y))

        if self.modulation != ModConvType.NONE:
            out = self.up1(out, y)
        else:
            out = self.up1(out)

        out = torch.cat(
            [x2, self.crop_to_shape(out, (x2.shape[-2], x2.shape[-1]))], dim=1
        )
        out = self._conv(self.conv_agg_1, out, y)
        out = out + self.conv2_2_act(self._conv(self.conv2_2, out, y))

        if self.modulation != ModConvType.NONE:
            out = self.up2(out, y)
        else:
            out = self.up2(out)

        out = torch.cat(
            [x1, self.crop_to_shape(out, (x1.shape[-2], x1.shape[-1]))], dim=1
        )
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
    """Deep Iterative Down-up convolutional Neural network (DIDN) implementation as in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. "Deep Iterative Down-Up CNN for Image Denoising." 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
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
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`DIDN`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels. First convolution out_channels. Default: 128.
        num_dubs: int
            Number of DUB networks. Default: 6.
        num_convs_recon: int
            Number of ReconBlock convolutions. Default: 9.
        skip_connection: bool
            Use skip connection. Default: False.
        modulation : ModConvType
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Auxiliary input features for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features for modulation MLP.
        fc_groups : int
            Groups for modulation MLP. Default: 1.
        fc_activation : ModConvActivation
            Activation for modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()
        self.modulation = modulation

        mod_kwargs = dict(
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )
        conv_mod_kwargs = dict(
            bias=ModConv2dBias.PARAM,
            **mod_kwargs,
        )

        self.conv_in = ModConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            **conv_mod_kwargs,
        )
        self.conv_in_act = nn.PReLU()
        self.down = ModConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **conv_mod_kwargs,
        )
        self.dubs = nn.ModuleList(
            [
                DUB(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    **mod_kwargs,
                )
                for _ in range(num_dubs)
            ]
        )
        self.recon_blocks = nn.ModuleList(
            [
                ReconBlock(
                    in_channels=hidden_channels, num_convs=num_convs_recon, **mod_kwargs
                )
                for _ in range(num_dubs)
            ]
        )
        self.recon_agg = ModConv2d(
            in_channels=hidden_channels * num_dubs,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            **conv_mod_kwargs,
        )
        self.conv = ModConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            **conv_mod_kwargs,
        )
        self.conv_act = nn.PReLU()
        self.up2 = Subpixel(hidden_channels, hidden_channels, 2, 1, **mod_kwargs)
        self.conv_out = ModConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            **conv_mod_kwargs,
        )
        self.num_dubs = num_dubs
        self.skip_connection = (in_channels == out_channels) and skip_connection

    def _conv(
        self, layer: ModConv2d, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.modulation != ModConvType.NONE:
            return layer(x, y)
        return layer(x)

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Crops ``x`` to specified shape.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape (\\*, H, W).
        shape: Tuple(int, int)
            Crop shape corresponding to H, W.

        Returns
        -------
        cropped_output: torch.Tensor
            Cropped tensor.
        """
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, channel_dim: int = 1
    ) -> torch.Tensor:
        """Takes as input a torch.Tensor `x` and computes DIDN(x).

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        y: torch.Tensor, optional
            Auxiliary signal for modulation of shape (N, aux_in_features).
        channel_dim: int
            Channel dimension. Default: 1.

        Returns
        -------
        out: torch.Tensor
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
