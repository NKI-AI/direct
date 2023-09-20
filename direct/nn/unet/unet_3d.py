import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from direct.nn.conv.conv import CWN_Conv3d, CWN_ConvTranspose3d


class ConvBlock3D(nn.Module):
    """3D U-Net convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_probability),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_probability),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.layers(input_data)


class TransposeConvBlock3D(nn.Module):
    """3D U-Net Transpose Convolutional Block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.layers(input_data)


class CWNConvBlock3D(nn.Module):
    """U-Net convolutional block for 3D data."""

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            CWN_Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_probability),
            CWN_Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_probability),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.layers(input_data)

    def __repr__(self):
        return (
            f"CWNConvBlock3D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class CWNTransposeConvBlock3D(nn.Module):
    """U-Net Transpose Convolutional Block for 3D data."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            CWN_ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.layers(input_data)


class UnetModel3d(nn.Module):
    """PyTorch implementation of a 3D U-Net model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        cwn_conv: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        if cwn_conv:
            conv_block = CWNConvBlock3D
            transpose_conv_block = CWNTransposeConvBlock3D
        else:
            conv_block = ConvBlock3D
            transpose_conv_block = TransposeConvBlock3D

        self.down_sample_layers = nn.ModuleList([conv_block(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [conv_block(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = conv_block(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [transpose_conv_block(ch * 2, ch)]
            self.up_conv += [conv_block(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [transpose_conv_block(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                conv_block(ch * 2, ch, dropout_probability),
                nn.Conv3d(ch, out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        stack = []
        output, inp_pad = pad_to_pow_of_2(input_data, self.num_pool_layers)

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
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
    """Implementation of a Normalized U-Net model for 3D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        norm_groups: int = 2,
        cwn_conv: bool = False,
    ):
        """Inits :class:`NormUnetModel3D`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the U-Net.
        out_channels: int
            Number of output channels to the U-Net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        norm_groups: int,
            Number of normalization groups.
        cwn_conv : bool
            Apply centered weight normalization to convolutions. Default: False.
        """
        super().__init__()

        self.unet3d = UnetModel3d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
            cwn_conv=cwn_conv,
        )

        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        # Group norm
        b, c, z, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, z, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(
        input_data: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        b, c, z, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, z, h, w)

    @staticmethod
    def pad(
        input_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int, List[int], List[int]]]:
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
        h_pad: List[int],
        w_pad: List[int],
        z_pad: List[int],
        h_mult: int,
        w_mult: int,
        z_mult: int,
    ) -> torch.Tensor:
        return input_data[
            ..., z_pad[0] : z_mult - z_pad[1], h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]
        ]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`NormUnetModel3D`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)
        output = self.unet3d(output)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)

        return output


def pad_to_pow_of_2(inp, k):
    diffs = [_ - 2**k for _ in inp.shape[2:]]
    padding = [0, 0, 0, 0, 0, 0]
    for i, diff in enumerate(diffs[::-1]):
        if diff < 1:
            padding[2 * i] = abs(diff) // 2
            padding[2 * i + 1] = abs(diff) - padding[2 * i]

    if sum(padding) > 0:
        inp = F.pad(inp, padding)

    return inp, padding
