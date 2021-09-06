# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/
import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        dropout_probability : float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, input: torch.Tensor):
        """

        Parameters
        ----------
        input : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input)

    def __repr__(self):
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input: torch.Tensor):
        """

        Parameters
        ----------
        input : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input)

    def __repr__(self):
        return f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class UnetModel2d(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
    ):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels to the u-net.
        out_channels : int
            Number of output channels to the u-net.
        num_filters : int
            Number of output channels of the first convolutional layer.
        num_pool_layers : int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability : float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input: torch.Tensor):
        """

        Parameters
        ----------
        input : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class NormUnetModel2d(nn.Module):
    """
    Implementation of a Normalized U-Net model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
    ):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels to the u-net.
        out_channels : int
            Number of output channels to the u-net.
        num_filters : int
            Number of output channels of the first convolutional layer.
        num_pool_layers : int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability : float
            Dropout probability.
        """
        super().__init__()

        self.unet2d = UnetModel2d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
        )

    @staticmethod
    def norm(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = input.shape
        input = input.view(b, 2, c // 2 * h * w)

        mean = input.mean(dim=2).view(b, 2, 1, 1)
        std = input.std(dim=2).view(b, 2, 1, 1)

        input = input.view(b, c, h, w)
        output = (input - mean) / std

        return output, mean, std

    @staticmethod
    def unnorm(input: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return input * std + mean

    @staticmethod
    def pad(input: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = input.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        output = F.pad(input, w_pad + h_pad)

        return output, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(
        input: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:

        return input[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        input : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        output, mean, std = self.norm(input)
        output, pad_sizes = self.pad(output)
        output = self.unet2d(output)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std)

        return output
