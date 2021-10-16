# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/
import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from direct.data import transforms as T


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
        for _ in range(num_pool_layers - 1):
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
        norm_groups: int = 2,
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
        norm_groups : int,
            Number of normalization groups.
        """
        super().__init__()

        self.unet2d = UnetModel2d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
        )

        self.norm_groups = norm_groups

    @staticmethod
    def norm(input: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = input.shape
        input = input.reshape(b, groups, -1)

        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)

        output = (input - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = input.shape
        input = input.reshape(b, groups, -1)
        return (input * std + mean).reshape(b, c, h, w)

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

        output, mean, std = self.norm(input, self.norm_groups)
        output, pad_sizes = self.pad(output)
        output = self.unet2d(output)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)

        return output


class Unet2d(nn.Module):
    """
    PyTorch implementation of a U-Net model for MRI Reconstruction.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        skip_connection: bool = False,
        normalized: bool = False,
        image_initialization: str = "zero_filled",
        **kwargs,
    ):
        """

        Parameters
        ----------
        forward_operator : Callable
            Forward Operator.
        backward_operator : Callable
            Backward Operator.
        num_filters : int
            Number of first layer filters.
        num_pool_layers : int
            Number of pooling layers.
        dropout_probability : float
            Dropout probability.
        skip_connection : bool
            If True, skip connection is used for the output. Default: False.
        normalized : bool
            If True, Normalized Unet is used. Default: False.
        image_initialization : str
            Type of image initialization. Default: "zero-filled".
        kwargs: dict
        """
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "sensitivity_map_model",
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        if normalized:
            self.unet = NormUnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )
        else:
            self.unet = UnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.skip_connection = skip_connection

        self.image_initialization = image_initialization

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace, sensitivity_map):

        input_image = T.complex_multiplication(
            T.conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)

        return input_image

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.

        Returns
        -------
        torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """

        if self.image_initialization == "sense":
            if sensitivity_map is None:
                raise ValueError("Expected sensitivity_map not to be None with 'sense' image_initialization.")
            input_image = self.compute_sense_init(
                kspace=masked_kspace,
                sensitivity_map=sensitivity_map,
            )
        elif self.image_initialization == "zero_filled":
            input_image = self.backward_operator(masked_kspace).sum(self._coil_dim)
        else:
            raise ValueError(
                f"Unknown image_initialization. Expected `sense` or `zero_filled`. "
                f"Got {self.image_initialization}."
            )

        output = self.unet(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.skip_connection:
            output += input_image

        return output
