# Copyright (c) DIRECT Contributors

"""Code for three-dimensional U-Net adapted from the 3d variant."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from direct.data import transforms as T
from direct.nn.types import InitType


class ConvBlock3D(nn.Module):
    """3D U-Net convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float) -> None:
        """Inits :class:`ConvBlock3D`.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolutional layers.
        dropout_probability : float
            Dropout probability applied after convolutional layers.
        """
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
        """Performs the forward pass of :class:`ConvBlock3D`..

        Parameters
        ----------
        input_data : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)


class TransposeConvBlock3D(nn.Module):
    """3D U-Net Transpose Convolutional Block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Inits :class:`TransposeConvBlock3D`.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolutional layers.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`TransposeConvBlock3D`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)


class UnetModel3d(nn.Module):
    """PyTorch implementation of a 3D U-Net model.

    This class defines a 3D U-Net architecture consisting of down-sampling and up-sampling layers with 3D convolutional
    blocks. This is an extension to 3D volumes of :class:`direct.nn.unet.unet_3d.UnetModel3d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
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
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList([ConvBlock3D(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock3D(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock3D(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock3D(ch * 2, ch)]
            self.up_conv += [ConvBlock3D(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock3D(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock3D(ch * 2, ch, dropout_probability),
                nn.Conv3d(ch, out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel3d`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor of shape (N, in_channels, slice/time, height, width).

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, slice/time, height, width).
        """
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
    """Implementation of a Normalized U-Net model for 3D data.

    This is an extension to 3D volumes of :class:`direct.nn.unet.unet_3d.NormUnetModel3d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        norm_groups: int = 2,
    ) -> None:
        """Inits :class:`NormUnetModel3D`.

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
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()

        self.unet3d = UnetModel3d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
        )

        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies group normalization for 3D data.

        Parameters
        ----------
        input_data : torch.Tensor
            The input tensor to normalize.
        groups : int
            The number of groups to divide the tensor into for normalization.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the normalized tensor, the mean, and the standard deviation used for normalization.
        """
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
        """Reverts the normalization applied to the 3D tensor.

        Parameters
        ----------
        input_data : torch.Tensor
            The normalized tensor to revert normalization on.
        mean : torch.Tensor
            The mean used during normalization.
        std : torch.Tensor
            The standard deviation used during normalization.
        groups : int
            The number of groups the tensor was divided into during normalization.

        Returns
        -------
        torch.Tensor
            The tensor after reverting the normalization.
        """
        b, c, z, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, z, h, w)

    @staticmethod
    def pad(
        input_data: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[list[int], list[int], int, int, list[int], list[int]]]:
        """Applies padding to the input 3D tensor to ensure its dimensions are multiples of 16.

        Parameters
        ----------
        input_data : torch.Tensor
            The input tensor to pad.

        Returns
        -------
        tuple[torch.Tensor, tuple[list[int], list[int], int, int, list[int], list[int]]]
            A tuple containing the padded tensor and a tuple with the padding applied to each dimension
            (height, width, depth) and the target dimensions after padding.
        """
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
        """Removes padding from the 3D input tensor, reverting it to its original dimensions before padding was applied.

        This method is typically used after the model has processed the padded input.

        Parameters
        ----------
        input_data : torch.Tensor
            The tensor from which padding will be removed.
        h_pad : list[int]
            Padding applied to the height, specified as [top, bottom].
        w_pad : list[int]
            Padding applied to the width, specified as [left, right].
        z_pad : list[int]
            Padding applied to the depth, specified as [front, back].
        h_mult : int
            The height as computed in the `pad` method.
        w_mult : int
            The width as computed in the `pad` method.
        z_mult : int
            The depth as computed in the `pad` method.

        Returns
        -------
        torch.Tensor
            The tensor with padding removed, restored to its original dimensions.
        """
        return input_data[
            ..., z_pad[0] : z_mult - z_pad[1], h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]
        ]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`NormUnetModel3D`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor of shape (N, in_channels, slice/time, height, width).

        Returns
        -------
        torch.Tensor
            Output of shape (N, out_channels, slice/time, height, width).
        """
        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)
        output = self.unet3d(output)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)

        return output


class Unet3d(nn.Module):
    """PyTorch implementation of a U-Net model for MRI Reconstruction in 3D."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        skip_connection: bool = False,
        normalized: bool = False,
        image_initialization: InitType = InitType.ZERO_FILLED,
        **kwargs,
    ):
        """Inits :class:`Unet3d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_filters: int
            Number of first layer filters.
        num_pool_layers: int
            Number of pooling layers.
        dropout_probability: float
            Dropout probability.
        skip_connection: bool
            If True, skip connection is used for the output. Default: False.
        normalized: bool
            If True, Normalized Unet is used. Default: False.
        image_initialization: InitType
            Type of image initialization. Default: InitType.ZERO_FILLED.
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
        self.unet: nn.Module
        if normalized:
            self.unet = NormUnetModel3d(
                in_channels=2,
                out_channels=2,
                num_filters=num_filters,
                num_pool_layers=num_pool_layers,
                dropout_probability=dropout_probability,
            )
        else:
            self.unet = UnetModel3d(
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
        self._spatial_dims = (3, 4)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:

        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, slice/time, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, slice/time, height, width, complex=2).

        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
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
        """Computes forward pass of Unet3d.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, slice/time, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, slice/time, height, width, complex=2). Default: None.

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, slice/time, height, width, complex=2).
        """
        if self.image_initialization == InitType.SENSE:
            if sensitivity_map is None:
                raise ValueError("Expected sensitivity_map not to be None with InitType.SENSE image_initialization.")
            input_image = self.compute_sense_init(
                kspace=masked_kspace,
                sensitivity_map=sensitivity_map,
            )
        elif self.image_initialization == InitType.ZERO_FILLED:
            input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
        else:
            raise ValueError(
                f"Unknown image_initialization. Expected InitType.ZERO_FILLED or InitType.SENSE. "
                f"Got {self.image_initialization}."
            )

        output = self.unet(input_image.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        if self.skip_connection:
            output += input_image
        return output


def pad_to_pow_of_2(inp: torch.Tensor, k: int) -> tuple[torch.Tensor, list[int]]:
    """Pads the input tensor along the spatial dimensions (depth, height, width) to the nearest power of 2.

    This is necessary for certain operations in the 3D U-Net architecture to maintain dimensionality.

    Parameters
    ----------
    inp : torch.Tensor
        The input tensor to be padded.
    k : int
        The exponent to which the base of 2 is raised to determine the padding. Used to calculate
        the target dimension size as a power of 2.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        A tuple containing the padded tensor and a list of padding applied to each spatial dimension
        in the format [depth_front, depth_back, height_top, height_bottom, width_left, width_right].

    Examples
    --------
    >>> inp = torch.rand(1, 1, 15, 15, 15)  # A random tensor with shape [1, 1, 15, 15, 15]
    >>> padded_inp, padding = pad_to_pow_of_2(inp, 4)
    >>> print(padded_inp.shape, padding)
    torch.Size([...]), [1, 1, 1, 1, 1, 1]
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
