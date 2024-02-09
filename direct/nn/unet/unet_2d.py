# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from direct.data import transforms as T
from direct.nn.conv.modulated_conv import ModConv2d, ModConv2dBias, ModConvActivation, ModConvTranspose2d, ModConvType


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dropout_probability: float,
        modulation: ModConvType = ModConvType.NONE,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`ConvModule`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
             Size of the convolutional kernel.
        padding : int
             Padding added to all sides of the input.
        dropout_probability: float
            Dropout probability.
        modulation : ModConvType, optional
            If not ModConvType.NONE, it will apply modulation using an MLP on the auxiliary variable `y`.
            By default ModConvType.NONE. Can be ModConvType.NONE, ModConvType.FULL, ModConvType.PARTIAL_IN, or
            ModConvType.PARTIAL_OUT.
        bias : ModConv2dBias, optional
            Type of bias, by default ModConv2dBias.PARAM.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`. Ignored if `modulation` is ModConvType.NONE.
            Default: None.
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulation MLP unit. Ignored if `modulation` is ModConvType.NONE.
            Default: None.
        fc_groups : int, optional
            Number of MLP groups for the modulation MLP unit. Ignored if `modulation` is ModConvType.NONE
            or ModConvType.SUM. Default: 1.
        fc_activation : ModConvActivation
            Activation function to be applied in the modulation MLP unit. Ignored if `modulation` is ModConvType.NONE.
            Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        """
        super().__init__()

        self.modulation = modulation

        self.conv = ModConv2d(
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
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_probability)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvModule`.

        Parameters
        ----------
        x : torch.Tensor
        y : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self.modulation != ModConvType.NONE:
            x = self.conv(x, y)
        else:
            x = self.conv(x)
        x = self.instance_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


class ConvBlock(nn.Module):
    """U-Net convolutional block.

    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_probability: float,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
    ):
        """Inits :class:`ConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        modulation : ModConvType
            If not ModConvType.None, modulated convolutions will be used. Default: ModConvType.None.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`. Ignored if `modulation` is ModConvType.NONE.
            Default: None.
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulation MLP units. Ignored if `modulation` is ModConvType.NONE.
            Default: None.
        fc_groups : int, optional
            Number of MLP groups for the modulation MLP unit. Ignored if `modulation` is ModConvType.NONE
            or ModConvType.SUM. Default: 1.
        fc_activation : ModConvActivation
            Activation function to be applied in the MLP units. Ignored if `modulation` is ModConvType.NONE.
            Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_groups = fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        self.layer_1, self.layer_2 = [
            ConvModule(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=ModConv2dBias.NONE if self.modulation == ModConvType.NONE else ModConv2dBias.LEARNED,
                dropout_probability=dropout_probability,
                modulation=modulation,
                aux_in_features=aux_in_features,
                fc_hidden_features=fc_hidden_features,
                fc_groups=fc_groups,
                fc_activation=fc_activation,
                num_weights=num_weights,
            )
            for i in range(2)
        ]

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock`.

        Parameters
        ----------
        input_data : torch.Tensor
        aux_data : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self.modulation != ModConvType.NONE:
            out = self.layer_2(self.layer_1(input_data, aux_data), aux_data)
        else:
            out = self.layer_2(self.layer_1(input_data))
        return out

    def __repr__(self):
        """Representation of :class:`ConvBlock`."""
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability}, modulation={self.modulation}, "
            f"aux_in_features={self.aux_in_features}, fc_hidden_features={self.fc_hidden_features},"
            f"fc_groups={self.fc_groups}, fc_activation={self.fc_activation}, num_weights={self.num_weights})"
        )


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block.

    It consists of one convolution transpose layers followed by instance normalization and LeakyReLU activation.
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
        """Inits :class:`TransposeConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        modulation : ModConvType
            If not ModConvType.None, modulated convolutions will be used. Default: ModConvType.None.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`. Ignored if `modulation` is ModConvType.NONE.
            Default: None.
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulation MLP unit. Ignored if `modulation` is ModConvType.NONE.
            Default: None.
        fc_groups : int, optional
            Number of MLP groups for the modulation MLP unit. Ignored if `modulation` is ModConvType.NONE.
            Default: 1.
        fc_activation : ModConvActivation
            Activation function to be applied in the MLP units. Ignored if `modulation` is ModConvType.NONE.
            Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_groups = fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        self.conv = ModConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=ModConv2dBias.NONE if self.modulation == ModConvType.NONE else ModConv2dBias.LEARNED,
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the forward pass of :class:`TransposeConvBlock`.

        Parameters
        ----------
        input_data : torch.Tensor
        aux_data : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self.modulation != ModConvType.NONE:
            out = self.conv(input_data, aux_data)
        else:
            out = self.conv(input_data)
        return self.leaky_relu(self.instance_norm(out))

    def __repr__(self):
        """Representation of "class:`TransposeConvBlock`."""
        return (
            f"TransposeConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"modulation={self.modulation}, aux_in_features={self.aux_in_features}, "
            f"fc_hidden_features={self.fc_hidden_features}, fc_groups={self.fc_groups}, "
            f"fc_activation={self.fc_activation}, num_weights={self.num_weights})"
        )


class UnetModel2d(nn.Module):
    """PyTorch implementation of a U-Net model based on [1]_.

    References
    ----------

    .. [1] Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Medical Image
    Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab et al., Springer International
    Publishing, 2015, pp. 234–41. Springer Link, https://doi.org/10.1007/978-3-319-24574-4_28.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        modulation_at_input: bool = False,
    ):
        """Inits :class:`UnetModel2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        modulation : ModConvType
            If not ModConvType.None, modulated convolutions will be used. Default: ModConvType.None.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`. Ignored if `modulation` is ModConvType.None.
            Default: None.
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulated convolutions. Ignored if `modulation` is ModConvType.None.
            Default: None.
        fc_groups : int, optional
            Number of groups in the modulated convolutions. Ignored if `modulation` is ModConvType.NONE
            or ModConvType.SUM. Default: 1.
        fc_activation : ModConvActivation
            Activation function to be applied in the MLP units for modulated convolutions.
            Ignored if `modulation` is ModConvType.None. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        modulation_at_input : bool, optional
            If True, apply modulation only at the initial convolutional layer. Default: False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability
        self.modulation = modulation

        self.down_sample_layers = nn.ModuleList(
            [
                ConvBlock(
                    in_channels,
                    num_filters,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                )
            ]
        )

        if modulation != ModConvType.NONE and modulation_at_input:
            modulation = ModConvType.NONE

        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [
                ConvBlock(
                    ch,
                    ch * 2,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                )
            ]
            ch *= 2
        self.conv = ConvBlock(
            ch,
            ch * 2,
            dropout_probability,
            modulation,
            aux_in_features,
            fc_hidden_features,
            fc_groups,
            fc_activation,
            num_weights,
        )

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [
                TransposeConvBlock(
                    ch * 2, ch, modulation, aux_in_features, fc_hidden_features, fc_groups, fc_activation, num_weights
                )
            ]
            self.up_conv += [
                ConvBlock(
                    ch * 2,
                    ch,
                    dropout_probability,
                    modulation,
                    aux_in_features,
                    fc_hidden_features,
                    fc_groups,
                    fc_activation,
                    num_weights,
                )
            ]
            ch //= 2

        self.up_transpose_conv += [
            TransposeConvBlock(
                ch * 2, ch, modulation, aux_in_features, fc_hidden_features, fc_groups, fc_activation, num_weights
            )
        ]
        self.up_conv += [
            ConvBlock(
                ch * 2,
                ch,
                dropout_probability,
                modulation,
                aux_in_features,
                fc_hidden_features,
                fc_groups,
                fc_activation,
                num_weights,
            )
        ]

        self.conv_out = ModConv2d(
            ch,
            self.out_channels,
            kernel_size=1,
            stride=1,
            modulation=modulation,
            bias=ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel2d`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data tensor of shape (N, `in_channels`, H, W).
        aux_data : torch.Tensor, optional
            Auxiliary data tensor of shape (N, `aux_in_features`) to be used if `modulation` is set to True.
            Default: None

        Returns
        -------
        torch.Tensor
            Output data tensor of shape (N, `out_channels`, H, W).
        """
        stack = []
        output = input_data

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            if self.modulation != ModConvType.NONE:
                output = layer(output, aux_data)
            else:
                output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        if self.modulation != ModConvType.NONE:
            output = self.conv(output, aux_data)
        else:
            output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            if self.modulation != ModConvType.NONE:
                output = transpose_conv(output, aux_data)
            else:
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
            if self.modulation != ModConvType.NONE:
                output = conv(output, aux_data)
            else:
                output = conv(output)

        if self.modulation != ModConvType.NONE:
            output = self.conv_out(output, aux_data)
        else:
            output = self.conv_out(output)

        return output


class NormUnetModel2d(nn.Module):
    """Implementation of a Normalized U-Net model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        norm_groups: int = 2,
        modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        modulation_at_input: bool = False,
    ):
        """Inits :class:`NormUnetModel2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        norm_groups: int,
            Number of normalization groups.
        modulation : ModConvType
            If not ModConvType.None, modulated convolutions will be used. Default: ModConvType.None.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`. Ignored if `modulation` is ModConvType.None.
            Default: None.
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulated convolutions. Ignored if `modulation` is ModConvType.None.
            Default: None.
        fc_groups : int, optional
            Number of groups in the modulated convolutions. Ignored if `modulation` is ModConvType.NONE
            or ModConvType.SUM. Default: 1.
        fc_activation : ModConvActivation
            Activation function to be applied in the MLP units for modulated convolutions.
            Ignored if `modulation` is ModConvType.None. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        modulation_at_input : bool, optional
            If True, apply modulation only at the initial convolutional layer. Default: False.
        """
        super().__init__()

        self.unet2d = UnetModel2d(
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
            modulation_at_input=modulation_at_input,
            num_weights=num_weights,
        )
        self.modulation = modulation
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        # group norm
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    @staticmethod
    def pad(input_data: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = input_data.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        output = F.pad(input_data, w_pad + h_pad)
        return output, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(
        input_data: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return input_data[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`NormUnetModel2d`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data tensor of shape (N, `in_channels`, H, W).
        aux_data : torch.Tensor, optional
            Auxiliary data tensor of shape (N, `aux_in_features`) to be used if `modulation` is set to True.
            Default: None

        Returns
        -------
        torch.Tensor
            Output data tensor of shape (N, `out_channels`, H, W).
        """

        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)

        output = self.unet2d(output, aux_data)

        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)

        return output


class Unet2d(nn.Module):
    """PyTorch implementation of a U-Net model for MRI Reconstruction."""

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
        """Inits :class:`Unet2d`.

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
        image_initialization: str
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
        self.unet: nn.Module
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

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:

        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

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
        """Computes forward pass of Unet2d.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.

        Returns
        -------
        output: torch.Tensor
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
            input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
        else:
            raise ValueError(
                f"Unknown image_initialization. Expected `sense` or `zero_filled`. "
                f"Got {self.image_initialization}."
            )

        output = self.unet(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.skip_connection:
            output += input_image
        return output
