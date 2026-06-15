# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from direct.data import transforms as T
from direct.nn.adain.adain import AdaIN2d, NormType
from direct.nn.conv.modulated_conv import ModConv2d, ModConv2dBias, ModConvActivation, ModConvTranspose2d, ModConvType
from direct.nn.types import InitType
from direct.types import FFTOperator


class ConvModule(nn.Module):
    """Single convolution + norm + activation + dropout module supporting modulated convolutions."""

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
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: Optional[tuple[int] | int] = None,
    ):
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
        self.norm_type = norm_type
        if norm_type == NormType.ADAIN:
            if adain_hidden_features is None:
                raise ValueError("AdaIN hidden features must be provided if norm_type is NormType.ADAIN.")
            self.instance_norm = AdaIN2d(
                num_channels=out_channels,
                aux_in_features=aux_in_features,
                hidden_features=adain_hidden_features,
            )
        else:
            self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_probability)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
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


class ConvBlock(nn.Module):
    """U-Net convolutional block.

    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    Supports modulated convolutions and AdaIN normalization.
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
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: Optional[tuple[int] | int] = None,
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
            Modulation type. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of auxiliary input features for modulation/AdaIN.
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
            Hidden features for AdaIN. Required if norm_type is NormType.ADAIN.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.modulation = modulation
        self.norm_type = norm_type

        self.layer_1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED,
            dropout_probability=dropout_probability,
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            norm_type=norm_type,
            adain_hidden_features=adain_hidden_features,
        )
        self.layer_2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED,
            dropout_probability=dropout_probability,
            modulation=modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            norm_type=norm_type,
            adain_hidden_features=adain_hidden_features,
        )

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock`.

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

    def __repr__(self):
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability}, modulation={self.modulation})"
        )


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block with optional modulation.

    It consists of one convolution transpose layer followed by instance normalization and LeakyReLU activation.
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
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: Optional[tuple[int] | int] = None,
    ):
        """Inits :class:`TransposeConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
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

        self.conv = ModConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            modulation=modulation,
            bias=ModConv2dBias.NONE if modulation == ModConvType.NONE else ModConv2dBias.LEARNED,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )
        if norm_type == NormType.ADAIN:
            if adain_hidden_features is None:
                raise ValueError("AdaIN hidden features must be provided if norm_type is NormType.ADAIN.")
            self.instance_norm = AdaIN2d(
                num_channels=out_channels,
                aux_in_features=aux_in_features,
                hidden_features=adain_hidden_features,
            )
        else:
            self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`TransposeConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor
        aux_data: torch.Tensor, optional

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

    def __repr__(self):
        return f"TransposeConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class UnetModel2d(nn.Module):
    """PyTorch implementation of a U-Net model based on [1]_.

    Supports optional modulated convolutions and AdaIN normalization conditioned
    on an auxiliary input signal.

    References
    ----------
    .. [1] Ronneberger, Olaf, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
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
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: Optional[tuple[int] | int] = None,
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
            [ConvBlock(in_channels, num_filters, dropout_probability, modulation, aux_in_features,
                       fc_hidden_features, fc_groups, fc_activation, num_weights, norm_type, adain_hidden_features)]
        )
        ch = num_filters

        if modulation != ModConvType.NONE and modulation_at_input:
            modulation = ModConvType.NONE

        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [
                ConvBlock(ch, ch * 2, dropout_probability, modulation, aux_in_features,
                          fc_hidden_features, fc_groups, fc_activation, num_weights, norm_type, adain_hidden_features)
            ]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability, modulation, aux_in_features,
                              fc_hidden_features, fc_groups, fc_activation, num_weights, norm_type,
                              adain_hidden_features)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [
                TransposeConvBlock(ch * 2, ch, modulation, aux_in_features, fc_hidden_features,
                                   fc_groups, fc_activation, num_weights, norm_type, adain_hidden_features)
            ]
            self.up_conv += [
                ConvBlock(ch * 2, ch, dropout_probability, modulation, aux_in_features,
                          fc_hidden_features, fc_groups, fc_activation, num_weights, norm_type, adain_hidden_features)
            ]
            ch //= 2

        self.up_transpose_conv += [
            TransposeConvBlock(ch * 2, ch, modulation, aux_in_features, fc_hidden_features,
                               fc_groups, fc_activation, num_weights, norm_type, adain_hidden_features)
        ]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout_probability, modulation, aux_in_features,
                          fc_hidden_features, fc_groups, fc_activation, num_weights, norm_type, adain_hidden_features),
                nn.Conv2d(ch, out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor, aux_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel2d`.

        Parameters
        ----------
        input_data: torch.Tensor
        aux_data: torch.Tensor, optional
            Auxiliary data for modulation/AdaIN.

        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input_data

        for layer in self.down_sample_layers:
            if self.modulation != ModConvType.NONE or self.norm_type == NormType.ADAIN:
                output = layer(output, aux_data)
            else:
                output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

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

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
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

        return output


class NormUnetModel2d(nn.Module):
    """Implementation of a Normalized U-Net model with optional modulation support."""

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
        norm_type: NormType = NormType.INSTANCE,
        adain_hidden_features: Optional[tuple[int] | int] = None,
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
            norm_type=norm_type,
            adain_hidden_features=adain_hidden_features,
            num_weights=num_weights,
        )
        self.modulation = modulation
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
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
        aux_data : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
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
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        skip_connection: bool = False,
        normalized: bool = False,
        image_initialization: InitType = InitType.ZERO_FILLED,
        **kwargs,
    ):
        """Inits :class:`Unet2d`.

        Parameters
        ----------
        forward_operator: FFTOperator
            Forward Operator.
        backward_operator: FFTOperator
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

        output = self.unet(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.skip_connection:
            output += input_image
        return output
