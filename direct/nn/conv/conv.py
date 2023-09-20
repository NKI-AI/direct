# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Conv2d(nn.Module):
    """Implementation of a simple cascade of 2D convolutions.

    If `batchnorm` is set to True, batch normalization layer is applied after each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_convs: int = 3,
        activation: nn.Module = nn.PReLU(),
        batchnorm: bool = False,
    ):
        """Inits :class:`Conv2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels.
        n_convs: int
            Number of convolutional layers.
        activation: nn.Module
            Activation function.
        batchnorm: bool
            If True a batch normalization layer is applied after every convolution.
        """
        super().__init__()

        conv: List[nn.Module] = []
        for idx in range(n_convs):
            conv.append(
                nn.Conv2d(
                    in_channels if idx == 0 else hidden_channels,
                    hidden_channels if idx != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if batchnorm:
                conv.append(nn.BatchNorm2d(hidden_channels if idx != n_convs - 1 else out_channels, eps=1e-4))
            if idx != n_convs - 1:
                conv.append(activation)
        self.conv = nn.Sequential(*conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`Conv2d`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        out: torch.Tensor
            Convoluted output.
        """
        out = self.conv(x)
        return out


# Centered Weight Normalization module
class CWNorm(nn.Module):
    """Centered Weight Normalization module.

    This module performs centered weight normalization on the weight tensors of Conv2d layers.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass of the centered weight normalization module.

        Parameters
        ----------
        weight : torch.Tensor
            The weight tensor of the Conv2d layer.

        Returns
        -------
        torch.Tensor
            he normalized weight tensor.
        """
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_CWN = weight_ / norm
        return weight_CWN.view(weight.size())


# Custom Conv2d layer with centered weight normalization
class CWN_Conv2d(nn.Conv2d):
    """Convolutional layer with Centered Weight Normalization.

    This layer extends the functionality of the standard Conv2d layer in PyTorch by applying
    centered weight normalization to its weight tensors.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        NScale=1.414,
        adjustScale=False,
        *args,
        **kwargs,
    ):
        """Inits :class:`CWN_Conv2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple
            Size of the convolutional kernel.
        stride : int or tuple, optional
            Stride for the convolution operation. Default: 1.
        padding : int or tuple, optional
            Padding for the convolution operation. Default: 0.
        dilation : int or tuple, optional
            Dilation rate for the convolution operation. Default: 1.
        groups : int, optional
            Number of groups for grouped convolution. Default: 1.
        bias : bool, optional
            If True, the layer has a bias term. Default: True.
        NScale : float, optional
            The scale factor for the centered weight normalization. Default: 1.414.
        adjustScale : bool, optional
            If True, the scale factor is adjusted as a learnable parameter. Default: False.
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, *args, **kwargs
        )
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            self.register_buffer("WNScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CWN_Conv2d layer.

        Parameters
        ----------
        input_f : torch.Tensor
            The input tensor to the convolutional layer.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the convolution operation with centered weight normalization.
        """
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class CWN_ConvTranspose2d(nn.ConvTranspose2d):
    """Transposed Convolutional layer with Centered Weight Normalization.

    This layer extends the functionality of the standard ConvTranspose2d layer in PyTorch by applying
    centered weight normalization to its weight tensors.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        NScale=1.414,
        adjustScale=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            *args,
            **kwargs,
        )
        """Inits :class:`CWN_ConvTranspose2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple
            Size of the convolutional kernel.
        stride : int or tuple, optional
            Stride for the convolution operation. Default: 1.
        padding : int or tuple, optional
            Padding for the convolution operation. Default: 0.
        dilation : int or tuple, optional
            Dilation rate for the convolution operation. Default: 1.
        groups : int, optional
            Number of groups for grouped convolution. Default: 1.
        bias : bool, optional
            If True, the layer has a bias term. Default: True.
        NScale : float, optional
            The scale factor for the centered weight normalization. Default: 1.414.
        adjustScale : bool, optional
            If True, the scale factor is adjusted as a learnable parameter. Default: False.
        """
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(in_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            self.register_buffer("WNScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv_transpose2d(
            input_f, weight_q, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        return out


class CWN_Conv3d(nn.Conv3d):
    """Convolutional layer with Centered Weight Normalization for 3D data."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        NScale=1.414,
        adjustScale=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, *args, **kwargs
        )
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            self.register_buffer("WNScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv3d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class CWN_ConvTranspose3d(nn.ConvTranspose3d):
    """Transposed Convolutional layer with Centered Weight Normalization for 3D data."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        NScale=1.414,
        adjustScale=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            *args,
            **kwargs,
        )
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(in_channels, 1, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            self.register_buffer("WNScale", self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv_transpose3d(
            input_f, weight_q, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        return out
