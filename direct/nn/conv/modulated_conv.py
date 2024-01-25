# Copyright (c) DIRECT Contributors

"""direct.nn.conv.modulated_conv module"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.types import DirectEnum, IntOrTuple

__all__ = ["ModConv2d", "ModConv2dBias", "ModConvActivation", "ModConvTranspose2d"]


class ModConv2dBias(DirectEnum):
    LEARNED = "learned"
    PARAM = "param"
    NONE = "none"


class ModConvActivation(DirectEnum):
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    NONE = "none"


class ModConv2d(nn.Module):
    """Modulated Conv2d module.

    If `modulation`=False and `bias`=ModConv2dBias.PARAM this is identical to nn.Conv2d:

    .. math ::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k=0}^{C_{\text{in}}-1} \text{weight}(C_{\text{out}_j}, k) * \text{input}(N_i, k)



    If `modulation`=True, this will compute:

    .. math ::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k=0}^{C_{\text{in}}-1} \text{MLP}(y(N_i))(C_{\text{out}_j}, k) \text{weight}(C_{\text{out}_j}, k)
        * \text{input}(N_i, k).

    where :math`*` is a 2D cross-correlation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrTuple,
        modulation: bool = False,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[int] = None,
        fc_bias: Optional[bool] = True,
        fc_groups: Optional[int] = None,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
    ):
        """Inits :class:`ModConv2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple of int
            Size of the convolutional kernel.
        modulation : bool, optional
            If True, apply modulation using an MLP on the auxiliary variable `y`, by default False.
        stride : int or tuple of int, optional
            Stride of the convolution, by default 1.
        padding : int or tuple of int, optional
            Padding added to all sides of the input, by default 0.
        dilation : int or tuple of int, optional
            Spacing between kernel elements, by default 1.
        bias : ModConv2dBias, optional
            Type of bias, by default ModConv2dBias.PARAM.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`, by default None.
        fc_hidden_features : int, optional
            Number of hidden features in the modulation MLP, by default None.
        fc_bias : bool, optional
            If True, enable bias in the modulation MLP, by default True.
        fc_groups : int, optional
            If not None and greater than 1, then the MLP output shape will be divided by `fc_grpups` ** 2, and
            expanded to the convolution weight via 'nearest' interpolation. Can be used to reduce memory. Default: None.
        fc_activation : ModConvActivation
            Activation function to be applied after MLP layer. Default: ModConvActivation.SIGMOID.
        """
        super().__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        k = math.sqrt(1 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = nn.Parameter(
            torch.FloatTensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias

        if modulation:
            if aux_in_features is None:
                raise ValueError(f"Value for `aux_in_features` can't be None with `modulation`=True.")
            if fc_hidden_features is None:
                raise ValueError(f"Value for `fc_hidden_features` can't be None with `modulation`=True.")

            if fc_groups is not None and fc_groups > 1:
                fc_out_features = (out_channels // fc_groups) * (in_channels // fc_groups)
            else:
                fc_out_features = out_channels * in_channels

            self.fc_groups = fc_groups

            self.fc = nn.Sequential(
                nn.Linear(aux_in_features, fc_hidden_features, bias=fc_bias),
                nn.PReLU(),
                nn.Linear(fc_hidden_features, fc_out_features, bias=fc_bias),
                *(
                    (nn.Sigmoid(),)
                    if fc_activation == ModConvActivation.SIGMOID
                    else (nn.Softplus(),)
                    if fc_activation == ModConvActivation.SOFTPLUS
                    else ()
                ),
            )

            self.fc_activation = fc_activation

        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if not modulation:
                raise ValueError(
                    f"Bias can only be set to ModConv2dBias.LEARNED if `modulation`=True, "
                    f"but modulation is set to False."
                )
            self.bias = nn.Sequential(
                nn.Linear(aux_in_features, fc_hidden_features, bias=fc_bias),
                nn.PReLU(),
                nn.Linear(fc_hidden_features, out_channels, bias=fc_bias),
            )
        else:
            self.bias = None

    def __repr__(self):
        """Representation of "class:`ModConv2d`."""
        return (
            f"ModConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, bias={self.bias}, aux_in_features={self.aux_in_features}, "
            f"fc_hidden_features={self.fc_hidden_features}, fc_bias={self.fc_bias}, fc_groups={self.fc_groups}, "
            f"fc_activation={self.fc_activation})"
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of :class:`ModConv2d`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, `in_channels`, H, W).
        y : torch.Tensor, optional
            Auxiliary variable of shape (N, `aux_in_features`) to be used if `modulation` is set to True. Default: None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, `out_channels`, H_out, W_out).
        """
        if not self.modulation:
            out = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            fc_out = self.fc(y)

            if self.fc_groups is not None and self.fc_groups > 1:
                fc_out = fc_out.view(-1, 1, self.out_channels // self.fc_groups, self.in_channels // self.fc_groups)
                fc_out = nn.functional.interpolate(fc_out, size=(self.out_channels, self.in_channels), mode="nearest")

            fc_out = fc_out.view(x.shape[0], self.out_channels, self.in_channels, 1, 1)
            out = torch.cat(
                [
                    F.conv2d(
                        x[i : i + 1],
                        fc_out[i] * self.weight,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )

        if self.bias is not None:
            if isinstance(self.bias, nn.parameter.Parameter):
                bias = self.bias.view(1, -1, 1, 1)
            else:
                bias = self.bias(y).view(x.shape[0], -1, 1, 1)
            out = out + bias

        return out


class ModConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrTuple,
        modulation: bool = False,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[int] = None,
        fc_bias: Optional[bool] = True,
        fc_groups: Optional[int] = None,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
    ):
        """Inits :class:`ModConvTranspose2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple of int
            Size of the convolutional kernel.
        modulation : bool, optional
            If True, apply modulation using an MLP on the auxiliary variable `y`, by default False.
        stride : int or tuple of int, optional
            Stride of the convolution, by default 1.
        padding : int or tuple of int, optional
            Padding added to all sides of the input, by default 0.
        dilation : int or tuple of int, optional
            Spacing between kernel elements, by default 1.
        bias : ModConv2dBias, optional
            Type of bias, by default ModConv2dBias.PARAM.
        aux_in_features : int, optional
            Number of features in the auxiliary input variable `y`, by default None.
        fc_hidden_features : int, optional
            Number of hidden features in the modulation MLP, by default None.
        fc_bias : bool, optional
            If True, enable bias in the modulation MLP, by default True.
        fc_groups : int, optional
            If not None and greater than 1, then the MLP output shape will be divided by `fc_grpups` ** 2, and
            expanded to the convolution weight via 'nearest' interpolation. Can be used to reduce memory. Default: None.
        fc_activation : ModConvActivation
            Activation function to be applied after MLP layer. Default: ModConvActivation.SIGMOID.
        """
        super().__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        k = math.sqrt(1 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = nn.Parameter(
            torch.FloatTensor(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias

        if modulation:
            if aux_in_features is None:
                raise ValueError(f"Value for `aux_in_features` can't be None with `modulation`=True.")
            if fc_hidden_features is None:
                raise ValueError(f"Value for `fc_hidden_features` can't be None with `modulation`=True.")

            if fc_groups is not None and fc_groups > 1:
                fc_out_features = (in_channels // fc_groups) * (out_channels // fc_groups)
            else:
                fc_out_features = in_channels * out_channels

            self.fc_groups = fc_groups

            self.fc = nn.Sequential(
                nn.Linear(aux_in_features, fc_hidden_features, bias=fc_bias),
                nn.PReLU(),
                nn.Linear(fc_hidden_features, fc_out_features, bias=fc_bias),
                *(
                    (nn.Sigmoid(),)
                    if fc_activation == ModConvActivation.SIGMOID
                    else (nn.Softplus(),)
                    if fc_activation == ModConvActivation.SOFTPLUS
                    else ()
                ),
            )

            self.fc_activation = fc_activation

        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if not modulation:
                raise ValueError(
                    f"Bias can only be set to ModConv2dBias.LEARNED if `modulation`=True, "
                    f"but modulation is set to False."
                )
            self.bias = nn.Sequential(
                nn.Linear(aux_in_features, fc_hidden_features, bias=fc_bias),
                nn.PReLU(),
                nn.Linear(fc_hidden_features, out_channels, bias=fc_bias),
            )
        else:
            self.bias = None

    def __repr__(self):
        """Representation of "class:`ModConvTranspose2d`."""
        return (
            f"ModConvTranspose2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, bias={self.bias}, aux_in_features={self.aux_in_features}, "
            f"fc_hidden_features={self.fc_hidden_features}, fc_bias={self.fc_bias}, fc_groups={self.fc_groups}, "
            f"fc_activation={self.fc_activation})"
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of :class:`ModConvTranspose2d`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, `in_channels`, H, W).
        y : torch.Tensor, optional
            Auxiliary variable of shape (N, `aux_in_features`) to be used if `modulation` is set to True. Default: None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, `out_channels`, H_out, W_out).
        """
        if not self.modulation:
            out = F.conv_transpose2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            fc_out = self.fc(y)

            if self.fc_groups is not None and self.fc_groups > 1:
                fc_out = fc_out.view(-1, 1, self.in_channels // self.fc_groups, self.out_channels // self.fc_groups)
                fc_out = nn.functional.interpolate(fc_out, size=(self.in_channels, self.out_channels), mode="nearest")

            fc_out = fc_out.view(x.shape[0], self.in_channels, self.out_channels, 1, 1)
            out = torch.cat(
                [
                    F.conv_transpose2d(
                        x[i : i + 1],
                        fc_out[i] * self.weight,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )

        if self.bias is not None:
            if isinstance(self.bias, nn.parameter.Parameter):
                bias = self.bias.view(1, -1, 1, 1)
            else:
                bias = self.bias(y).view(x.shape[0], -1, 1, 1)
            out = out + bias

        return out
