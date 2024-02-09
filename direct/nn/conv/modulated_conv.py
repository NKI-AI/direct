# coding=utf-8
# Copyright (c) DIRECT Contributors

"""direct.nn.conv.modulated_conv module"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.types import DirectEnum, IntOrTuple

__all__ = ["ModConv2d", "ModConv2dBias", "ModConvActivation", "ModConvType", "ModConvTranspose2d"]


class ModConv2dBias(DirectEnum):
    LEARNED = "learned"
    PARAM = "param"
    NONE = "none"


class ModConvActivation(DirectEnum):
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"


class ModConvType(DirectEnum):
    FEATURES = "features"
    PARTIAL_IN = "partial_in"
    PARTIAL_OUT = "partial_out"
    FULL = "full"
    SUM = "sum"
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
        modulation: ModConvType = ModConvType.NONE,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_bias: Optional[bool] = True,
        fc_groups: int | None = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
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
        modulation : ModConvType, optional
            If not ModConvType.NONE, it will apply modulation using an MLP on the auxiliary variable `y`.
            By default ModConvType.NONE. Can be ModConvType.NONE, ModConvType.FULL, ModConvType.PARTIAL_IN, or
            ModConvType.PARTIAL_OUT.
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
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulation MLP, by default None.
        fc_bias : bool, optional
            If True, enable bias in the modulation MLP, by default True.
        fc_groups : int, optional
            If not None and greater than 1, then the MLP output shape will be divided by `fc_grpups` ** 2, and
            expanded to the convolution weight via 'nearest' interpolation. Can be used to reduce memory.
            Ignored if modulation is ModConvType.None or ModConvType.SUM. Default: None.
        fc_activation : ModConvActivation
            Activation function to be applied after MLP layer. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        """
        super().__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias
        self.fc_groups = fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        if modulation != ModConvType.NONE:
            if aux_in_features is None:
                raise ValueError(
                    f"Value for `aux_in_features` can't be None with `modulation` not set to ModConvType.NONE."
                )
            if fc_hidden_features is None:
                raise ValueError(
                    f"Value for `fc_hidden_features` can't be None with `modulation` not set to ModConvType.NONE."
                )
            if isinstance(fc_hidden_features, int):
                fc_hidden_features = (fc_hidden_features,)
            if fc_groups is None:
                raise ValueError(f"Value for `fc_groups` can't be None with `modulation` not set to ModConvType.NONE.")
            if fc_groups < 1:
                raise ValueError(f"Value for `fc_groups` must be 1 or greater.")

            if modulation == ModConvType.FEATURES:
                mod_out_features = (out_channels // fc_groups) * (in_channels // fc_groups)
            elif modulation == ModConvType.FULL:
                mod_out_features = (
                    (out_channels // fc_groups)
                    * (in_channels // fc_groups)
                    * self.kernel_size[0]
                    * self.kernel_size[1]
                )
            elif modulation == ModConvType.PARTIAL_OUT:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (out_channels // fc_groups)
            elif modulation == ModConvType.PARTIAL_IN:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (in_channels // fc_groups)
            else:
                if (num_weights is None) or (num_weights < 1):
                    raise ValueError(
                        f"Value for `modulation` is set to ModConvType.SUM but received "
                        f"`num_weights`={num_weights}. This should be an integer greater than 1."
                    )
                mod_out_features = num_weights

            fc_hidden_features = fc_hidden_features + (mod_out_features,)

            fc = [nn.Linear(aux_in_features, fc_hidden_features[0], bias=fc_bias), nn.PReLU()]
            for i in range(0, len(fc_hidden_features) - 1):
                fc.append(nn.Linear(fc_hidden_features[i], fc_hidden_features[i + 1]))
                fc.append(nn.PReLU())
            self.fc = nn.Sequential(
                *fc,
                *(
                    (nn.Sigmoid(),)
                    if fc_activation == ModConvActivation.SIGMOID
                    else (nn.Softplus(),)
                    if fc_activation == ModConvActivation.SOFTPLUS
                    else ()
                ),
            )

        # Shape of conv weights
        weight_shape = (out_channels, in_channels, *self.kernel_size)
        # Adjust the weight shape based on modulation type
        if modulation == ModConvType.SUM:
            weight_shape = (num_weights,) + weight_shape
        k = math.sqrt(1 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        # Initialize the weight parameter with uniform random values in the range [-k, k]
        self.weight = nn.Parameter(torch.FloatTensor(*weight_shape).uniform_(-k, k))

        self.bias_type = bias
        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if modulation == ModConvType.NONE:
                raise ValueError(
                    f"Bias can only be set to ModConv2dBias.LEARNED if `modulation` is not ModConvType.NONE, "
                    f"but modulation is set to ModConvType.NONE."
                )
            bias = [nn.Linear(aux_in_features, fc_hidden_features[0], bias=fc_bias)]
            for i in range(0, len(fc_hidden_features) - 1):
                bias.append(nn.PReLU())
                bias.append(
                    nn.Linear(
                        fc_hidden_features[i],
                        fc_hidden_features[i + 1] if i != (len(fc_hidden_features) - 2) else out_channels,
                    )
                )
            self.bias = nn.Sequential(*bias)
        else:
            self.bias = None

    def __repr__(self):
        """Representation of "class:`ModConv2d`."""
        return (
            f"ModConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, bias={self.bias_type}, aux_in_features={self.aux_in_features}, "
            f"fc_hidden_features={self.fc_hidden_features}, fc_bias={self.fc_bias}, "
            f"fc_groups={self.fc_groups}, fc_activation={self.fc_activation}, num_weights{self.num_weights})"
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
        if self.modulation == ModConvType.NONE:
            out = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            fc_out = self.fc(y)

            if self.modulation == ModConvType.SUM:
                weight = (fc_out.view(x.shape[0], -1, 1, 1, 1, 1) * self.weight).sum(1)
            else:
                if self.modulation == ModConvType.FEATURES:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0], 1, self.out_channels // self.fc_groups, self.in_channels // self.fc_groups
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.out_channels, self.in_channels), mode="nearest"
                        )
                    fc_out = fc_out.view(x.shape[0], self.out_channels, self.in_channels, 1, 1)

                elif self.modulation == ModConvType.FULL:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.out_channels // self.fc_groups,
                            self.in_channels // self.fc_groups,
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.kernel_size[1], self.out_channels, self.in_channels), mode="nearest"
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0], self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
                    )

                elif self.modulation == ModConvType.PARTIAL_OUT:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.out_channels // self.fc_groups,
                            1,
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.kernel_size[1], self.out_channels, 1), mode="nearest"
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(x.shape[0], self.out_channels, 1, self.kernel_size[0], self.kernel_size[1])

                else:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0], self.kernel_size[0], self.kernel_size[1], 1, self.in_channels // self.fc_groups
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.kernel_size[1], 1, self.in_channels), mode="nearest"
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(x.shape[0], 1, self.in_channels, self.kernel_size[0], self.kernel_size[1])

                weight = fc_out * self.weight

            out = torch.cat(
                [
                    F.conv2d(
                        x[i : i + 1],
                        weight[i],
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
        modulation: ModConvType = ModConvType.NONE,
        stride: IntOrTuple = 1,
        padding: IntOrTuple = 0,
        dilation: IntOrTuple = 1,
        bias: ModConv2dBias = ModConv2dBias.PARAM,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_bias: Optional[bool] = True,
        fc_groups: int | None = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
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
        modulation : ModConvType, optional
            If not ModConvType.NONE, it will apply modulation using an MLP on the auxiliary variable `y`.
            By default ModConvType.NONE. Can be ModConvType.NONE, ModConvType.FULL, ModConvType.PARTIAL_IN, or
            ModConvType.PARTIAL_OUT.
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
        fc_hidden_features : int or tuple of int, optional
            Number of hidden features in the modulation MLP, by default None.
        fc_bias : bool, optional
            If True, enable bias in the modulation MLP, by default True.
        fc_groups : int, optional
            If not None and greater than 1, then the MLP output shape will be divided by `fc_grpups` ** 2, and
            expanded to the convolution weight via 'nearest' interpolation. Can be used to reduce memory.
            Ignored if modulation is ModConvType.None or ModConvType.SUM. Default: None.
        fc_activation : ModConvActivation
            Activation function to be applied after MLP layer. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weights to use in case modulation is ModConvType.SUM. Default: None.
        """
        super().__init__()

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modulation = modulation
        self.aux_in_features = aux_in_features
        self.fc_hidden_features = fc_hidden_features
        self.fc_bias = fc_bias
        self.fc_groups = fc_groups
        self.fc_activation = fc_activation
        self.num_weights = num_weights

        if modulation != ModConvType.NONE:
            if aux_in_features is None:
                raise ValueError(
                    f"Value for `aux_in_features` can't be None with `modulation` not set to ModConvType.NONE."
                )
            if fc_hidden_features is None:
                raise ValueError(
                    f"Value for `fc_hidden_features` can't be None with `modulation` not set to ModConvType.NONE."
                )
            if isinstance(fc_hidden_features, int):
                fc_hidden_features = (fc_hidden_features,)
            if fc_groups is None:
                raise ValueError(f"Value for `fc_groups` can't be None with `modulation` not set to ModConvType.NONE.")
            if fc_groups < 1:
                raise ValueError(f"Value for `fc_groups` must be 1 or greater.")

            if modulation == ModConvType.FEATURES:
                mod_out_features = (in_channels // fc_groups) * (out_channels // fc_groups)
            elif modulation == ModConvType.FULL:
                mod_out_features = (
                    (in_channels // fc_groups)
                    * (out_channels // fc_groups)
                    * self.kernel_size[0]
                    * self.kernel_size[1]
                )
            elif modulation == ModConvType.PARTIAL_OUT:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (out_channels // fc_groups)
            elif modulation == ModConvType.PARTIAL_IN:
                mod_out_features = self.kernel_size[0] * self.kernel_size[1] * (in_channels // fc_groups)
            else:
                if (num_weights is None) or (num_weights < 1):
                    raise ValueError(
                        f"Value for `modulation` is set to ModConvType.SUM but received "
                        f"`num_weights`={num_weights}. This should be an integer greater than 1."
                    )
                mod_out_features = num_weights

            fc_hidden_features = fc_hidden_features + (mod_out_features,)

            fc = [nn.Linear(aux_in_features, fc_hidden_features[0], bias=fc_bias), nn.PReLU()]

            for i in range(0, len(fc_hidden_features) - 1):
                fc.append(nn.Linear(fc_hidden_features[i], fc_hidden_features[i + 1]))
                fc.append(nn.PReLU())
            self.fc = nn.Sequential(
                *fc,
                *(
                    (nn.Sigmoid(),)
                    if fc_activation == ModConvActivation.SIGMOID
                    else (nn.Softplus(),)
                    if fc_activation == ModConvActivation.SOFTPLUS
                    else ()
                ),
            )

        # Shape of conv weights
        weight_shape = (in_channels, out_channels, *self.kernel_size)
        # Adjust the weight shape based on modulation type
        if modulation == ModConvType.SUM:
            weight_shape = (num_weights,) + weight_shape
        k = math.sqrt(1 / (out_channels * self.kernel_size[0] * self.kernel_size[1]))
        # Initialize the weight parameter with uniform random values in the range [-k, k]
        self.weight = nn.Parameter(torch.FloatTensor(*weight_shape).uniform_(-k, k))

        self.bias_type = bias
        if bias == ModConv2dBias.PARAM:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).uniform_(-k, k))
        elif bias == ModConv2dBias.LEARNED:
            if modulation == ModConvType.NONE:
                raise ValueError(
                    f"Bias can only be set to ModConv2dBias.LEARNED if `modulation` is not ModConvType.NONE, "
                    f"but modulation is set to ModConvType.NONE."
                )
            bias = [nn.Linear(aux_in_features, fc_hidden_features[0], bias=fc_bias)]
            for i in range(0, len(fc_hidden_features) - 1):
                bias.append(nn.PReLU())
                bias.append(
                    nn.Linear(
                        fc_hidden_features[i],
                        fc_hidden_features[i + 1] if i != (len(fc_hidden_features) - 2) else out_channels,
                    )
                )
            self.bias = nn.Sequential(*bias)
        else:
            self.bias = None

    def __repr__(self):
        """Representation of "class:`ModConvTranspose2d`."""
        return (
            f"ModConvTranspose2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, modulation={self.modulation}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias_type}, "
            f"aux_in_features={self.aux_in_features}, fc_hidden_features={self.fc_hidden_features}, "
            f"fc_bias={self.fc_bias}, fc_groups={self.fc_groups}, fc_activation={self.fc_activation}, "
            f"num_weights{self.num_weights})"
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
        if self.modulation == ModConvType.NONE:
            out = F.conv_transpose2d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            fc_out = self.fc(y)

            if self.modulation == ModConvType.SUM:
                weight = (fc_out.view(x.shape[0], -1, 1, 1, 1, 1) * self.weight).sum(1)
            else:
                if self.modulation == ModConvType.FEATURES:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0], 1, self.in_channels // self.fc_groups, self.out_channels // self.fc_groups
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.in_channels, self.out_channels), mode="nearest"
                        )
                    fc_out = fc_out.view(x.shape[0], self.in_channels, self.out_channels, 1, 1)

                elif self.modulation == ModConvType.FULL:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            self.in_channels // self.fc_groups,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.kernel_size[1], self.in_channels, self.out_channels), mode="nearest"
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(
                        x.shape[0], self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]
                    )

                elif self.modulation == ModConvType.PARTIAL_OUT:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0],
                            self.kernel_size[0],
                            self.kernel_size[1],
                            1,
                            self.out_channels // self.fc_groups,
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.kernel_size[1], 1, self.out_channels), mode="nearest"
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(x.shape[0], 1, self.out_channels, self.kernel_size[0], self.kernel_size[1])

                else:
                    if self.fc_groups > 1:
                        fc_out = fc_out.view(
                            x.shape[0], self.kernel_size[0], self.kernel_size[1], self.in_channels // self.fc_groups, 1
                        )
                        fc_out = nn.functional.interpolate(
                            fc_out, size=(self.kernel_size[1], self.in_channels, 1), mode="nearest"
                        )
                        fc_out = fc_out.permute(0, 3, 4, 1, 2)
                    fc_out = fc_out.view(x.shape[0], self.in_channels, 1, self.kernel_size[0], self.kernel_size[1])

                weight = fc_out * self.weight

            out = torch.cat(
                [
                    F.conv_transpose2d(
                        x[i : i + 1],
                        weight[i],
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
