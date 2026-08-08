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
"""This module provides the implementation of vSHARP model.

Most specifically, vSHARP is the variable Splitting Half-quadratic ADMM algorithm for Reconstruction
of inverse-Problems (vSHARPP) model as presented in [1]_.


References
----------

.. [1] George Yiasemis et. al. vSHARP: variable Splitting Half-quadratic ADMM algorithm for Reconstruction
    of inverse-Problems (2023). https://arxiv.org/abs/2309.09954.

"""

from __future__ import annotations

from typing import Optional, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import apply_mask, expand_operator, reduce_operator
from direct.nn.adain.adain import NormType
from direct.nn.conv.modulated import (
    ModConv2dBias,
    ModConv3d,
    ModConvActivation,
    ModConvType,
    ModulationParams,
    mod_conv2d,
)
from direct.nn.get_nn_model_config import ModelName, _get_model_config, _get_relu_activation
from direct.nn.types import ActivationType, InitType
from direct.nn.unet.unet_3d import NormUnetModel3d, UnetModel3d
from direct.types import FFTOperator


class LagrangeMultipliersInitializer(nn.Module):
    """A convolutional neural network model that initializes the Lagrange multiplier of the :class:`VSharpNet` [1]_.

    More specifically, it produces an initial value for the Lagrange Multiplier based on the zero-filled image:

    .. math::

        u^0 = \\mathcal{G}_{\\psi}(x^0).

    References
    ----------
    .. [1] George Yiasemis et al., "VSHARP: Variable Splitting Half-quadratic ADMM Algorithm for Reconstruction
        of Inverse Problems" (2023). https://arxiv.org/abs/2309.09954.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        multiscale_depth: int = 1,
        activation: ActivationType = ActivationType.PRELU,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        modulation_at_input: bool = False,
    ) -> None:
        """Inits :class:`LagrangeMultipliersInitializer`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        channels : tuple of ints
            Number of output channels for each convolutional layer.
        dilations : tuple of ints
            Dilation factor for each convolutional layer.
        multiscale_depth : int
            Number of multiscale features to include in the output. Default: 1.
        activation : ActivationType
            Activation function. Default: ActivationType.PRELU.
        conv_modulation : ModConvType
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
            If True, modulation is only applied at the first layers. Default: False.
        """
        super().__init__()

        modulation_params = ModulationParams(
            modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        self.conv_blocks = nn.ModuleList()
        tch = in_channels
        for i, (curr_channels, curr_dilations) in enumerate(zip(channels, dilations)):
            block_modulation_params = modulation_params
            if conv_modulation != ModConvType.NONE:
                if modulation_at_input and i > 1:
                    block_modulation_params = ModulationParams(
                        modulation=ModConvType.NONE,
                        aux_in_features=aux_in_features,
                        fc_hidden_features=fc_hidden_features,
                        fc_groups=fc_groups,
                        fc_activation=fc_activation,
                        num_weights=num_weights,
                    )

            block = nn.ModuleList(
                [
                    nn.ReplicationPad2d(curr_dilations),
                    mod_conv2d(
                        tch,
                        curr_channels,
                        kernel_size=3,
                        padding=0,
                        dilation=curr_dilations,
                        # PARAM = standard nn.Parameter bias (LEARNED requires modulation).
                        bias=ModConv2dBias.PARAM,
                        modulation_params=block_modulation_params,
                    ),
                ]
            )
            tch = curr_channels
            self.conv_blocks.append(block)

        out_modulation_params = modulation_params
        if (conv_modulation != ModConvType.NONE) and modulation_at_input:
            out_modulation_params = ModulationParams(
                modulation=ModConvType.NONE,
                aux_in_features=aux_in_features,
                fc_hidden_features=fc_hidden_features,
                fc_groups=fc_groups,
                fc_activation=fc_activation,
                num_weights=num_weights,
            )
        tch = np.sum(channels[-multiscale_depth:]).item()
        self.out_block = mod_conv2d(
            tch,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=ModConv2dBias.PARAM,
            modulation_params=out_modulation_params,
        )

        self.multiscale_depth = multiscale_depth
        self.activation = _get_relu_activation(activation)
        self.conv_modulation = conv_modulation

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of :class:`LagrangeMultipliersInitializer`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).
        y : torch.Tensor, optional
            Auxiliary tensor of shape (batch_size, aux_in_features). Default: None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        features = []
        for block in self.conv_blocks:
            block_modules = cast(nn.ModuleList, block)
            x = block_modules[0](x)
            if self.conv_modulation != ModConvType.NONE:
                x = F.relu(block_modules[1](x, y), inplace=True)
            else:
                x = F.relu(block_modules[1](x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)

        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)

        if self.conv_modulation != ModConvType.NONE:
            return self.activation(self.out_block(x, y))
        return self.activation(self.out_block(x))


class VSharpNet(nn.Module):
    """Variable Splitting Half-quadratic ADMM algorithm for Reconstruction of Parallel MRI [1]_.

    This model incorporates an iterative optimization algorithm (z-step, x-step, u-step) and
    supports optional modulated convolutions conditioned on auxiliary data as proposed in [2]_.

    References
    ----------
    .. [1] George Yiasemis et al., "VSHARP: Variable Splitting Half-quadratic ADMM Algorithm for Reconstruction
        of Inverse Problems" (2023). https://arxiv.org/abs/2309.09954.

    .. [2] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_steps: int,
        num_steps_dc_gd: int,
        image_init: InitType = InitType.SENSE,
        no_parameter_sharing: bool = True,
        image_model_architecture: ModelName = ModelName.UNET,
        initializer_channels: tuple[int, ...] = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        initializer_activation: ActivationType = ActivationType.PRELU,
        auxiliary_steps: int = 0,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        modulation_at_input: bool = False,
        **kwargs,
    ) -> None:
        """Inits :class:`VSharpNet`.

        Parameters
        ----------
        forward_operator : FFTOperator
            Forward operator function.
        backward_operator : FFTOperator
            Backward operator function.
        num_steps : int
            Number of steps in the ADMM algorithm.
        num_steps_dc_gd : int
            Number of gradient descent steps for data consistency.
        image_init : InitType
            Image initialization method. Default: InitType.SENSE.
        no_parameter_sharing : bool
            If True, each ADMM step has its own denoiser. Default: True.
        image_model_architecture : ModelName
            Denoiser model architecture. Default: ModelName.UNET.
        initializer_channels : tuple[int, ...]
            Output channels for the Lagrange initializer layers. Default: (32, 32, 64, 64).
        initializer_dilations : tuple[int, ...]
            Dilations for the Lagrange initializer layers. Default: (1, 1, 2, 4).
        initializer_multiscale : int
            Multiscale depth for the initializer. Default: 1.
        initializer_activation : ActivationType
            Activation for the initializer output. Default: ActivationType.PRELU.
        auxiliary_steps : int
            Number of auxiliary output steps. -1 uses all steps. Default: 0.
        conv_modulation : ModConvType
            Modulation type for convolutions. Default: ModConvType.NONE.
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
            If True, only the first layers use modulation. Default: False.
        **kwargs
            Additional keyword arguments for the image model.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in ("model_name", "log_aux", "auxiliary_features") and not extra_key.startswith("image_"):
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
        self.num_steps = num_steps
        self.num_steps_dc_gd = num_steps_dc_gd
        self.no_parameter_sharing = no_parameter_sharing
        self.conv_modulation = conv_modulation

        image_model, image_model_kwargs = _get_model_config(
            image_model_architecture,
            in_channels=COMPLEX_SIZE * 3,
            out_channels=COMPLEX_SIZE,
            modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            modulation_at_input=modulation_at_input,
            **{k.replace("image_", ""): v for (k, v) in kwargs.items() if "image_" in k},
        )

        self.denoiser_blocks = nn.ModuleList()
        for _ in range(num_steps if self.no_parameter_sharing else 1):
            self.denoiser_blocks.append(image_model(**image_model_kwargs))

        self.initializer = LagrangeMultipliersInitializer(
            in_channels=COMPLEX_SIZE,
            out_channels=COMPLEX_SIZE,
            channels=initializer_channels,
            dilations=initializer_dilations,
            multiscale_depth=initializer_multiscale,
            activation=initializer_activation,
            conv_modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            modulation_at_input=modulation_at_input,
        )

        self.learning_rate_eta = nn.Parameter(torch.ones(num_steps_dc_gd, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_eta, 0.0, 1.0, 0.0)

        self.rho = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.trunc_normal_(self.rho, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in [InitType.SENSE, InitType.ZERO_FILLED]:
            raise ValueError(
                f"Unknown image_initialization. Expected InitType.SENSE or InitType.ZERO_FILLED. Got {image_init}."
            )
        self.image_init = image_init

        if not ((auxiliary_steps == -1) or (0 < auxiliary_steps <= num_steps)):
            raise ValueError(
                f"Number of auxiliary steps should be -1 to use all steps or a positive"
                f" integer <= than `num_steps`. Received {auxiliary_steps}."
            )
        if auxiliary_steps == -1:
            self.auxiliary_steps = list(range(num_steps))
        else:
            self.auxiliary_steps = list(range(num_steps - min(auxiliary_steps, num_steps), num_steps))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Computes forward pass of :class:`VSharpNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        auxiliary_data: torch.Tensor, optional
            Auxiliary tensor of shape (N, aux_in_features). Default: None.

        Returns
        -------
        out : list of torch.Tensors
            List of output images of shape (N, height, width, complex=2).
        """
        out = []
        if self.image_init == InitType.SENSE:
            x = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
        else:
            x = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)

        z = x.clone()

        if self.conv_modulation != ModConvType.NONE:
            u = self.initializer(x.permute(0, 3, 1, 2), auxiliary_data).permute(0, 2, 3, 1)
        else:
            u = self.initializer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        for admm_step in range(self.num_steps):
            denoiser_input = [
                torch.cat(
                    [z, x, u / self.rho[admm_step]],
                    dim=self._complex_dim,
                ).permute(0, 3, 1, 2)
            ]
            if auxiliary_data is not None and (
                self.conv_modulation != ModConvType.NONE
                or (
                    hasattr(self.denoiser_blocks[0], "norm_type")
                    and self.denoiser_blocks[0].norm_type == NormType.ADAIN
                )
            ):
                denoiser_input.append(auxiliary_data)

            z = self.denoiser_blocks[admm_step if self.no_parameter_sharing else 0](*denoiser_input).permute(0, 2, 3, 1)

            for dc_gd_step in range(self.num_steps_dc_gd):
                dc = apply_mask(
                    self.forward_operator(
                        expand_operator(x, sensitivity_map, self._coil_dim),
                        dim=self._spatial_dims,
                    )
                    - masked_kspace,
                    sampling_mask,
                    return_mask=False,
                )
                dc = self.backward_operator(dc, dim=self._spatial_dims)
                dc = reduce_operator(dc, sensitivity_map, self._coil_dim)

                x = x - self.learning_rate_eta[dc_gd_step] * (dc + self.rho[admm_step] * (x - z) + u)

            if admm_step in self.auxiliary_steps:
                out.append(x)

            u = u + self.rho[admm_step] * (x - z)

        return out


class LagrangeMultipliersInitializer3D(torch.nn.Module):
    """A convolutional neural network model that initializes the Lagrange multiplier of :class:`VSharpNet3D`.

    This is an extension to 3D data of :class:`LagrangeMultipliersInitializer`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        multiscale_depth: int = 1,
        activation: ActivationType = ActivationType.PRELU,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        modulation_at_input: bool = False,
    ):
        """Initializes :class:`LagrangeMultipliersInitializer3D`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        channels : tuple of ints
            Number of output channels for each convolutional layer.
        dilations : tuple of ints
            Dilation factor for each convolutional layer.
        multiscale_depth : int
            Number of multiscale features to include in the output. Default: 1.
        activation : ActivationType
            Activation function. Default: ActivationType.PRELU.
        conv_modulation : ModConvType
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
            If True, modulation is only applied at the first layers. Default: False.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        tch = in_channels
        for i, (curr_channels, curr_dilations) in enumerate(zip(channels, dilations)):
            modulation = conv_modulation
            if conv_modulation != ModConvType.NONE:
                if modulation_at_input and i > 1:
                    modulation = ModConvType.NONE

            block = nn.ModuleList(
                [
                    nn.ReplicationPad3d(curr_dilations),
                    ModConv3d(
                        tch,
                        curr_channels,
                        kernel_size=3,
                        padding=0,
                        dilation=curr_dilations,
                        modulation=modulation,
                        bias=ModConv2dBias.PARAM,
                        aux_in_features=aux_in_features,
                        fc_hidden_features=fc_hidden_features,
                        fc_groups=fc_groups,
                        fc_activation=fc_activation,
                        num_weights=num_weights,
                    ),
                ]
            )
            tch = curr_channels
            self.conv_blocks.append(block)

        modulation = (
            ModConvType.NONE if ((conv_modulation != ModConvType.NONE) and modulation_at_input) else conv_modulation
        )
        tch = np.sum(channels[-multiscale_depth:]).item()
        self.out_block = ModConv3d(
            tch,
            out_channels,
            kernel_size=1,
            padding=0,
            modulation=modulation,
            bias=ModConv2dBias.PARAM,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        self.multiscale_depth = multiscale_depth
        self.activation = _get_relu_activation(activation)
        self.conv_modulation = conv_modulation

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of :class:`LagrangeMultipliersInitializer3D`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, z, x, y).
        y : torch.Tensor, optional
            Auxiliary tensor of shape (batch_size, aux_in_features). Default: None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, z, x, y).
        """
        features = []
        for block in self.conv_blocks:
            block_modules = cast(nn.ModuleList, block)
            x = block_modules[0](x)
            if self.conv_modulation != ModConvType.NONE:
                x = F.relu(block_modules[1](x, y), inplace=True)
            else:
                x = F.relu(block_modules[1](x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)

        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)

        if self.conv_modulation != ModConvType.NONE:
            return self.activation(self.out_block(x, y))
        return self.activation(self.out_block(x))


class VSharpNet3D(nn.Module):
    """VSharpNet 3D version using 3D U-Nets as denoisers.

    This is an extension to 3D of :class:`VSharpNet`. For the original paper refer to [1]_.
    Supports conditional weight modulation as proposed in [2]_.

    References
    ----------
    .. [1] George Yiasemis et al., "VSHARP: Variable Splitting Half-quadratic ADMM Algorithm for Reconstruction
        of Inverse Problems" (2023). https://arxiv.org/abs/2309.09954.

    .. [2] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_steps: int,
        num_steps_dc_gd: int,
        image_init: InitType = InitType.SENSE,
        no_parameter_sharing: bool = True,
        initializer_channels: tuple[int, ...] = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        initializer_activation: ActivationType = ActivationType.PRELU,
        auxiliary_steps: int = -1,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        modulation_at_input: bool = False,
        unet_num_filters: int = 32,
        unet_num_pool_layers: int = 4,
        unet_dropout: float = 0.0,
        unet_norm: bool = False,
        unet_norm_type: NormType = NormType.INSTANCE,
        unet_adain_hidden_features: Optional[tuple[int]] = None,
        **kwargs,
    ):
        """Inits :class:`VSharpNet3D`.

        Parameters
        ----------
        forward_operator : FFTOperator
            Forward operator function.
        backward_operator : FFTOperator
            Backward operator function.
        num_steps : int
            Number of steps in the ADMM algorithm.
        num_steps_dc_gd : int
            Number of gradient descent steps for data consistency.
        image_init : InitType
            Image initialization method. Default: InitType.SENSE.
        no_parameter_sharing : bool
            If True, each ADMM step has its own denoiser. Default: True.
        initializer_channels : tuple[int, ...]
            Output channels for the Lagrange initializer layers. Default: (32, 32, 64, 64).
        initializer_dilations : tuple[int, ...]
            Dilations for the Lagrange initializer layers. Default: (1, 1, 2, 4).
        initializer_multiscale : int
            Multiscale depth for the initializer. Default: 1.
        initializer_activation : ActivationType
            Activation for the initializer output. Default: ActivationType.PRELU.
        auxiliary_steps : int
            Number of auxiliary output steps. -1 uses all steps. Default: -1.
        conv_modulation : ModConvType
            Modulation type for convolutions. Default: ModConvType.NONE.
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
            If True, only the first layers use modulation. Default: False.
        unet_num_filters : int
            U-Net first layer filter count. Default: 32.
        unet_num_pool_layers : int
            U-Net depth. Default: 4.
        unet_dropout : float
            U-Net dropout. Default: 0.0.
        unet_norm : bool
            Whether to use normalized U-Net. Default: False.
        unet_norm_type : NormType
            Normalization type for U-Net. Default: NormType.INSTANCE.
        unet_adain_hidden_features : tuple[int], optional
            Hidden features for AdaIN in U-Net.
        **kwargs
            Additional keyword arguments (e.g., model_name, log_aux).
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in ("model_name", "log_aux", "auxiliary_features"):
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
        self.num_steps = num_steps
        self.num_steps_dc_gd = num_steps_dc_gd
        self.no_parameter_sharing = no_parameter_sharing
        self.conv_modulation = conv_modulation
        self.unet_norm_type = unet_norm_type

        unet = UnetModel3d if not unet_norm else NormUnetModel3d

        self.denoiser_blocks = nn.ModuleList()
        for _ in range(num_steps if self.no_parameter_sharing else 1):
            self.denoiser_blocks.append(
                unet(
                    in_channels=COMPLEX_SIZE * 3,
                    out_channels=COMPLEX_SIZE,
                    num_filters=unet_num_filters,
                    num_pool_layers=unet_num_pool_layers,
                    dropout_probability=unet_dropout,
                    modulation=conv_modulation,
                    aux_in_features=aux_in_features,
                    fc_hidden_features=fc_hidden_features,
                    fc_groups=fc_groups,
                    fc_activation=fc_activation,
                    num_weights=num_weights,
                    modulation_at_input=modulation_at_input,
                    norm_type=unet_norm_type,
                    adain_hidden_features=unet_adain_hidden_features,
                )
            )

        self.initializer = LagrangeMultipliersInitializer3D(
            in_channels=COMPLEX_SIZE,
            out_channels=COMPLEX_SIZE,
            channels=initializer_channels,
            dilations=initializer_dilations,
            multiscale_depth=initializer_multiscale,
            activation=initializer_activation,
            conv_modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
            modulation_at_input=modulation_at_input,
        )

        self.learning_rate_eta = nn.Parameter(torch.ones(num_steps_dc_gd, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_eta, 0.0, 1.0, 0.0)

        self.rho = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.trunc_normal_(self.rho, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in [InitType.SENSE, InitType.ZERO_FILLED]:
            raise ValueError(
                f"Unknown image_initialization. Expected InitType.SENSE or InitType.ZERO_FILLED. Got {image_init}."
            )
        self.image_init = image_init

        if not (auxiliary_steps == -1 or 0 < auxiliary_steps <= num_steps):
            raise ValueError(
                f"Number of auxiliary steps should be -1 to use all steps or a positive"
                f" integer <= than `num_steps`. Received {auxiliary_steps}."
            )
        if auxiliary_steps == -1:
            self.auxiliary_steps = list(range(num_steps))
        else:
            self.auxiliary_steps = list(range(num_steps - min(auxiliary_steps, num_steps), num_steps))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (3, 4)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Computes forward pass of :class:`VSharpNet3D`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, slice, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, slice, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, 1 or slice, height, width, 1).
        auxiliary_data: torch.Tensor, optional
            Auxiliary tensor of shape (N, aux_in_features). Default: None.

        Returns
        -------
        out : list of torch.Tensors
            List of output images of shape (N, slice, height, width, complex=2).
        """
        out = []
        if self.image_init == InitType.SENSE:
            x = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
        else:
            x = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)

        z = x.clone()

        if self.conv_modulation != ModConvType.NONE:
            u = self.initializer(x.permute(0, 4, 1, 2, 3), auxiliary_data).permute(0, 2, 3, 4, 1)
        else:
            u = self.initializer(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        for admm_step in range(self.num_steps):
            denoiser_input = [
                torch.cat(
                    [z, x, u / self.rho[admm_step]],
                    dim=self._complex_dim,
                ).permute(0, 4, 1, 2, 3)
            ]
            if auxiliary_data is not None and (
                self.conv_modulation != ModConvType.NONE or self.unet_norm_type == NormType.ADAIN
            ):
                denoiser_input.append(auxiliary_data)

            z = self.denoiser_blocks[admm_step if self.no_parameter_sharing else 0](*denoiser_input).permute(
                0, 2, 3, 4, 1
            )

            for dc_gd_step in range(self.num_steps_dc_gd):
                dc = apply_mask(
                    self.forward_operator(
                        expand_operator(x, sensitivity_map, self._coil_dim),
                        dim=self._spatial_dims,
                    )
                    - masked_kspace,
                    sampling_mask,
                    return_mask=False,
                )
                dc = self.backward_operator(dc, dim=self._spatial_dims)
                dc = reduce_operator(dc, sensitivity_map, self._coil_dim)

                x = x - self.learning_rate_eta[dc_gd_step] * (dc + self.rho[admm_step] * (x - z) + u)

            if admm_step in self.auxiliary_steps:
                out.append(x)

            u = u + self.rho[admm_step] * (x - z)

        return out
