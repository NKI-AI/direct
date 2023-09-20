# coding=utf-8
# Copyright (c) DIRECT Contributors


from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import apply_mask, expand_operator, reduce_operator
from direct.nn.get_nn_model_config import ModelName, _get_activation, _get_model_config
from direct.nn.types import ActivationType, InitType
from direct.nn.unet.unet_3d import NormUnetModel3d, UnetModel3d


class LagrangeMultipliersInitializer(nn.Module):
    """A convolutional neural network model that initializers the Lagrange multiplier of the vSHARPNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        multiscale_depth: int = 1,
        activation: ActivationType = ActivationType.prelu,
    ):
        """Inits :class:`LagrangeMultipliersInitializer`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        channels : tuple of ints
            Tuple of integers specifying the number of output channels for each convolutional layer in the network.
        dilations : tuple of ints
            Tuple of integers specifying the dilation factor for each convolutional layer in the network.
        multiscale_depth : int
            Number of multiscale features to include in the output. Default: 1.
        """
        super().__init__()

        # Define convolutional blocks
        self.conv_blocks = nn.ModuleList()
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = nn.Sequential(
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            )
            tch = curr_channels
            self.conv_blocks.append(block)

        # Define output block
        tch = np.sum(channels[-multiscale_depth:])
        block = nn.Conv2d(tch, out_channels, 1, padding=0)
        self.out_block = nn.Sequential(block)

        self.multiscale_depth = multiscale_depth

        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`LagrangeMultipliersInitializer`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """

        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)

        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)

        return self.activation(self.out_block(x))


class VSharpNet(nn.Module):
    """Variable Splitting Half-quadratic ADMM algorithm for Reconstruction of Parallel MRI.

    Variable Splitting Half Quadratic  VSharpNet is a deep learning model that solves the augmented Lagrangian derivation
    of the variable half quadratic splitting problem using ADMM (Alternating Direction Method of Multipliers).
    It is designed for solving inverse problems in magnetic resonance imaging (MRI).

    The VSharpNet model incorporates an iterative optimization algorithm that consists of three steps: z-step, x-step,
    and u-step:

    .. math ::
        \vec{z}^{t+1}  = \argmin_{\vec{z}}\, \lambda \, \mathcal{G}(\vec{z}) +
            \frac{\rho}{2} \big | \big | \vec{x}^{t} - \vec{z} + \frac{\vec{u}^t}{\rho} \big | \big |_2^2
             \quad \Big[\vec{z}\text{-step}\Big]
        \vec{x}^{t+1}  = \argmin_{\vec{x}}\, \frac{1}{2} \big | \big | \mathcal{A}_{\mat{U},\mat{S}}(\vec{x}) -
            \tilde{\vec{y}} \big | \big |_2^2 + \frac{\rho}{2} \big | \big | \vec{x} - \vec{z}^{t+1}
            + \frac{\vec{u}^t}{\rho} \big | \big |_2^2 \quad \Big[\vec{x}\text{-step}\Big]
        \vec{u}^{t+1} = \vec{u}^t + \rho (\vec{x}^{t+1} - \vec{z}^{t+1}) \quad \Big[\vec{u}\text{-step}\Big]


    In the z-step, the model minimizes the augmented Lagrangian function with respect to z using DL based
    denoisers.

    In the x-step, it optimizes x by minimizing the data consistency term by unrolling a
    gradient descent scheme (DC-GD).

    In the u-step, the model updates the Lagrange multiplier u. These steps are performed iteratively for
    a specified number of steps.

    The VSharpNet model supports both image and k-space domain parameterizations. It includes an initializer for
    Lagrange multipliers.

    It can also incorporate auxiliary steps during training for improved performance.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps: int,
        num_steps_dc_gd: int,
        image_init: str = InitType.sense,
        no_parameter_sharing: bool = True,
        image_model_architecture: ModelName = ModelName.unet,
        initializer_channels: tuple[int, ...] = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        initializer_activation: ActivationType = ActivationType.prelu,
        kspace_no_parameter_sharing: bool = True,
        kspace_model_architecture: Optional[ModelName] = None,
        auxiliary_steps: int = 0,
        **kwargs,
    ):
        """Inits :class:`VSharpNet`.

        Parameters
        ----------
        forward_operator : Callable
            Forward operator function.
        backward_operator : Callable
            Backward operator function.
        num_steps : int
            Number of steps in the ADMM algorithm.
        num_steps_dc_gd : int
            Number of steps in the Data Consistency using Gradient Descent step of ADMM.
        image_init : str
            Image initialization method. Default: 'sense'.
        no_parameter_sharing : bool
            Flag indicating whether parameter sharing is enabled in the denoiser blocks.
        image_model_architecture : ModelName
            Image model architecture. Default: ModelName.unet.
        initializer_channels : tuple[int, ...]
            Tuple of integers specifying the number of output channels for each convolutional layer in the
             Lagrange multiplier initializer. Default: (32, 32, 64, 64).
        initializer_dilations : tuple[int, ...]
            Tuple of integers specifying the dilation factor for each convolutional layer in the Lagrange multiplier
            initializer. Default: (1, 1, 2, 4).
        initializer_multiscale : int
            Number of multiscale features to include in the  Lagrange multiplier initializer output. Default: 1.
        initializer_activation : ActivationType
            Activation type for the Lagrange multiplier initializer. Default: ActivationType.relu.
        kspace_no_parameter_sharing : bool
            Flag indicating whether parameter sharing is enabled in the k-space denoiser. Ignored if input for
            `kspace_model_architecture` is None. Default: True.
        kspace_model_architecture : ModelName, optional
            K-space model architecture. Default: None.
        auxiliary_steps : int
            Number of auxiliary steps to output. Can be -1 or a positive integer lower or equal to `num_steps`.
            If -1, it uses all steps.
        **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.num_steps = num_steps
        self.num_steps_dc_gd = num_steps_dc_gd

        self.no_parameter_sharing = no_parameter_sharing

        if image_model_architecture not in [
            "unet",
            "normunet",
            "resnet",
            "didn",
            "conv",
            "uformer",
            "vision_transformer",
        ]:
            raise ValueError(f"Invalid value {image_model_architecture} for `image_model_architecture`.")
        if kspace_model_architecture not in [
            "unet",
            "normunet",
            "resnet",
            "didn",
            "conv",
            "uformer",
            "vision_transformer",
            None,
        ]:
            raise ValueError(f"Invalid value {kspace_model_architecture} for `kspace_model_architecture`.")

        image_model, image_model_kwargs = _get_model_config(
            image_model_architecture,
            in_channels=COMPLEX_SIZE * 4 if kspace_model_architecture else COMPLEX_SIZE * 3,
            out_channels=COMPLEX_SIZE,
            **{k.replace("image_", ""): v for (k, v) in kwargs.items() if "image_" in k},
        )

        if kspace_model_architecture:
            self.kspace_no_parameter_sharing = kspace_no_parameter_sharing
            kspace_model, kspace_model_kwargs = _get_model_config(
                kspace_model_architecture,
                in_channels=COMPLEX_SIZE,
                out_channels=COMPLEX_SIZE,
                **{k.replace("kspace_", ""): v for (k, v) in kwargs.items() if "kspace_" in k},
            )
            self.kspace_denoiser = kspace_model(**kspace_model_kwargs)
            self.scale_k = nn.Parameter(torch.ones(1, requires_grad=True))
            nn.init.trunc_normal_(self.scale_k, 0, 0.1, 0.0)
        else:
            self.kspace_denoiser = None

        self.denoiser_blocks = nn.ModuleList()
        for _ in range(num_steps if self.no_parameter_sharing else 1):
            self.denoiser_blocks.append(image_model(**image_model_kwargs))

        self.initializer = LagrangeMultipliersInitializer(
            COMPLEX_SIZE,
            COMPLEX_SIZE,
            channels=initializer_channels,
            dilations=initializer_dilations,
            multiscale_depth=initializer_multiscale,
            activation=initializer_activation,
        )

        self.learning_rate_eta = nn.Parameter(torch.ones(num_steps_dc_gd, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_eta, 0.0, 1.0, 0.0)

        self.rho = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.trunc_normal_(self.rho, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in ["sense", "zero_filled"]:
            raise ValueError(f"Unknown image_initialization. Expected 'sense' or 'zero_filled'. " f"Got {image_init}.")

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
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Computes forward pass of :class:`MRIVarSplitNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.
        sampling_mask: torch.Tensor

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        out = []
        if self.image_init == "sense":
            x = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
        else:
            x = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)

        z = x.clone()

        u = self.initializer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        for admm_step in range(self.num_steps):
            if self.kspace_denoiser:
                kspace_z = self.kspace_denoiser(
                    self.forward_operator(z.contiguous(), dim=[_ - 1 for _ in self._spatial_dims]).permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1)
                kspace_z = self.backward_operator(kspace_z.contiguous(), dim=[_ - 1 for _ in self._spatial_dims])

            z = self.denoiser_blocks[admm_step if self.no_parameter_sharing else 0](
                torch.cat(
                    [z, x, u / self.rho[admm_step]] + ([self.scale_k * kspace_z] if self.kspace_denoiser else []),
                    dim=self._complex_dim,
                ).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)

            for dc_gd_step in range(self.num_steps_dc_gd):
                dc = apply_mask(
                    self.forward_operator(expand_operator(x, sensitivity_map, self._coil_dim), dim=self._spatial_dims)
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
    """A convolutional neural network model that initializes the Lagrange multiplier of the vSHARPNet for 3D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        multiscale_depth: int = 1,
        activation: nn.Module = nn.PReLU(),
    ):
        """Initializes LagrangeMultipliersInitializer3D.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        channels : tuple of ints
            Tuple of integers specifying the number of output channels for each convolutional layer in the network.
        dilations : tuple of ints
            Tuple of integers specifying the dilation factor for each convolutional layer in the network.
        multiscale_depth : int
            Number of multiscale features to include in the output. Default: 1.
        activation : nn.Module
            Activation function. Default: PReLU.
        """
        super().__init__()

        # Define convolutional blocks
        self.conv_blocks = nn.ModuleList()
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = nn.Sequential(
                nn.ReplicationPad3d(curr_dilations),
                nn.Conv3d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            )
            tch = curr_channels
            self.conv_blocks.append(block)

        # Define output block
        tch = np.sum(channels[-multiscale_depth:])
        block = nn.Conv3d(tch, out_channels, 1, padding=0)
        self.out_block = nn.Sequential(block)

        self.multiscale_depth = multiscale_depth
        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of LagrangeMultipliersInitializer3D.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, z, x, y).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, z, x, y).
        """

        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)

        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)

        return self.activation(self.out_block(x))


class VSharpNet3D(nn.Module):
    """VharpNet 3D version."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps: int,
        num_steps_dc_gd: int,
        image_init: str = InitType.sense,
        no_parameter_sharing: bool = True,
        initializer_channels: tuple[int, ...] = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        initializer_activation: ActivationType = ActivationType.prelu,
        auxiliary_steps: int = -1,
        unet_num_filters: int = 32,
        unet_num_pool_layers: int = 4,
        unet_dropout: float = 0.0,
        unet_cwn_conv: bool = False,
        unet_norm: bool = False,
        **kwargs,
    ):
        """Inits :class:`VSharpNet`.

        Parameters
        ----------
        forward_operator : Callable
            Forward operator function.
        backward_operator : Callable
            Backward operator function.
        num_steps : int
            Number of steps in the ADMM algorithm.
        num_steps_dc_gd : int
            Number of steps in the Data Consistency using Gradient Descent step of ADMM.
        image_init : str
            Image initialization method. Default: 'sense'.
        no_parameter_sharing : bool
            Flag indicating whether parameter sharing is enabled in the denoiser blocks.
        image_model_architecture : ModelName
            Image model architecture. Default: ModelName.unet.
        initializer_channels : tuple[int, ...]
            Tuple of integers specifying the number of output channels for each convolutional layer in the
             Lagrange multiplier initializer. Default: (32, 32, 64, 64).
        initializer_dilations : tuple[int, ...]
            Tuple of integers specifying the dilation factor for each convolutional layer in the Lagrange multiplier
            initializer. Default: (1, 1, 2, 4).
        initializer_multiscale : int
            Number of multiscale features to include in the  Lagrange multiplier initializer output. Default: 1.
        initializer_activation : ActivationType
            Activation type for the Lagrange multiplier initializer. Default: ActivationType.relu.
        kspace_no_parameter_sharing : bool
            Flag indicating whether parameter sharing is enabled in the k-space denoiser. Ignored if input for
            `kspace_model_architecture` is None. Default: True.
        kspace_model_architecture : ModelName, optional
            K-space model architecture. Default: None.
        auxiliary_steps : int
            Number of auxiliary steps to output. Can be -1 or a positive integer lower or equal to `num_steps`.
            If -1, it uses all steps.
        **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.num_steps = num_steps
        self.num_steps_dc_gd = num_steps_dc_gd

        self.no_parameter_sharing = no_parameter_sharing

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
                    cwn_conv=unet_cwn_conv,
                )
            )

        self.initializer = LagrangeMultipliersInitializer3D(
            COMPLEX_SIZE,
            COMPLEX_SIZE,
            channels=initializer_channels,
            dilations=initializer_dilations,
            multiscale_depth=initializer_multiscale,
            activation=initializer_activation,
        )

        self.learning_rate_eta = nn.Parameter(torch.ones(num_steps_dc_gd, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_eta, 0.0, 1.0, 0.0)

        self.rho = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.trunc_normal_(self.rho, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in ["sense", "zero_filled"]:
            raise ValueError(f"Unknown image_initialization. Expected 'sense' or 'zero_filled'. " f"Got {image_init}.")

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
    ) -> list[torch.Tensor]:
        """Computes forward pass of :class:`MRIVarSplitNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.
        sampling_mask: torch.Tensor

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, slice, height, width, complex=2).
        """
        out = []
        if self.image_init == "sense":
            x = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
        else:
            x = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)

        z = x.clone()

        u = self.initializer(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        for admm_step in range(self.num_steps):
            z = self.denoiser_blocks[admm_step if self.no_parameter_sharing else 0](
                torch.cat(
                    [z, x, u / self.rho[admm_step]],
                    dim=self._complex_dim,
                ).permute(0, 4, 1, 2, 3)
            ).permute(0, 2, 3, 4, 1)

            for dc_gd_step in range(self.num_steps_dc_gd):
                dc = apply_mask(
                    self.forward_operator(expand_operator(x, sensitivity_map, self._coil_dim), dim=self._spatial_dims)
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
