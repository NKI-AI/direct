# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""Convolutional samplers for adaptive k-space line selection."""

import functools
import operator

import torch
from torch import nn

from direct.nn.types import ActivationType


class SingleConv2dBlock(nn.Module):
    """2D convolution block with instance norm, ReLU, dropout, and optional pooling."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        padding: int = 1,
        drop_prob: float = 0,
        pool_size: int = 2,
    ):
        """Inits :class:`SingleConv2dBlock`.

        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        kernel_size : int
            Kernel size. Default: 3.
        padding : int
            Padding. Default: 1.
        drop_prob : float
            Dropout probability. Default: 0.
        pool_size : int
            Size of 2D max-pooling operator. Default: 2.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_prob = drop_prob
        self.pool_size = pool_size

        layers = [
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        ]

        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))  # ty: ignore[invalid-argument-type]

        self.layers = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SingleConv2dBlock`.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape [batch_size, self.in_chans, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, self.out_chans, height, width].
        """
        return self.layers(inp)

    def __repr__(self):
        return (
            f"SingleConv2dBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"kernel_size={self.kernel_size}, padding={self.padding}, "
            f"drop_prob={self.drop_prob}, max_pool_size={self.pool_size})"
        )


class SingleConv3dBlock(nn.Module):
    """3D convolution block with instance norm, ReLU, dropout, and optional pooling."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        padding: int = 1,
        drop_prob: float = 0,
        pool_size: int = 2,
    ):
        """Inits :class:`SingleConv3dBlock`.

        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        kernel_size : int
            Kernel size. Default: 3.
        padding : int
            Padding. Default: 1.
        drop_prob : float
            Dropout probability. Default: 0.
        pool_size : int
            Size of 2D max-pooling operator. Default: 2.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_prob = drop_prob
        self.pool_size = pool_size

        layers = [
            nn.Conv3d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.InstanceNorm3d(out_chans),
            nn.ReLU(),
            nn.Dropout3d(drop_prob),
        ]

        if pool_size > 1:
            layers.append(nn.MaxPool3d(pool_size))  # ty: ignore[invalid-argument-type]

        self.layers = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SingleConv3dBlock`.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape [batch_size, self.in_chans, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, self.out_chans, height, width].
        """
        return self.layers(inp)

    def __repr__(self):
        return (
            f"SingleConv3dBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"kernel_size={self.kernel_size}, padding={self.padding}, "
            f"drop_prob={self.drop_prob}, max_pool_size={self.pool_size})"
        )


class LineConvSampler(nn.Module):
    """Base convolutional sampler that maps observations to action logits."""

    def __init__(
        self,
        input_dim: tuple[int, ...],
        num_actions: int,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        kernel_size: int = 3,
        drop_prob: float = 0,
        num_fc_layers: int = 3,
        activation: ActivationType = ActivationType.LEAKY_RELU,
    ):
        """Inits :class:`LineConvSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple of ints
            Input size of input image or k-space. Can be [self.in_chans, [slice or time], height, width] or
            [self.in_chans, [slice or time], height, width]. Required to dynamically compute the
            input feature dimensions to the linear module.
        num_actions : int
            Number of actions.
        kernel_size : int
            Convolution kernel size. Padding is computed as kernel_size // 2. Default: 3.
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        activation : ActivationType
            Activation function to use: ActivationType.LEAKY_RELU or ActivationType.ELU.
        """
        super().__init__()
        if len(input_dim) not in [3, 4]:
            raise ValueError(
                f"`input_dim` should have length equal to 3 for 2D input or 4 for 3D input."
                f" Received: `input_dim`={input_dim}."
            )

        conv_block = SingleConv2dBlock if (len(input_dim) - 1) == 2 else SingleConv3dBlock

        self.input_dim = input_dim
        self.in_chans = input_dim[0]
        self.num_actions = num_actions
        self.chans = chans

        self.num_pool_layers = num_pool_layers
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.pool_size = 2
        self.num_fc_layers = num_fc_layers
        self.activation = activation

        # Initial from in_chans to chans
        self.channel_layer = conv_block(
            self.in_chans,
            chans,
            kernel_size,
            kernel_size // 2,
            drop_prob,
            pool_size=1,
        )

        # Downsampling convolution
        # These are num_pool_layers layers where each layers 2x2 max pools, and doubles
        # the number of channels.
        self.down_sample_layers = nn.ModuleList(
            [
                conv_block(
                    chans * 2**i,
                    chans * 2 ** (i + 1),
                    kernel_size,
                    kernel_size // 2,
                    drop_prob,
                    pool_size=self.pool_size,
                )
                for i in range(num_pool_layers)
            ]
        )

        self.feature_extractor = nn.Sequential(self.channel_layer, *self.down_sample_layers)

        # Dynamically determinte size of fc_layer
        self.flattened_size = functools.reduce(
            operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape)
        )

        fc_out: list[nn.Module] = []
        for layer in range(self.num_fc_layers):
            in_features = fc_size
            out_features = fc_size
            if layer == 0:
                in_features = self.flattened_size
            if layer + 1 == self.num_fc_layers:
                out_features = num_actions
            fc_out.append(nn.Linear(in_features=in_features, out_features=out_features))

            if layer + 1 < self.num_fc_layers:
                act: nn.Module
                if activation == ActivationType.LEAKY_RELU:
                    act = nn.LeakyReLU()
                elif activation == ActivationType.ELU:  # ty: ignore[unresolved-attribute]
                    act = nn.ELU()
                else:
                    raise RuntimeError(
                        f"Invalid activation function {activation}. "
                        f"Should be ActivationType.LEAKY_RELU or ActivationType.ELU."
                    )
                fc_out.append(act)

        self.fc_out = nn.Sequential(*fc_out)

    def forward(self, observation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute action logits from an observation tensor.

        Parameters
        ----------
        observation : torch.Tensor
            Input tensor in image or k-space layout.
        mask : torch.Tensor
            Current sampling mask.

        Returns
        -------
        torch.Tensor
            Action logits of shape ``(batch, num_actions)``.
        """
        raise NotImplementedError


class ImageLineConvSampler(LineConvSampler):
    """Convolutional sampler operating on image-domain observations."""

    def __init__(
        self,
        input_dim: tuple[int, ...],
        num_actions: int,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        num_fc_layers: int = 3,
        activation: ActivationType = ActivationType.LEAKY_RELU,
    ):
        """Inits :class:`ImageLineConvSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple of ints
            Input size of input image. Can be [self.in_chans, [slice or time], height, width] or
            [self.in_chans, [slice or time], height, width]. Required to dynamically compute the
            input feature dimensions to the linear module.
        num_actions : int
            Number of actions.
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        activation : str
            Activation function to use: leakyrelu or elu.
        """
        super().__init__(
            input_dim=input_dim,
            num_actions=num_actions,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            drop_prob=drop_prob,
            num_fc_layers=num_fc_layers,
            activation=activation,
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        """Compute action logits from an image observation.

        Parameters
        ----------
        image : torch.Tensor
            Image tensor of shape ``[batch, channels, [slice,] height, width]``.
        mask : torch.Tensor
            Mask tensor of shape ``[batch, 1, [1,] 1 or height, width]``, containing 0s and 1s.

        Returns
        -------
        torch.Tensor
            Action logits of shape ``(batch, num_actions)``.
        """

        # Image embedding
        image_emb = self.feature_extractor(image)

        # flatten all but batch dimension before FC layers
        image_emb = image_emb.flatten(start_dim=1)

        out = self.fc_out(image_emb)

        return out


class KSpaceLineConvSampler(LineConvSampler):
    """Convolutional sampler operating on k-space observations."""

    def __init__(
        self,
        input_dim: tuple[int, ...],
        num_actions: int,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        num_fc_layers: int = 3,
        activation: ActivationType = ActivationType.LEAKY_RELU,
    ):
        """Inits :class:`KSpaceLineConvSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple of ints
            Input size of input k-space. Can be [self.in_chans, [slice or time], height, width] or
            [self.in_chans, [slice or time], height, width]. Required to dynamically compute the
            input feature dimensions to the linear module.
        num_actions : int
            Number of actions.
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        activation : str
            Activation function to use: leakyrelu or elu.
        """
        super().__init__(
            input_dim=input_dim,
            num_actions=num_actions,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            kernel_size=7,
            drop_prob=drop_prob,
            num_fc_layers=num_fc_layers,
            activation=activation,
        )
        self.coil_dim = 1

    def forward(self, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        """Compute action logits from a k-space observation.

        Parameters
        ----------
        kspace : torch.Tensor
            K-space tensor of shape ``[batch, coils, channels, [slice,] height, width]``.
        mask : torch.Tensor
            Mask tensor of shape ``[batch, 1, [1,] 1 or height, width]``, containing 0s and 1s.

        Returns
        -------
        torch.Tensor
            Action logits of shape ``(batch, num_actions)``.
        """

        # Kspace embeddings, aggregated via summation

        n_coils = kspace.shape[self.coil_dim]

        embeddings = []
        for i in range(n_coils):
            coil_emb = self.feature_extractor(kspace[:, i])
            # flatten all but batch dimension before FC layers
            coil_emb = coil_emb.flatten(start_dim=1)
            embeddings.append(coil_emb)

        out = self.fc_out(sum(embeddings))

        return out
