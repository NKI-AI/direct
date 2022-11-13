# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Tuple

import torch
from torch import nn

from direct.nn.types import ActivationType
from direct.nn.utils import get_activation_from_type


class DenseLayer(nn.Module):
    r"""Dense layer for :class:`DenseNet` [1]_.

    It implements a single layer inside :class:`DenseBlock`. It consists of a :math:`1 \times 1` convolution
    with `bottleneck_channels \* expansion` features followed by a :math:`3 \times 3` convolution with a number of
    `expansion` output channels. Each convolution is preceded by a batch normalization layer and an activation function.

    References
    ----------
    .. [1] Huang, Gao, et al. Densely Connected Convolutional Networks. arXiv, 28 Jan. 2018. arXiv.org,
        https://doi.org/10.48550/arXiv.1608.06993.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, expansion: int, activation: nn.Module) -> None:
        r"""Inits :class:

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        bottleneck_channels : int
            Number of bottleneck channels (multiplied by `expansion`) for the output of the :math:`1 \times 1`
            convolution.
        expansion : int
            Number of output channels.
        activation : nn.Module
            Activation factor.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation,
            nn.Conv2d(in_channels, bottleneck_channels * expansion, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(bottleneck_channels * expansion),
            activation,
            nn.Conv2d(bottleneck_channels * expansion, expansion, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`DenseLayer`.

        Parameters
        ----------
        x : torch.Tensor
            Input with number of channels `C`.

        Returns
        -------
        torch.Tensor
            Output wih number of channels `expansion + C`.
        """
        return torch.cat([self.net(x), x], dim=1)


class DenseBlock(nn.Module):
    """Main block of :class:`DenseNet` [1]_.

    References
    ----------
    .. [1] Huang, Gao, et al. Densely Connected Convolutional Networks. arXiv, 28 Jan. 2018. arXiv.org,
        https://doi.org/10.48550/arXiv.1608.06993.
    """

    def __init__(
        self, in_channels: int, num_layers: int, bottleneck_channels: int, expansion: int, activation: nn.Module
    ) -> None:
        """Inits :class:`DenseBlock`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_layers : int
            Number of :class:`DenseLayers` in the block.
        bottleneck_channels : int
            Number of bottleneck channels to use in :class:`DenseLayer`.
        expansion : int
            Expansion channels to use in :class:`DenseLayer`.
        activation : nn.Module
            Activation function to use in :class:`DenseLayer`.
        """
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(
                    in_channels=in_channels + layer_idx * expansion,
                    # Input channels are original plus the feature maps from previous layers
                    bottleneck_channels=bottleneck_channels,
                    expansion=expansion,
                    activation=activation,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`DenseBlock`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        return self.block(x)


class TransitionLayer(nn.Module):
    """Transition layer for :class:`DenseNet` [1]_.

    References
    ----------
    .. [1] Huang, Gao, et al. Densely Connected Convolutional Networks. arXiv, 28 Jan. 2018. arXiv.org,
        https://doi.org/10.48550/arXiv.1608.06993.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation,
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Average the output for each 2x2 pixel group
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    """Densely Connected Convolutional Network as implemented in [1]_.

    References
    ----------
    .. [1] Huang, Gao, et al. Densely Connected Convolutional Networks. arXiv, 28 Jan. 2018. arXiv.org,
        https://doi.org/10.48550/arXiv.1608.06993.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        num_layers: Tuple[int, ...] = (6, 6, 6, 6),
        bottleneck_channels: int = 2,
        expansion: int = 16,
        activation: ActivationType = ActivationType.relu,
    ):
        """Inits :class:`DenseNet`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_classes : int
            Number of output features/classes.
        num_layers : tuple of ints
            Number of layers per block.
        bottleneck_channels : int
            Bottleneck channels for :class:`DenseBlock`.
        expansion : int
            Expansion channels for :class:`DenseBlock`.
        activation : ActivationType
            Activation function.
        """
        super().__init__()

        hidden_channels = expansion * bottleneck_channels  # The start number of hidden channels

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1))
            # No batch norm or activation function as done inside the Dense layers
        )

        # Creating the dense blocks, eventually including transition layers
        blocks = []
        for block_idx, block_num_layers in enumerate(num_layers):
            blocks.append(
                DenseBlock(
                    in_channels=hidden_channels,
                    num_layers=block_num_layers,
                    bottleneck_channels=bottleneck_channels,
                    expansion=expansion,
                    activation=get_activation_from_type(activation),
                )
            )
            hidden_channels = hidden_channels + block_num_layers * expansion  # Overall output of the dense block
            if block_idx < len(num_layers) - 1:  # Don't apply transition layer on last block
                blocks.append(
                    TransitionLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels // 2,
                        activation=get_activation_from_type(activation),
                    )
                )
                hidden_channels = hidden_channels // 2

        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),  # The features have not passed a non-linearity until here.
            get_activation_from_type(activation),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),
        )
        self._init_params()

    def _init_params(self) -> None:
        """Inits parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`DenseNet`.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of shape (N, `num_classes`).
        """
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
