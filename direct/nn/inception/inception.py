# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Dict

import torch
from torch import nn

from direct.nn.types import ActivationType


def get_activation_from_type(act_type: ActivationType) -> nn.Module:
    """Returns activation (nn.Module) from input.

    Parameters
    ----------
    act_type : ActivationType
        Activation type.

    Returns
    -------
    nn.Module
    """
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "leaky_relu":
        return nn.LeakyReLU()
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "prelu":
        return nn.PReLU()
    return nn.Tanh()


class InceptionBlock(nn.Module):
    """Main block for Inception model as presented in [1]_.

    References
    ----------
    .. [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich,
        A. (2014). Going deeper with convolutions. ArXiv [Cs.CV]. https://doi.org/10.48550/ARXIV.1409.4842
    """

    def __init__(
        self,
        in_channels: int,
        reduction_channels: Dict[str, int],
        out_channels: Dict[str, int],
        activation: ActivationType,
    ):
        """Inits :class:`InceptionBlock`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        reduction_channels : dict
            Dictionary containing values for the '3x3' and '5x5' reduction convolution blocks.
        out_channels : dict
            Dictionary containing values for the '1x1', '3x3', '5x5' output convolution and max pooling ('max') blocks.
        activation : ActivationType
            Activation type.
        """
        super().__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels["1x1"], kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels["1x1"]),
            get_activation_from_type(activation),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels["3x3"], kernel_size=(1, 1)),
            nn.BatchNorm2d(reduction_channels["3x3"]),
            get_activation_from_type(activation),
            nn.Conv2d(reduction_channels["3x3"], out_channels["3x3"], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels["3x3"]),
            get_activation_from_type(activation),
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels["5x5"], kernel_size=(1, 1)),
            nn.BatchNorm2d(reduction_channels["5x5"]),
            get_activation_from_type(activation),
            nn.Conv2d(reduction_channels["5x5"], out_channels["5x5"], kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(out_channels["5x5"]),
            get_activation_from_type(activation),
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels, out_channels["max"], kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels["max"]),
            get_activation_from_type(activation),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`InceptionBlock`.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        return torch.cat([self.conv_1x1(inp), self.conv_3x3(inp), self.conv_5x5(inp), self.pool(inp)], dim=1)


class Inception(nn.Module):
    """Inception or GoogleNet as presented in [1]_.

    References
    ----------
    .. [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich,
        A. (2014). Going deeper with convolutions. ArXiv [Cs.CV]. https://doi.org/10.48550/ARXIV.1409.4842
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_features: int,
        activation: ActivationType = ActivationType.relu,
    ):
        """Inits :class:`Inception`.

        Parameters
        ----------
        in_channels : int
            Size of input channels.
        hidden_channels : int
            First convolution output channels.
        out_features : int
            Number of output channels/classes.
        activation : ActivationType
            Activation type. Default: "relu".
        """
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_channels),
            get_activation_from_type(activation),
        )
        self.inception = nn.Sequential(
            InceptionBlock(
                hidden_channels, {"3x3": 32, "5x5": 16}, {"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, activation
            ),
            InceptionBlock(64, {"3x3": 32, "5x5": 16}, {"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, activation),
            nn.MaxPool2d(3, padding=(1, 1), stride=(2, 2)),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, activation),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, activation),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, activation),
            InceptionBlock(96, {"3x3": 32, "5x5": 16}, {"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, activation),
            nn.MaxPool2d(3, padding=(1, 1), stride=(2, 2)),
            InceptionBlock(128, {"3x3": 48, "5x5": 16}, {"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, activation),
            InceptionBlock(128, {"3x3": 48, "5x5": 16}, {"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, activation),
        )
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, out_features))

        self._init_params(activation)

    def _init_params(self, activation: ActivationType) -> None:
        """Inits parameters.

        Parameters
        ----------
        activation : ActivationType
            Activation name.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity=activation)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`Inception`.

        Parameters
        ----------
        inp : torch.Tensor
            Input of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of shape (N, `out_features`).
        """
        return self.out(self.inception(self.conv_in(inp)))
