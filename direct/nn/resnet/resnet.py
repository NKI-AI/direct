# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Optional

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, scale: Optional[float] = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x.clone())))
        if self.scale:
            out = self.scale * out
        return x + out


class ResNet(nn.Module):
    """Simple residual network."""

    def __init__(
        self,
        hidden_channels: int,
        in_channels: int = 2,
        out_channels: Optional[int] = None,
        num_blocks: int = 15,
        batchnorm: bool = True,
        scale: Optional[float] = 0.1,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.resblocks = []
        for _ in range(num_blocks):
            self.resblocks.append(
                ResNetBlock(in_channels=hidden_channels, hidden_channels=hidden_channels, scale=scale)
            )
            if batchnorm:
                self.resblocks.append(nn.BatchNorm2d(num_features=hidden_channels))

        self.resblocks = nn.Sequential(*self.resblocks)
        if out_channels is None:
            out_channels = in_channels
        self.conv_out = nn.Sequential(
            *[
                nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            ]
        )

    def forward(
        self,
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`ResNet`.

        Parameters
        ----------
        input_image: torch.Tensor
            Masked k-space of shape (N, in_channels, height, width).

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        return self.conv_out(self.conv_in(input_image) + self.resblocks(self.conv_in(input_image)))
