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

"""direct.nn.resnet.resnet module."""

import torch
from torch import nn


class ResNetBlock(nn.Module):
    """Main block of :class:`ResNet`.

    Consisted of a convolutional layer followed by a relu activation, a second convolution, and finally a scaled
    skip connection with the input.
    """

    def __init__(self, in_channels: int, hidden_channels: int, scale: float | None = 0.1):
        """Inits :class:`ResNetBlock`.

        Args:
            in_channels: Input channels.
            hidden_channels: Hidden channels (output channels of firs conv).
            scale: Float that will scale the output of the convolutions before adding the input. Default is ``0.1``.
        """
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
        """Forward.

        Args:
            x: X.

        Returns:
            The result.
        """
        out = self.conv2(self.relu(self.conv1(x.clone())))
        if self.scale:
            out = self.scale * out
        return x + out


class ResNet(nn.Module):
    """Simple residual network.

    Consisted of a sequence of :class:`ResNetBlocks` followed optionally by batch normalization blocks, followed by
    an output convolution layer.
    """

    def __init__(
        self,
        hidden_channels: int,
        in_channels: int = 2,
        out_channels: int | None = None,
        num_blocks: int = 15,
        batchnorm: bool = True,
        scale: float | None = 0.1,
    ):
        """Inits :class:`ResNet`.

        Args:
            hidden_channels: Hidden dimension.
            in_channels: Input dimension. Default is ``2 (for MRI)``.
            out_channels: Output dimension. If None, will be the same as `in_channels`.
            num_blocks: Number of :class:`ResNetBlocks`. Default is ``15``.
            batchnorm: If True, batch normalization will be performed after each :class:`ResNetBlock`.
            scale: Scale parameter for :class:`ResNetBlock`. Default is ``0.1``.
        """
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        resblocks: list[nn.Module] = []
        for _ in range(num_blocks):
            resblocks.append(ResNetBlock(in_channels=hidden_channels, hidden_channels=hidden_channels, scale=scale))
            if batchnorm:
                resblocks.append(nn.BatchNorm2d(num_features=hidden_channels))

        self.resblocks = nn.Sequential(*resblocks)
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

        Args:
            input_image: Masked k-space of shape (N, in_channels, height, width).

        Returns:
            Output image of shape (N, height, width, complex=2).
        """
        return self.conv_out(self.conv_in(input_image) + self.resblocks(self.conv_in(input_image)))
