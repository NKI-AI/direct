# coding=utf-8
# Copyright (c) DIRECT Contributors

import math
from enum import Enum
from typing import List, Optional, Tuple

import torch
from torch import nn

from direct.types import DirectEnum

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


RESNET18_NUM_LAYERS = [2, 2, 2, 2]
RESNET34_NUM_LAYERS = [3, 4, 6, 3]
RESNET50_NUM_LAYERS = [3, 4, 6, 3]
RESNET101_NUM_LAYERS = [3, 4, 23, 3]
RESNET152_NUM_LAYERS = [3, 8, 36, 3]


class BasicBlock(nn.Module):
    EXPANSION = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        downsample_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample_net = downsample_net
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample_net is not None:
            residual = self.downsample_net(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: Tuple[int, int] = (1, 1),
        downsample_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample_net = downsample_net
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_net is not None:
            residual = self.downsample_net(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNetBlock(DirectEnum):
    basic = "basic-block"
    bottleneck = "bottleneck"


class ResNetNumLayers(List[int], Enum):
    RESNET18 = [2, 2, 2, 2]
    RESNET34 = [3, 4, 6, 3]
    RESNET50 = [3, 4, 6, 3]
    RESNET101 = [3, 4, 23, 3]
    RESNET152 = [3, 8, 36, 3]


class ResNet(nn.Module):
    """Deep Residual Learning for Image Recognition as in [1]_.

    References
    ----------
    .. [1] He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” ArXiv.org, 10 Dec. 2015,
        https://arxiv.org/abs/1512.03385.
    """

    MIN_SPATIAL_DIM = 200
    IN_CHANNELS = 64

    def __init__(self, block: ResNetBlock, in_channels: int, layers: ResNetNumLayers, num_classes: int = 100):
        """Inits :class:`ResNet`.

        Parameters
        ----------
        block : ResNetBlock
            Block can be Bottleneck or Basic block.
        in_channels : int
            Input channels.
        layers : ResNetNumLayers
            Number of layers in each stack.
        num_classes : int
            Number of output features/classes.
        """
        super().__init__()
        self.block = BasicBlock if block == "basic-block" else Bottleneck
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=(2, 2))
        self.stack3 = self.make_stack(256, layers[2], stride=(2, 2))
        self.stack4 = self.make_stack(512, layers[3], stride=(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.EXPANSION, num_classes)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def make_stack(self, in_channels, blocks, stride=(1, 1)):
        downsample_net = None
        layers = []

        if stride != (1, 1) or self.IN_CHANNELS != in_channels * self.block.EXPANSION:
            downsample_net = nn.Sequential(
                nn.Conv2d(
                    self.IN_CHANNELS, in_channels * self.block.EXPANSION, kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(in_channels * self.block.EXPANSION),
            )

        layers.append(self.block(self.IN_CHANNELS, in_channels, stride, downsample_net))
        self.IN_CHANNELS = in_channels * self.block.EXPANSION
        for i in range(1, blocks):
            layers.append(self.block(self.IN_CHANNELS, in_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Pad input to minimum shape of (N, C, 200, 200)
        _, _, h, w = x.shape
        padding_left, padding_top, = (
            max(0, self.MIN_SPATIAL_DIM - w) // 2,
            max(0, self.MIN_SPATIAL_DIM - h) // 2,
        )
        padding_right = padding_left + w % 2
        padding_bottom = padding_top + h % 2
        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet18(ResNet):
    """Residual network with 18 layers."""

    def __init__(self, in_channels, num_classes=1000):
        """Inits :class:`ResNet18`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes.
        """
        super().__init__(ResNetBlock.basic, in_channels, ResNetNumLayers.RESNET18, num_classes)


class ResNet34(ResNet):
    """Residual network with 34 layers."""

    def __init__(self, in_channels, num_classes=1000):
        """Inits :class:`ResNet34`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes.
        """
        super().__init__(ResNetBlock.basic, in_channels, ResNetNumLayers.RESNET34, num_classes)


class ResNet50(ResNet):
    """Residual network with 50 layers."""

    def __init__(self, in_channels, num_classes=1000):
        """Inits :class:`ResNet50`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes.
        """
        super().__init__(ResNetBlock.bottleneck, in_channels, ResNetNumLayers.RESNET50, num_classes)


class ResNet101(ResNet):
    """Residual network with 101 layers."""

    def __init__(self, in_channels, num_classes=1000):
        """Inits :class:`ResNet101`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes.
        """
        super().__init__(ResNetBlock.bottleneck, in_channels, ResNetNumLayers.RESNET101, num_classes)


class ResNet152(ResNet):
    """Residual network with 152 layers."""

    def __init__(self, in_channels, num_classes=1000):
        """Inits :class:`ResNet152`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes.
        """
        super().__init__(ResNetBlock.bottleneck, in_channels, ResNetNumLayers.RESNET152, num_classes)
