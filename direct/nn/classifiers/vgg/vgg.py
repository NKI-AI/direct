# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import List, Union

import torch
from torch import nn

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]


VGG_CFGS = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(in_channels: int, vgg_cfg: List[Union[str, int]], batchnorm: bool = False) -> nn.Sequential:
    """Created VGG layers from configuration.

    Parameters
    ----------
    in_channels : int
        Input channels.
    vgg_cfg : list of str or ints
        List of number of channels. Can also be "M" for max pooling. The length of `vgg_cfg` denotes the number of
        layers that will be created.
    batchnorm : bool
        If True batch normalization is applied after each convolution.

    Returns
    -------

    """
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    for channels in vgg_cfg:
        if channels == "M":
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        else:
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            if batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(True))
            in_channels = channels

    return layers


class VGG(nn.Module):
    """VGG implementation as in [1]_.

    Code adapted from [2]_ with Apache License, Version 2.0.

    References
    ----------
    .. [1] Simonyan, Karen, and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition.
        arXiv, 10 Apr. 2015. arXiv.org, https://doi.org/10.48550/arXiv.1409.1556.
    .. [2] https://github.com/Lornatang/VGG-PyTorch/blob/main/model.py
    """

    def __init__(
        self, in_channels, vgg_cfg: List[Union[str, int]], num_classes: int = 1000, batchnorm: bool = False
    ) -> None:
        """Inits :class:`VGG`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        vgg_cfg : list of str or ints
            List of number of channels. Can also be "M" for max pooling. The length of `vgg_cfg` denotes the number of
            layers that will be created.
        num_classes : int
            Number of output features/classes. Default: 1000.
        batchnorm : bool
            If True batch normalization is applied after each convolution.
        """
        super().__init__()

        self.features = _make_layers(in_channels, vgg_cfg, batchnorm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7**2, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        self._init_params()

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`VGG`.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of shape (N, `num_classes`).
        """
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class VGG11(VGG):
    """VGG with 11 layers."""

    def __init__(self, in_channels, num_classes: int = 1000, batchnorm: bool = False) -> None:
        """Inits :class:`VGG11`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes. Default: 1000.
        batchnorm : bool
            If True batch normalization is applied after each convolution.
        """
        super().__init__(
            in_channels=in_channels, vgg_cfg=VGG_CFGS["VGG11"], num_classes=num_classes, batchnorm=batchnorm
        )


class VGG13(VGG):
    """VGG with 13 layers."""

    def __init__(self, in_channels, num_classes: int = 1000, batchnorm: bool = False) -> None:
        """Inits :class:`VGG13`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes. Default: 1000.
        batchnorm : bool
            If True batch normalization is applied after each convolution.
        """
        super().__init__(
            in_channels=in_channels, vgg_cfg=VGG_CFGS["VGG11"], num_classes=num_classes, batchnorm=batchnorm
        )


class VGG16(VGG):
    """VGG with 16 layers."""

    def __init__(self, in_channels, num_classes: int = 1000, batchnorm: bool = False) -> None:
        """Inits :class:`VGG16`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes. Default: 1000.
        batchnorm : bool
            If True batch normalization is applied after each convolution.
        """
        super().__init__(
            in_channels=in_channels, vgg_cfg=VGG_CFGS["VGG11"], num_classes=num_classes, batchnorm=batchnorm
        )


class VGG19(VGG):
    """VGG with 19 layers."""

    def __init__(self, in_channels, num_classes: int = 1000, batchnorm: bool = False) -> None:
        """Inits :class:`VGG19`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        num_classes : int
            Number of output features/classes. Default: 1000.
        batchnorm : bool
            If True batch normalization is applied after each convolution.
        """
        super().__init__(
            in_channels=in_channels, vgg_cfg=VGG_CFGS["VGG11"], num_classes=num_classes, batchnorm=batchnorm
        )
