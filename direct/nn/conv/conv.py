# coding=utf-8
# Copyright (c) DIRECT Contributors

import torch.nn as nn


class Conv2d(nn.Module):
    """
    Simple 2D convolutional model.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, n_convs=3, activation=nn.PReLU(), batchnorm=False):
        super(Conv2d, self).__init__()

        self.conv = []

        for i in range(n_convs):

            self.conv.append(
                nn.Conv2d(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels if i != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if batchnorm:
                self.conv.append(nn.BatchNorm2d(hidden_channels, eps=1e-4))
            if i != n_convs - 1:
                self.conv.append(activation)

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        x = self.conv(x)

        return x
