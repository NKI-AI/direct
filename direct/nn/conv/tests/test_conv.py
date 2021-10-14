# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch
import torch.nn as nn

from direct.nn.conv.conv import Conv2d


def create_input(shape):

    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 16, 16],
    ],
)
@pytest.mark.parametrize(
    "out_channels",
    [3, 5],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [16, 8],
)
@pytest.mark.parametrize(
    "n_convs",
    [2, 4],
)
@pytest.mark.parametrize(
    "act",
    [nn.ReLU(), nn.PReLU()],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
def test_conv(shape, out_channels, hidden_channels, n_convs, act, batchnorm):
    model = Conv2d(shape[1], out_channels, hidden_channels, n_convs, act, batchnorm)

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == [shape[0]] + [out_channels] + shape[2:]
