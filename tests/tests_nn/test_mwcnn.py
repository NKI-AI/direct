# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch
import torch.nn as nn

from direct.nn.mwcnn.mwcnn import MWCNN


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 20, 34],
    ],
)
@pytest.mark.parametrize(
    "first_conv_hidden_channels",
    [4, 8],
)
@pytest.mark.parametrize(
    "n_scales",
    [2, 3],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
@pytest.mark.parametrize(
    "act",
    [nn.ReLU(), nn.PReLU()],
)
def test_mwcnn(shape, first_conv_hidden_channels, n_scales, bias, batchnorm, act):
    model = MWCNN(shape[1], first_conv_hidden_channels, n_scales, bias, batchnorm, act)

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == shape
