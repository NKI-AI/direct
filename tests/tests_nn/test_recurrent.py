# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.nn.recurrent.recurrent import Conv2dGRU


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
    "hidden_channels",
    [4, 8],
)
def test_conv2dgru(shape, hidden_channels):
    model = Conv2dGRU(shape[1], hidden_channels, shape[1])
    data = create_input(shape).cpu()

    out = model(data, None)[0]

    assert list(out.shape) == shape
