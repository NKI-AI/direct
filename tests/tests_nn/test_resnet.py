# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.nn.resnet.resnet import ResNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 16, 16],
        [3, 2, 15, 17],
    ],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [4, 8],
)
@pytest.mark.parametrize(
    "out_channels",
    [2, None],
)
@pytest.mark.parametrize(
    "num_blocks",
    [3, 5],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
@pytest.mark.parametrize(
    "scale",
    [None, 0.1],
)
def test_resnet(shape, hidden_channels, out_channels, num_blocks, batchnorm, scale):
    model = ResNet(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        batchnorm=batchnorm,
        scale=scale,
    )
    data = create_input(shape).cpu()
    out = model(data)
    assert list(out.shape) == shape
