# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.varnet.varnet import EndToEndVarNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [[4, 3, 32, 32], [4, 5, 40, 20]],
)
@pytest.mark.parametrize(
    "num_layers",
    [2, 3, 6],
)
@pytest.mark.parametrize(
    "num_filters",
    [2, 4],
)
@pytest.mark.parametrize(
    "num_pull_layers",
    [2, 4],
)
def test_varnet(shape, num_layers, num_filters, num_pull_layers):
    model = EndToEndVarNet(fft2, ifft2, num_layers, num_filters, num_pull_layers, in_channels=2).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()

    out = model(kspace, mask, sens)

    assert list(out.shape) == shape + [2]
