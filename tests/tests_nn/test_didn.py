# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.nn.didn.didn import DIDN


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
    "n_dubs",
    [3, 4],
)
@pytest.mark.parametrize(
    "num_convs_recon",
    [3, 4],
)
@pytest.mark.parametrize(
    "skip",
    [True, False],
)
def test_didn(shape, out_channels, hidden_channels, n_dubs, num_convs_recon, skip):
    model = DIDN(shape[1], out_channels, hidden_channels, n_dubs, num_convs_recon, skip)

    data = create_input(shape).cpu()

    out = model(data)

    assert list(out.shape) == [shape[0]] + [out_channels] + shape[2:]
