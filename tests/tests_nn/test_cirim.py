# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.cirim.cirim import CIRIM


def create_input(shape):
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 16, 16],
        [2, 5, 16, 32],
    ],
)
@pytest.mark.parametrize(
    "depth",
    [2, 4],
)
@pytest.mark.parametrize(
    "time_steps",
    [8, 16],
)
@pytest.mark.parametrize(
    "recurrent_hidden_channels",
    [64, 128],
)
@pytest.mark.parametrize(
    "num_cascades",
    [1, 2, 8],
)
@pytest.mark.parametrize(
    "no_parameter_sharing",
    [True, False],
)
def test_cirim(shape, depth, time_steps, recurrent_hidden_channels, num_cascades, no_parameter_sharing):
    model = CIRIM(
        fft2,
        ifft2,
        depth=depth,
        time_steps=time_steps,
        recurrent_hidden_channels=recurrent_hidden_channels,
        num_cascades=num_cascades,
        no_parameter_sharing=no_parameter_sharing,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()

    out = next(model(kspace, mask, sens))[-1][-1]  # prediction of the last time step of the last cascade

    assert out.shape == (shape[0], shape[2], shape[3])
