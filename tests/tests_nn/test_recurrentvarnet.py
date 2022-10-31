# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.recurrentvarnet.recurrentvarnet import RecurrentVarNet
from direct.nn.types import InitType


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 16, 16],
        [2, 5, 16, 32],
    ],
)
@pytest.mark.parametrize(
    "num_steps",
    [3, 5],
)
@pytest.mark.parametrize(
    "recurrent_hidden_channels",
    [4, 8],
)
@pytest.mark.parametrize(
    "recurrent_num_layers",
    [1, 2],
)
@pytest.mark.parametrize(
    "no_parameter_sharing",
    [True, False],
)
@pytest.mark.parametrize(
    "learned_initializer, initializer_initialization, initializer_channels, initializer_dilations",
    [
        [True, InitType.sense, (4, 4, 8, 8), (1, 1, 1, 2)],
        [True, InitType.zero_filled, (2, 4, 2, 4), (1, 2, 1, 3)],
        [False, None, None, None],
    ],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_recurrentvarnet(
    shape,
    num_steps,
    recurrent_hidden_channels,
    recurrent_num_layers,
    no_parameter_sharing,
    learned_initializer,
    initializer_initialization,
    initializer_channels,
    initializer_dilations,
    normalized,
):
    model = RecurrentVarNet(
        fft2,
        ifft2,
        num_steps=num_steps,
        recurrent_hidden_channels=recurrent_hidden_channels,
        recurrent_num_layers=recurrent_num_layers,
        no_parameter_sharing=no_parameter_sharing,
        learned_initializer=learned_initializer,
        initializer_initialization=initializer_initialization,
        initializer_channels=initializer_channels,
        initializer_dilations=initializer_dilations,
        normalized=normalized,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()

    out_kspace = model(kspace, mask, sens)

    assert list(out_kspace.shape) == shape + [2]
