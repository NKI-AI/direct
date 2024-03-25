# Copyright (c) DIRECT Contributors

"""Tests for the direct.nn.vsharp module."""

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.get_nn_model_config import ModelName
from direct.nn.types import InitType
from direct.nn.vsharp.vsharp import VSharpNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize("shape", [[1, 3, 16, 16]])
@pytest.mark.parametrize("num_steps", [3])
@pytest.mark.parametrize("num_steps_dc_gd", [2])
@pytest.mark.parametrize("image_init", [InitType.SENSE, InitType.ZERO_FILLED])
@pytest.mark.parametrize(
    "image_model_architecture, image_model_kwargs",
    [
        [ModelName.UNET, {"image_unet_num_filters": 4, "image_unet_num_pool_layers": 2}],
        [ModelName.DIDN, {"image_didn_hidden_channels": 4, "image_didn_num_dubs": 2, "image_didn_num_convs_recon": 2}],
    ],
)
@pytest.mark.parametrize(
    "initializer_channels, initializer_dilations",
    [
        [(8, 8, 16), (1, 1, 4)],
    ],
)
@pytest.mark.parametrize("aux_steps", [-1, 1])
def test_varsplitnet(
    shape,
    num_steps,
    num_steps_dc_gd,
    image_init,
    image_model_architecture,
    image_model_kwargs,
    initializer_channels,
    initializer_dilations,
    aux_steps,
):
    model = VSharpNet(
        fft2,
        ifft2,
        num_steps=num_steps,
        num_steps_dc_gd=num_steps_dc_gd,
        image_init=image_init,
        no_parameter_sharing=False,
        initializer_channels=initializer_channels,
        initializer_dilations=initializer_dilations,
        auxiliary_steps=aux_steps,
        image_model_architecture=image_model_architecture,
        **image_model_kwargs,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()
    out = model(kspace, sens, mask)

    for i in range(len(out)):
        assert list(out[i].shape) == [shape[0]] + shape[2:] + [2]
