# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.varsplitnet.varsplitnet import MRIVarSplitNet


def create_input(shape):

    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [[4, 3, 32, 32], [4, 5, 40, 20]],
)
@pytest.mark.parametrize(
    "num_steps_reg",
    [2, 3],
)
@pytest.mark.parametrize(
    "num_steps_dc",
    [1, 4],
)
@pytest.mark.parametrize(
    "image_init",
    ["sense"],
)
@pytest.mark.parametrize(
    "no_parameter_sharing",
    [True, False],
)
@pytest.mark.parametrize(
    "image_model_architecture",
    ["unet", "normunet", "didn", "resnet"],
)
@pytest.mark.parametrize(
    "kspace_no_parameter_sharing",
    [True, False],
)
@pytest.mark.parametrize(
    "kspace_model_architecture",
    ["conv", None],
)
def test_varsplitnet(
    shape,
    num_steps_reg,
    num_steps_dc,
    image_init,
    no_parameter_sharing,
    image_model_architecture,
    kspace_no_parameter_sharing,
    kspace_model_architecture,
):
    model = MRIVarSplitNet(
        fft2,
        ifft2,
        num_steps_reg,
        num_steps_dc,
        image_init,
        no_parameter_sharing,
        image_model_architecture,
        kspace_no_parameter_sharing,
        kspace_model_architecture,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()

    out = model(kspace, sens, mask)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
