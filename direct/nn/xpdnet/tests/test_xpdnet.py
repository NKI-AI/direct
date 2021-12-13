# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.xpdnet.xpdnet import XPDNet


def create_input(shape):

    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "num_iter",
    [2, 3],
)
@pytest.mark.parametrize(
    "num_primal",
    [2, 3],
)
@pytest.mark.parametrize(
    "image_model_architecture",
    ["MWCNN"],
)
@pytest.mark.parametrize(
    "primal_only, kspace_model_architecture, num_dual",
    [
        [True, None, 1],
        [False, "CONV", 3],
        [False, "DIDN", 2],
    ],
)
@pytest.mark.parametrize(
    "normalize",
    [True, False],
)
def test_xpdnet(
    shape,
    num_iter,
    num_primal,
    num_dual,
    image_model_architecture,
    kspace_model_architecture,
    primal_only,
    normalize,
):
    model = XPDNet(
        fft2,
        ifft2,
        num_iter=num_iter,
        num_primal=num_primal,
        num_dual=num_dual,
        image_model_architecture=image_model_architecture,
        kspace_model_architecture=kspace_model_architecture,
        use_primal_only=primal_only,
        normalize=normalize,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()

    out = model(kspace, mask, sens)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
