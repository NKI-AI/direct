# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.jointicnet.jointicnet import JointICNet


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
    "num_iter",
    [2, 4],
)
@pytest.mark.parametrize(
    "use_norm_unet",
    [True, False],
)
def test_jointicnet(shape, num_iter, use_norm_unet):
    model = JointICNet(
        fft2,
        ifft2,
        num_iter,
        use_norm_unet,
        image_unet_num_pool_layers=2,
        kspace_unet_num_pool_layers=2,
        sens_unet_num_pool_layers=2,
    ).cpu()

    kspace = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    sens = create_input(shape + [2]).cpu()

    out = model(kspace, mask, sens)

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
