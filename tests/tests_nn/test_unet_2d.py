# coding=utf-8
# Copyright (c) DIRECT Contributors


import numpy as np
import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.unet.unet_2d import NormUnetModel2d, Unet2d

try:
    from direct.nn.openvino.openvino_model import OpenVINOModel

    openvino_available = True
except Exception:
    openvino_available = False


def create_input(shape):
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3, 16, 16],
        [4, 5, 16, 32],
        [3, 4, 32, 32],
        [3, 4, 40, 20],
    ],
)
@pytest.mark.parametrize(
    "num_filters",
    [4, 6, 8],
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [2, 3],
)
@pytest.mark.parametrize(
    "skip",
    [True, False],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_unet_2d(shape, num_filters, num_pool_layers, skip, normalized):
    model = Unet2d(
        fft2,
        ifft2,
        num_filters=num_filters,
        num_pool_layers=num_pool_layers,
        skip_connection=skip,
        normalized=normalized,
        dropout_probability=0.05,
    ).cpu()

    data = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()

    model.eval()

    out = model(data, sens)

    if openvino_available:
        ov_model = OpenVINOModel(model)
        ov_out = ov_model(data, sens)

        assert torch.max(torch.abs(out[0][-1] - ov_out[0][-1])) < 1e-5

    assert list(out.shape) == [shape[0]] + shape[2:] + [2]
