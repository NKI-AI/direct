# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.lpd.lpd import LPDNet


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
    "num_dual",
    [3],
)
@pytest.mark.parametrize(
    "primal_model_architecture",
    ["MWCNN", "UNET", "NORMUNET", None],
)
@pytest.mark.parametrize(
    "dual_model_architecture",
    ["CONV", "DIDN", "UNET", "NORMUNET", None],
)
def test_lpd(
    shape,
    num_iter,
    num_primal,
    num_dual,
    primal_model_architecture,
    dual_model_architecture,
):
    kwargs = {
        "forward_operator": fft2,
        "backward_operator": ifft2,
        "num_iter": num_iter,
        "num_primal": num_primal,
        "num_dual": num_dual,
        "primal_model_architecture": primal_model_architecture,
        "dual_model_architecture": dual_model_architecture,
    }
    if primal_model_architecture is None or dual_model_architecture is None:
        with pytest.raises(NotImplementedError):
            model = LPDNet(**kwargs).cpu()
    else:
        model = LPDNet(**kwargs).cpu()

        kspace = create_input(shape + [2]).cpu()
        sens = create_input(shape + [2]).cpu()
        mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()

        out = model(kspace, sens, mask)

        assert list(out.shape) == [shape[0]] + shape[2:] + [2]
