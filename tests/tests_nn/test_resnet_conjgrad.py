# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.resnet.conj import CGUpdateType
from direct.nn.resnet.resnet import ResNetConjGrad


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [[3, 3, 32, 32]],
)
@pytest.mark.parametrize("nums_steps", [3])
@pytest.mark.parametrize(
    "resnet_hidden_channels, resnet_num_blocks, resnet_batchnorm, resnet_scale", [[8, 4, True, None]]
)
@pytest.mark.parametrize(
    "cg_param_update_type", [CGUpdateType.FR, CGUpdateType.PRP, CGUpdateType.DY, CGUpdateType.BAN]
)
@pytest.mark.parametrize("image_init", ["sense", "zero_filled", "invalid"])
@pytest.mark.parametrize("no_parameter_sharing", [True, False])
@pytest.mark.parametrize("cg_iters", [5, 10])
def test_resnet_conjgrad(
    shape,
    nums_steps,
    resnet_hidden_channels,
    resnet_num_blocks,
    resnet_batchnorm,
    resnet_scale,
    cg_param_update_type,
    image_init,
    no_parameter_sharing,
    cg_iters,
):
    kwargs = {
        "forward_operator": fft2,
        "backward_operator": ifft2,
        "num_steps": nums_steps,
        "resnet_batchnorm": resnet_batchnorm,
        "resnet_scale": resnet_scale,
        "resnet_num_blocks": resnet_num_blocks,
        "resnet_hidden_channels": resnet_hidden_channels,
        "image_init": image_init,
        "no_parameter_sharing": no_parameter_sharing,
        "cg_iters": cg_iters,
    }

    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()
    if image_init == "invalid":
        with pytest.raises(ValueError):
            model = ResNetConjGrad(**kwargs)
    else:
        model = ResNetConjGrad(**kwargs)
        out = model(kspace, sens, mask)
        assert list(out.shape) == [shape[0]] + shape[2:] + [2]
