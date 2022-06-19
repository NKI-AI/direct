# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.rim.rim import RIM

try:
    from direct.nn.openvino.openvino_model import OpenVINOModel

    openvino_available = True
except Exception:
    openvino_available = False


def create_input(shape):

    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3, 11, 12],
    ],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [4],
)
@pytest.mark.parametrize(
    "length",
    [3],
)
@pytest.mark.parametrize(
    "depth",
    [2],
)
@pytest.mark.parametrize(
    "no_parameter_sharing",
    [True, False],
)
@pytest.mark.parametrize(
    "instance_norm",
    [True, False],
)
@pytest.mark.parametrize(
    "dense_connect",
    [True, False],
)
@pytest.mark.parametrize(
    "skip_connections",
    [True, False],
)
@pytest.mark.parametrize(
    "image_init",
    ["zero_filled", "sense", "input_kspace", "input_image", None],
)
@pytest.mark.parametrize(
    "learned_initializer",
    [True, False],
)
@pytest.mark.parametrize(
    "input_image_is_None",
    [True, False],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_rim(
    shape,
    hidden_channels,
    length,
    depth,
    no_parameter_sharing,
    instance_norm,
    dense_connect,
    skip_connections,
    image_init,
    learned_initializer,
    input_image_is_None,
    normalized,
):
    model = RIM(
        fft2,
        ifft2,
        hidden_channels=hidden_channels,
        length=length,
        depth=depth,
        no_parameter_sharing=no_parameter_sharing,
        instance_norm=instance_norm,
        dense_connect=dense_connect,
        skip_connections=skip_connections,
        image_initialization=image_init,
        learned_initializer=learned_initializer,
        normalized=normalized,
    ).cpu()

    inputs = {
        "input_image": create_input([shape[0]] + shape[2:] + [2]).cpu() if not input_image_is_None else None,
        "masked_kspace": create_input(shape + [2]).cpu(),
        "sensitivity_map": create_input(shape + [2]).cpu(),
        "sampling_mask": create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu(),
    }
    if input_image_is_None:
        if image_init == "input_image":
            inputs["initial_image"] = create_input([shape[0]] + shape[2:] + [2]).cpu()
        elif image_init == "input_kspace":
            inputs["initial_kspace"] = create_input(shape + [2]).cpu()
    if image_init is None and input_image_is_None:
        with pytest.raises(ValueError):
            out = model(**inputs)[0][-1]
    else:
        out = model(**inputs)
        assert list(out[0][-1].shape) == [shape[0]] + [2] + shape[2:]

    if openvino_available and not input_image_is_None and skip_connections:
        ov_model = OpenVINOModel(model)
        ov_out = ov_model(**inputs)

        assert torch.max(torch.abs(out[0][-1] - ov_out[0][-1])) < 1e-5
        assert torch.max(torch.abs(out[1] - ov_out[1])) < 1e-4
