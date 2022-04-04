# coding=utf-8
# Copyright (c) DIRECT Contributors

import functools

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.rim.config import RIMConfig
from direct.nn.rim.rim import RIM
from direct.nn.rim.rim_engine import RIMEngine


def create_sample(shape, **kwargs):

    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sampling_mask"] = torch.from_numpy(np.random.randn(1, shape[1], shape[2], 1)).float()
    sample["target"] = torch.from_numpy(np.random.randn(shape[1], shape[2])).float()
    sample["scaling_factor"] = torch.tensor(shape[0])

    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


@pytest.mark.parametrize(
    "shape",
    [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "ssim_loss"]],
)
@pytest.mark.parametrize(
    "length, depth",
    [[3, 2]],
)
@pytest.mark.parametrize(
    "scale_log",
    [None, 0.2],
)
def test_lpd_engine(shape, loss_fns, length, depth, scale_log):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    model = RIM(
        forward_operator, backward_operator, hidden_channels=4, length=length, depth=depth, no_parameter_sharing=False
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    model_config = RIMConfig(scale_loglikelihood=scale_log)
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    # Define engine
    engine = RIMEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).float(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        scaling_factor=torch.ones(1),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image.shape == (shape[0],) + (2,) + tuple(shape[2:-1])
