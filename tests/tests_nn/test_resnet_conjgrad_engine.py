# coding=utf-8
# Copyright (c) DIRECT Contributors

import functools

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.resnet.conj import CGUpdateType
from direct.nn.resnet.resnet import ResNetConjGrad
from direct.nn.resnet.resnetconj_engine import ResNetConjGradEngine


def create_sample(shape, **kwargs):
    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["scaling_factor"] = torch.tensor([1.0])
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    for k, v in sample.items():
        sample[k] = v.cuda()
    return sample


@pytest.mark.parametrize("shape", [(4, 3, 10, 16, 2)])
@pytest.mark.parametrize("loss_fns", [["l1_loss"]])
@pytest.mark.parametrize("nums_steps", [2])
@pytest.mark.parametrize(
    "resnet_hidden_channels, resnet_num_blocks, resnet_batchnorm, resnet_scale", [[8, 4, True, 1.0]]
)
@pytest.mark.parametrize("cg_param_update_type", [CGUpdateType.FR])
def test_resnetconjgrad_engine(
    shape,
    loss_fns,
    nums_steps,
    resnet_hidden_channels,
    resnet_num_blocks,
    resnet_batchnorm,
    resnet_scale,
    cg_param_update_type,
):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    kwargs = {
        "forward_operator": forward_operator,
        "backward_operator": backward_operator,
        "num_steps": nums_steps,
        "resnet_batchnorm": resnet_batchnorm,
        "resnet_scale": resnet_scale,
        "resnet_num_blocks": resnet_num_blocks,
        "resnet_hidden_channels": resnet_hidden_channels,
    }
    model = ResNetConjGrad(**kwargs).cuda()
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1).cuda()
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    config = DefaultConfig(training=training_config, validation=validation_config)
    # Define engine
    engine = ResNetConjGradEngine(config, model, "cuda", fft2, ifft2, sensitivity_model=sensitivity_model)
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).float(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image.shape == (shape[0],) + tuple(shape[2:-1])
