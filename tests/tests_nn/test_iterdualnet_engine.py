# coding=utf-8
# Copyright (c) DIRECT Contributors

import functools

import numpy as np
import pytest
import torch

from direct.config.defaults import (
    DefaultConfig,
    FunctionConfig,
    InferenceConfig,
    LossConfig,
    TrainingConfig,
    ValidationConfig,
)
from direct.data.transforms import fft2, ifft2
from direct.nn.iterdualnet.iterdualnet import IterDualNet
from direct.nn.iterdualnet.iterdualnet_engine import IterDualNetEngine


def create_sample(shape, **kwargs):

    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


@pytest.mark.parametrize("shape", [(5, 3, 10, 16, 2)])
@pytest.mark.parametrize("loss_fns", [["l1_loss", "ssim_loss", "l2_loss"]])
@pytest.mark.parametrize("num_iter", [3])
@pytest.mark.parametrize("compute_per_coil", [True])
def test_iterdualnet_engine(shape, loss_fns, num_iter, compute_per_coil):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)

    # Models
    model = IterDualNet(
        forward_operator,
        backward_operator,
        num_iter=5,
        compute_per_coil=compute_per_coil,
        image_unet_num_filters=4,
        image_unet_num_pool_layers=3,
        kspace_unet_num_filters=4,
        kspace_unet_num_pool_layers=3,
    ).cpu()
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)

    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    inference_config = InferenceConfig(batch_size=shape[0] // 2)
    config = DefaultConfig(training=training_config, validation=validation_config, inference=inference_config)
    # Define engine
    engine = IterDualNetEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)

    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).float(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    assert out.output_image.shape == (shape[0],) + tuple(shape[2:-1])
