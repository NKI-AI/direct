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
from direct.nn.multidomainnet.multidomainnet import MultiDomainNet
from direct.nn.multidomainnet.multidomainnet_engine import MultiDomainNetEngine


def create_sample(shape, **kwargs):
    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sampling_mask"] = torch.from_numpy(np.random.randn(1, shape[1], shape[2], 1)).float()
    sample["target"] = torch.from_numpy(np.random.randn(shape[1], shape[2])).float()
    sample["scaling_factor"] = torch.tensor([1.0])
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


@pytest.mark.parametrize("shape", [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)])
@pytest.mark.parametrize("loss_fns", [["l1_loss", "ssim_loss", "l2_loss"]])
@pytest.mark.parametrize("standardization", [True])
@pytest.mark.parametrize(
    "num_filters",
    [4, 8],  # powers of 2
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [2],
)
@pytest.mark.parametrize(
    "dataset_num_samples",
    [3, 9],
)
def test_multidomainnet_engine(shape, loss_fns, standardization, num_filters, num_pool_layers, dataset_num_samples):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    kwargs = {
        "forward_operator": fft2,
        "backward_operator": ifft2,
        "standardization": standardization,
        "num_filters": num_filters,
        "num_pool_layers": num_pool_layers,
    }
    model = MultiDomainNet(**kwargs)
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)

    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    inference_config = InferenceConfig(batch_size=shape[0] // 2)
    config = DefaultConfig(training=training_config, validation=validation_config, inference=inference_config)
    # Define engine
    engine = MultiDomainNetEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2
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
