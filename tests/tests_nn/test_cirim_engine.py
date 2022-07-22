# coding=utf-8
# Copyright (c) DIRECT Contributors

import functools
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.cirim.cirim import CIRIM
from direct.nn.cirim.cirim_engine import CIRIMEngine


def create_sample(shape, **kwargs):
    sample = {
        "masked_kspace": torch.from_numpy(np.random.randn(*shape)).float(),
        "sensitivity_map": torch.from_numpy(np.random.randn(*shape)).float(),
        "sampling_mask": torch.from_numpy(np.random.randn(1, shape[1], shape[2], 1)).float(),
        "target": torch.from_numpy(np.random.randn(shape[0], shape[1], shape[2])).float(),
        "scaling_factor": torch.tensor([1.0]),
    }
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


def create_dataset(num_samples, shape):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, shape):
            self.num_samples = num_samples
            self.shape = shape
            self.ndim = 2
            self.volume_indices = {}
            current_slice_number = 0
            for idx in range(num_samples):
                self.volume_indices["filename_{idx}"] = range(current_slice_number, current_slice_number + shape[0])
                current_slice_number += shape[0]

        def __len__(self):
            return self.num_samples * self.shape[0]

        def __getitem__(self, idx):
            sample = {}
            filename = f"filename_{idx // self.shape[0]}"
            slice_no = idx % shape[0]

            seed = tuple(map(ord, str(filename + str(slice_no))))
            np.random.seed(seed)

            return create_sample(shape, filename=filename, slice_no=slice_no)

    dataset = Dataset(num_samples, shape[1:])
    return dataset


@pytest.mark.parametrize(
    "shape",
    [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "ssim_loss", "l2_loss"]],
)
@pytest.mark.parametrize(
    "time_steps, num_cascades, recurrent_hidden_channels",
    [[8, 4, 128]],
)
def test_cirim_engine(shape, loss_fns, time_steps, num_cascades, recurrent_hidden_channels):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    model = CIRIM(
        forward_operator,
        backward_operator,
        depth=2,
        time_steps=time_steps,
        recurrent_hidden_channels=recurrent_hidden_channels,
        num_cascades=num_cascades,
        no_parameter_sharing=True,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    config = DefaultConfig(training=training_config, validation=validation_config)
    # Define engine
    engine = CIRIMEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
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
