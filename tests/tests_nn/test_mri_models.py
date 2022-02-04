# coding=utf-8
# Copyright (c) DIRECT Contributors

import functools
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from direct.config.defaults import (
    CheckpointerConfig,
    DefaultConfig,
    FunctionConfig,
    InferenceConfig,
    LossConfig,
    TrainingConfig,
    ValidationConfig,
)
from direct.data.transforms import fft2, ifft2
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine


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


def create_dataset(num_samples, shape, text_description="training"):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, shape, text_description):
            self.num_samples = num_samples
            self.shape = shape
            self.ndim = 2
            self.volume_indices = {}
            current_slice_number = 0
            for idx in range(num_samples):
                filename = pathlib.PosixPath(f"file_{idx}")
                self.volume_indices[filename] = range(current_slice_number, current_slice_number + shape[0])
                current_slice_number += shape[0]
            self.text_description = text_description + str(np.random.randint(0, 1000))

        def __len__(self):
            return self.num_samples * self.shape[0]

        def __getitem__(self, idx):
            sample = {}
            filename = f"file_{idx // self.shape[0]}"
            slice_no = idx % shape[0]

            seed = tuple(map(ord, str(filename + str(slice_no))))
            np.random.seed(seed)

            return create_sample(self.shape[1:], filename=filename, slice_no=slice_no)

    return Dataset(num_samples, shape, text_description)


def create_eninge():
    class TestEngine(MRIModelEngine):
        def __init__(
            self,
            cfg,
            model,
            device,
            forward_operator,
            backward_operator,
            mixed_precision=False,
            **models,
        ):
            super().__init__(
                cfg,
                model,
                device,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                mixed_precision=mixed_precision,
                **models,
            )

        def _do_iteration(self, data, loss_fns=None, regularizer_fns=None):
            output_image = self.model(data["masked_kspace"].sum(self._coil_dim).permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1
            )
            output_image = output_image.sum(self._complex_dim)

            return DoIterationOutput(
                output_image=output_image,
                sensitivity_map=data["sensitivity_map"],
                data_dict={},
            )

    return TestEngine


@pytest.mark.parametrize(
    "shape",
    [(5, 3, 10, 16, 2)],
)
@pytest.mark.parametrize(
    "dataset_num_samples",
    [3, 9],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "ssim_loss", "l2_loss"]],
)
@pytest.mark.parametrize(
    "train_iters, val_iters, checkpointer_iters",
    [[20, 10, 10]],
)
def test_mri_model_engine(shape, loss_fns, dataset_num_samples, train_iters, val_iters, checkpointer_iters):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Models
    model = torch.nn.Conv2d(2, 2, kernel_size=1)
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)

    # Configs
    checkpointer_config = CheckpointerConfig(checkpoint_steps=checkpointer_iters)
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(
        loss=loss_config, checkpointer=checkpointer_config, num_iterations=train_iters, validation_steps=val_iters
    )
    validation_config = ValidationConfig(crop=None)
    inference_config = InferenceConfig(batch_size=shape[0] // 2)
    config = DefaultConfig(training=training_config, validation=validation_config, inference=inference_config)

    # Define engine
    engine = create_eninge()(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
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

    # Test predict function.
    # We have to mock a dataset here.
    dataset = create_dataset(dataset_num_samples, shape)

    with tempfile.TemporaryDirectory() as tempdir:
        out = engine.predict(dataset, pathlib.Path(tempdir))
        assert len(out) == dataset_num_samples
        for data in out:
            assert data[0].shape == (shape[0], 1) + shape[2:-1]

    batch_sampler = engine.build_batch_sampler(
        dataset,
        batch_size=config.inference.batch_size,
        sampler_type="sequential",
        limit_number_of_volumes=None,
    )
    data_loader = engine.build_loader(
        dataset,
        batch_sampler=batch_sampler,
    )
    _, _, visualize_imgs, _ = engine.evaluate(data_loader, loss_fns)
    assert (len(visualize_imgs)) == min(dataset_num_samples, config.logging.tensorboard.num_images)

    # Test train method.
    optimizer = torch.optim.Adam(model.parameters())

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    with tempfile.TemporaryDirectory() as tempdir:
        engine.train(
            optimizer,
            lr_scheduler,
            [create_dataset(dataset_num_samples, shape), create_dataset(dataset_num_samples, shape)],
            pathlib.Path(tempdir),
            validation_datasets=[create_dataset(dataset_num_samples, shape)],
        )
