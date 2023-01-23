import pathlib
import tempfile
from dataclasses import field, make_dataclass

import pytest
from omegaconf import DictConfig, OmegaConf

from direct.config.defaults import (
    CheckpointerConfig,
    DefaultConfig,
    FunctionConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
)
from direct.data.datasets_config import DatasetConfig, MaskingConfig, TransformsConfig
from direct.launch import launch
from direct.train import setup_train


def create_test_cfg(
    train_dataset_shape,
    val_dataset_shape,
    train_batch_size,
    val_batch_size,
    loss_fns,
    train_iters,
    val_iters,
    checkpointer_iters,
    inference_batch_size,
):
    # Configs
    transforms_config = TransformsConfig(
        estimate_sensitivity_maps=True,
        scaling_key="masked_kspace",
        masking=MaskingConfig(name="FastMRIRandom"),
        crop=(64, 64),
    )

    new_class = make_dataclass(
        "",
        fields=[
            ("sample_size", int, field(init=False)),
            ("spatial_shape", list, field(init=False)),
            ("num_coils", int, field(init=False)),
        ],
        bases=(DatasetConfig,),
    )

    train_dataset_config = DatasetConfig(
        name="FakeMRIBlobs", transforms=transforms_config, text_description="training"
    )
    train_dataset_config.__class__ = new_class
    train_dataset_config.sample_size = train_dataset_shape[0]
    train_dataset_config.num_coils = train_dataset_shape[2]
    train_dataset_config.spatial_shape = (train_dataset_shape[1],) + train_dataset_shape[3:]

    val_dataset_config = DatasetConfig(
        name="FakeMRIBlobs", transforms=transforms_config, text_description="validation"
    )
    val_dataset_config.__class__ = new_class
    val_dataset_config.sample_size = val_dataset_shape[0]
    val_dataset_config.num_coils = val_dataset_shape[2]
    val_dataset_config.spatial_shape = (val_dataset_shape[1],) + val_dataset_shape[3:]

    checkpointer_config = CheckpointerConfig(checkpoint_steps=checkpointer_iters)
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])

    training_config = TrainingConfig(
        loss=loss_config,
        checkpointer=checkpointer_config,
        num_iterations=train_iters,
        validation_steps=val_iters,
        datasets=[train_dataset_config],
        batch_size=train_batch_size,
    )

    validation_config = ValidationConfig(crop=None, datasets=[val_dataset_config], batch_size=val_batch_size)

    inference_config = InferenceConfig(dataset=DatasetConfig(name="FakeMRIBlobs"), batch_size=inference_batch_size)

    model = ModelConfig(model_name="unet.unet_2d.Unet2d")
    config = DefaultConfig(
        training=training_config,
        validation=validation_config,
        inference=inference_config,
        model=model,
    )
    config.__class__ = make_dataclass(
        "",
        fields=[("additional_models", DictConfig, field(init=False))],
        bases=(DefaultConfig,),
    )
    config.additional_models = DictConfig({"senistivity_model": ModelConfig(model_name="unet.unet_2d.UnetModel2d")})
    return OmegaConf.create(config)


@pytest.mark.parametrize(
    "train_dataset_shape, val_dataset_shape,",
    [[(6, 5, 3, 120, 120), (6, 5, 3, 120, 120)]],
)
@pytest.mark.parametrize(
    "train_batch_size, val_batch_size, inference_batch_size",
    [[3, 3, 5]],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "ssim_loss", "l2_loss"]],
)
@pytest.mark.parametrize(
    "train_iters, val_iters, checkpointer_iters",
    [[41, 20, 20]],
)
def test_setup_train(
    train_dataset_shape,
    val_dataset_shape,
    train_batch_size,
    val_batch_size,
    loss_fns,
    train_iters,
    val_iters,
    checkpointer_iters,
    inference_batch_size,
):

    cfg = create_test_cfg(
        train_dataset_shape,
        val_dataset_shape,
        train_batch_size,
        val_batch_size,
        loss_fns,
        train_iters,
        val_iters,
        checkpointer_iters,
        inference_batch_size,
    )

    with tempfile.TemporaryDirectory() as tempdir:
        cfg_filename = pathlib.Path(tempdir) / "cfg_test.yaml"
        with open(cfg_filename, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))

        run_name = "test"
        training_root = None
        validation_root = None
        base_directory = pathlib.Path(tempdir) / "base_dir"
        num_machines = 1
        num_gpus = 0
        dist_url = "tcp://127.0.0.1:1234"
        force_validation = False
        initialization_checkpoint = None
        initialization_images = None
        initialization_kspace = None
        noise = None
        device = "cpu"
        num_workers = 1
        resume = False
        machine_rank = 0
        mixed_precision = False
        debug = False

        launch(
            setup_train,
            num_machines,
            num_gpus,
            machine_rank,
            dist_url,
            run_name,
            training_root,
            validation_root,
            base_directory,
            cfg_filename,
            force_validation,
            initialization_checkpoint,
            initialization_images,
            initialization_kspace,
            noise,
            device,
            num_workers,
            resume,
            machine_rank,
            mixed_precision,
            debug,
        )
        save_directory = base_directory / run_name
        assert cfg_filename.is_file()

        for idx in range(checkpointer_iters, train_iters + 1, checkpointer_iters):
            assert (save_directory / f"model_{idx}.pt").is_file()
        for idx in range(val_iters, train_iters + 1, val_iters):
            assert (save_directory / f"metrics_val_validation_{idx}.json").is_file()
