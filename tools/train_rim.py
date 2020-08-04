# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import random
import numpy as np
import torch
import os
import sys
import pathlib

from direct.common.subsample import build_masking_function
from direct.data.mri_transforms import build_mri_transforms
from direct.data.datasets import build_dataset
from direct.data.lr_scheduler import WarmupMultiStepLR
from direct.environment import setup_environment, Args
from direct.launch import launch
from direct.utils import str_to_class


logger = logging.getLogger(__name__)


def setup_train(
    run_name,
    training_root,
    validation_root,
    base_directory,
    cfg_filename,
    checkpoint,
    device,
    num_workers,
    resume,
    machine_rank,
    mixed_precision,
    debug,
):

    (
        cfg,
        experiment_directory,
        forward_operator,
        backward_operator,
        engine,
    ) = setup_environment(
        run_name,
        base_directory,
        cfg_filename,
        device,
        machine_rank,
        mixed_precision,
        debug=debug,
    )

    # Create training and validation data
    # Transforms configuration
    train_transforms = build_mri_transforms(
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        mask_func=build_masking_function(**cfg.training.dataset.transforms.masking),
        crop=cfg.training.dataset.transforms.crop,
        image_center_crop=False,
        estimate_sensitivity_maps=cfg.training.dataset.transforms.estimate_sensitivity_maps,
    )
    logger.debug(f"Train transforms:\n{train_transforms}")

    # Training data
    training_data = build_dataset(
        cfg.training.dataset.name,
        training_root,
        sensitivity_maps=None,
        transforms=train_transforms,
    )
    logger.info(f"Training data size: {len(training_data)}.")
    logger.debug(f"Training dataset:\n{training_data}")

    # Validation is the same as training, but looped over all datasets
    if validation_root:
        validation_data = []
        for idx, dataset_config in enumerate(cfg.validation.datasets):
            val_transforms = build_mri_transforms(
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                mask_func=build_masking_function(**dataset_config.transforms.masking),
                # TODO(jt): Batch sampler needs to make sure volumes of same shape get passed.
                crop=dataset_config.transforms.crop,
                image_center_crop=True,
                estimate_sensitivity_maps=dataset_config.transforms.estimate_sensitivity_maps,
            )
            curr_validation_data = build_dataset(
                dataset_config.name,
                validation_root,
                sensitivity_maps=None,
                transforms=val_transforms,
            )
            logger.info(
                f"Validation data size for dataset"
                f" {dataset_config.name} ({idx + 1}/{len(cfg.validation.datasets)}): {len(curr_validation_data)}."
            )
            validation_data.append(curr_validation_data)
    else:
        logger.info(f"No validation data.")
        validation_data = None

    # Create the optimizers
    logger.info("Building optimizers.")
    optimizer: torch.optim.Optimizer = str_to_class(
        "torch.optim", cfg.training.optimizer
    )(  # noqa
        engine.model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )  # noqa

    # Build the LR scheduler, we use a fixed LR schedule step size, no adaptive training schedule.
    solver_steps = list(
        range(
            cfg.training.lr_step_size,
            cfg.training.num_iterations,
            cfg.training.lr_step_size,
        )
    )
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        solver_steps,
        cfg.training.lr_gamma,
        warmup_factor=1 / 3.0,
        warmup_iterations=cfg.training.lr_warmup_iter,
        warmup_method="linear",
    )

    # Just to make sure.
    torch.cuda.empty_cache()

    engine.train(
        optimizer,
        lr_scheduler,
        training_data,
        experiment_directory,
        validation_data=validation_data,
        resume=resume,
        initialization=checkpoint,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} training_set validation_set experiment_dir --num-gpus 8 --cfg cfg.yaml
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} training_set validation_set experiment_dir --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} training_set validation_set experiment_dir --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument(
        "training_root", type=pathlib.Path, help="Path to the training data."
    )
    parser.add_argument(
        "validation_root", type=pathlib.Path, help="Path to the validation data."
    )
    parser.add_argument(
        "experiment_directory",
        type=pathlib.Path,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=pathlib.Path,
        help="If this value is set to a proper checkpoint when training starts, "
        "the model will be initialized with the weights given. "
        "No other keys in the checkpoint will be loaded. "
        "When another checkpoint would be available and the --resume flag is used, "
        "this flag is ignored.",
    )
    parser.add_argument(
        "--resume", help="Resume training if possible.", action="store_true"
    )
    parser.add_argument(
        "--mixed-precision", help="Use mixed precision training.", action="store_true"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = (
        args.name if args.name is not None else os.path.basename(args.cfg_file)[:-5]
    )

    # TODO(jt): Duplicate params
    launch(
        setup_train,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        run_name,
        args.training_root,
        args.validation_root,
        args.experiment_directory,
        args.cfg_file,
        args.initialization_checkpoint,
        args.device,
        args.num_workers,
        args.resume,
        args.machine_rank,
        args.mixed_precision,
        args.debug,
    )
