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
from direct.environment import setup_training_environment, Args
from direct.launch import launch
from direct.utils import str_to_class
from direct.utils.io import read_list


logger = logging.getLogger(__name__)


def get_filenames_for_datasets(cfg, files_root, data_root):
    """
    Given a list of filenames of data points, concatenate these into a large list of full filenames

    Parameters
    ----------
    cfg : cfg-object
        cfg object having property lists having the relative paths compared to files root.
    files_root : pathlib.Path
    data_root : pathlib.Path

    Returns
    -------

    """
    if not cfg.lists:
        return []
    filter_filenames = []
    for curr_list in cfg.lists:
        filter_filenames += [
            data_root / pathlib.Path(_)
            for _ in read_list(pathlib.Path(files_root) / curr_list)
        ]

    return filter_filenames


def build_dataset_from_environment(
    env, datasets_config, lists_root, data_root, type_data, **kwargs
):
    datasets = []
    for idx, dataset_config in enumerate(datasets_config):
        transforms = build_mri_transforms(
            forward_operator=env.forward_operator,
            backward_operator=env.backward_operator,
            mask_func=build_masking_function(**dataset_config.transforms.masking),
            crop=dataset_config.transforms.crop,
            crop_type=dataset_config.transforms.crop_type,
            image_center_crop=dataset_config.transforms.image_center_crop,
            estimate_sensitivity_maps=dataset_config.transforms.estimate_sensitivity_maps,
            pad_coils=dataset_config.transforms.pad_coils,
        )
        logger.debug(f"Transforms for {type_data}: {idx}:\n{transforms}")

        # Only give fancy names when validating
        # TODO(jt): Perhaps this can be split up to just a description parameters, and parse config in the main func.
        if type_data == "validation":
            if dataset_config.text_description:
                text_description = dataset_config.text_description
            else:
                text_description = f"ds{idx}" if len(datasets_config) > 1 else None
        elif type_data == "training":
            text_description = None
        else:
            raise ValueError(
                f"Type of data needs to be either `validation` or `training`, got {type_data}."
            )

        dataset = build_dataset(
            dataset_config.name,
            data_root,
            filenames_filter=get_filenames_for_datasets(
                dataset_config, lists_root, data_root
            ),
            sensitivity_maps=None,
            transforms=transforms,
            text_description=text_description,
            kspace_context=dataset_config.kspace_context,
            **kwargs,
        )
        datasets.append(dataset)
        logger.info(
            f"Data size for {type_data} dataset"
            f" {dataset_config.name} ({idx + 1}/{len(datasets_config)}): {len(dataset)}."
        )

    return datasets


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

    env = setup_training_environment(
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
    training_datasets = build_dataset_from_environment(
        env=env,
        datasets_config=env.cfg.training.datasets,
        lists_root=cfg_filename.parents[0],
        data_root=training_root,
        type_data="training",
    )
    training_data_sizes = [len(_) for _ in training_datasets]
    logger.info(
        f"Training data sizes: {training_data_sizes} (sum={sum(training_data_sizes)})."
    )

    if validation_root:
        validation_data = build_dataset_from_environment(
            env=env,
            datasets_config=env.cfg.validation.datasets,
            lists_root=cfg_filename.parents[0],
            data_root=validation_root,
            type_data="validation",
        )
    else:
        logger.info(f"No validation data.")
        validation_data = None

    # Create the optimizers
    logger.info("Building optimizers.")
    optimizer_params = [{"params": env.engine.model.parameters()}]
    for curr_model_name in env.engine.models:
        # TODO(jt): Can get learning rate from the config per additional model too.
        curr_learning_rate = env.cfg.training.lr
        logger.info(
            f"Adding model parameters of {curr_model_name} with learning rate {curr_learning_rate}."
        )
        optimizer_params.append(
            {
                "params": env.engine.models[curr_model_name].parameters(),
                "lr": curr_learning_rate,
            }
        )

    optimizer: torch.optim.Optimizer = str_to_class(
        "torch.optim", env.cfg.training.optimizer
    )(  # noqa
        optimizer_params,
        lr=env.cfg.training.lr,
        weight_decay=env.cfg.training.weight_decay,
    )  # noqa

    # Build the LR scheduler, we use a fixed LR schedule step size, no adaptive training schedule.
    solver_steps = list(
        range(
            env.cfg.training.lr_step_size,
            env.cfg.training.num_iterations,
            env.cfg.training.lr_step_size,
        )
    )
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        solver_steps,
        env.cfg.training.lr_gamma,
        warmup_factor=1 / 3.0,
        warmup_iterations=env.cfg.training.lr_warmup_iter,
        warmup_method="linear",
    )

    # Just to make sure.
    torch.cuda.empty_cache()

    env.engine.train(
        optimizer,
        lr_scheduler,
        training_datasets,
        env.experiment_dir,
        validation_data=validation_data,
        resume=resume,
        initialization=checkpoint,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a l ot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    # Remove warnings from named tensors being experimental
    os.environ["PYTHONWARNINGS"] = "ignore"

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
        "experiment_dir",
        type=pathlib.Path,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for training.",
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=pathlib.Path,  # noqa
        help="If this value is set to a proper checkpoint when training starts, "
        "the model will be initialized with the weights given. "
        "No other keys in the checkpoint will be loaded. "
        "When another checkpoint would be available and the --resume flag is used, "
        "this flag is ignored.",
    )
    parser.add_argument(
        "--resume", help="Resume training if possible.", action="store_true"
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
        args.experiment_dir,
        args.cfg_file,
        args.initialization_checkpoint,
        args.device,
        args.num_workers,
        args.resume,
        args.machine_rank,
        args.mixed_precision,
        args.debug,
    )
