# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import random
import numpy as np
import torch
import os
import sys
import pathlib
import functools


from direct.common.subsample import build_masking_function
from direct.data.mri_transforms import build_mri_transforms
from direct.data.datasets import build_dataset
from direct.data.lr_scheduler import WarmupMultiStepLR
from direct.environment import setup_training_environment, Args
from direct.launch import launch
from direct.utils import str_to_class
from direct.utils.io import read_list, read_json

from collections import defaultdict


logger = logging.getLogger(__name__)


def remove_keys(input_dict, keys):
    input_dict = dict(input_dict).copy()
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for key in keys:
        if key not in input_dict:
            continue
        del input_dict[key]
    return input_dict


def parse_noise_dict(noise_dict, percentile=1.0, multiplier=1.0):
    logger.info(f"Parsing noise dictionary...")
    output = defaultdict(dict)
    for filename in noise_dict:
        data_per_volume = noise_dict[filename]
        for slice_no in data_per_volume:
            curr_data = data_per_volume[slice_no]
            if percentile != 1.0:
                lower_clip = np.percentile(curr_data, 100 * (1 - percentile))
                upper_clip = np.percentile(curr_data, 100 * percentile)
                curr_data = np.clip(curr_data, lower_clip, upper_clip)

            output[filename][int(slice_no)] = (
                curr_data * multiplier
            ) ** 2  # np.asarray(curr_data) * multiplier# (np.clip(curr_data, lower_clip, upper_clip) * multiplier) ** 2

    return output


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


# TODO: Use build_dataset_from_environment in predict and validation.
def build_dataset_from_environment(
    env,
    datasets_config,
    lists_root,
    data_root,
    initial_images=None,
    initial_kspaces=None,
    pass_text_description=True,
    pass_dictionaries=None,
    **kwargs,
):

    datasets = []
    for idx, dataset_config in enumerate(datasets_config):
        mri_transforms_func = functools.partial(
            build_mri_transforms,
            forward_operator=env.engine.forward_operator,
            backward_operator=env.engine.backward_operator,
            mask_func=build_masking_function(**dataset_config.transforms.masking),
        )

        transforms = mri_transforms_func(
            **remove_keys(dataset_config.transforms, "masking")
        )

        logger.debug(f"Transforms ({idx + 1} / {len(datasets_config)} :\n{transforms}")

        if pass_text_description:
            if not dataset_config.text_description:
                dataset_config.text_description = f"ds{idx}" if len(datasets_config) > 1 else None
        else:
            dataset_config.text_description = None

        pass_h5s = None
        if initial_images is not None and initial_kspaces is not None:
            raise ValueError(
                f"initial_images and initial_kspaces are mutually exclusive. "
                f"Got {initial_images} and {initial_kspaces}."
            )

        if initial_images:
            pass_h5s = {
                "initial_image": (dataset_config.input_image_key, initial_images)
            }

        if initial_kspaces:
            pass_h5s = {
                "initial_kspace": (dataset_config.input_kspace_key, initial_kspaces)
            }

        filenames_filter = get_filenames_for_datasets(
                dataset_config, lists_root, data_root
            )

        dataset = build_dataset(
            root=data_root,
            filenames_filter=filenames_filter,
            transforms=transforms,
            pass_h5s=pass_h5s,
            pass_dictionaries=pass_dictionaries,
            **remove_keys(dataset_config, ["transforms", "lists"]),
        )
        datasets.append(dataset)
        logger.info(
            f"Data size for"
            f" {dataset_config.text_description} ({idx + 1}/{len(datasets_config)}): {len(dataset)}."
        )

    return datasets


def setup_train(
    run_name,
    training_root,
    validation_root,
    base_directory,
    cfg_filename,
    checkpoint,
    initial_images,
    initial_kspace,
    noise,
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

    if initial_kspace is not None and initial_images is not None:
        raise ValueError(f"Cannot both provide initial kspace or initial images.")

    pass_dictionaries = {}
    if noise is not None:
        if not env.cfg.physics.use_noise_matrix:
            raise ValueError(
                f"cfg.physics.use_noise_matrix is null, yet command line passed noise files."
            )

        noise = [read_json(fn) for fn in noise]
        pass_dictionaries["loglikelihood_scaling"] = [
            parse_noise_dict(
                _, percentile=0.999, multiplier=env.cfg.physics.noise_matrix_scaling
            )
            for _ in noise
        ]

    # Create training and validation data
    # Transforms configuration
    # TODO: More ** passing...

    training_datasets = build_dataset_from_environment(
        env=env,
        datasets_config=env.cfg.training.datasets,
        lists_root=cfg_filename.parents[0],
        data_root=training_root,
        initial_images=None if initial_images is None else initial_images[0],
        initial_kspaces=None if initial_kspace is None else initial_kspace[0],
        pass_text_description=False,
        pass_dictionaries=pass_dictionaries,
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
            initial_images=None if initial_images is None else initial_images[1],
            initial_kspaces=None if initial_kspace is None else initial_kspace[1],
            pass_text_description=True,
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
        validation_datasets=validation_data,
        resume=resume,
        initialization=checkpoint,
        num_workers=num_workers,
    )


def check_train_val(key, name):
    if key is not None and len(key) != 2:
        sys.exit(
            f"--{name} has to be of the form `train_folder, validation_folder` if a validation folder is set."
        )


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    parser.add_argument("--name", help="Run name.", required=False, type=str)

    args = parser.parse_args()

    if (
        args.initialization_images is not None
        and args.initialization_kspace is not None
    ):
        sys.exit(
            f"--initialization-images and --initialization-kspace are mutually exclusive."
        )
    check_train_val(args.initialization_images, "initialization-images")
    check_train_val(args.initialization_kspace, "initialization-kspace")
    check_train_val(args.noise, "noise")

    set_all_seeds(args.seed)

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
        args.initialization_images,
        args.initialization_kspace,
        args.noise,
        args.device,
        args.num_workers,
        args.resume,
        args.machine_rank,
        args.mixed_precision,
        args.debug,
    )
