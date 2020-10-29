# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import os
import sys
import torch
import pathlib
import direct.utils.logging

from collections import namedtuple
from typing import Callable, Optional, Union

from direct.config.defaults import (
    DefaultConfig,
    TrainingConfig,
    InferenceConfig,
    ValidationConfig,
)
from torch.utils import collect_env
from direct.nn.rim.mri_models import MRIReconstruction
from direct.utils import communication, str_to_class, count_parameters

from omegaconf import OmegaConf
import omegaconf

import logging

logger = logging.getLogger(__name__)


def load_model_config_from_name(model_name):
    """
    Load specific configuration module for

    Parameters
    ----------
    model_name : path to model relative to direct.nn

    Returns
    -------
    model configuration.
    """
    module_path = f"direct.nn.{model_name.split('.')[0].lower()}.config"
    model_name += "Config"
    config_name = model_name.split(".")[-1]
    try:
        model_cfg = str_to_class(module_path, config_name)
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(
            f"Path {module_path} for config_name {config_name} does not exist (err = {e})."
        )
        sys.exit(-1)
    return model_cfg


def load_model_from_name(model_name):
    module_path = (
        f"direct.nn.{'.'.join([_.lower() for _ in model_name.split('.')[:-1]])}"
    )
    module_name = model_name.split(".")[-1]
    try:
        model = str_to_class(module_path, module_name)
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(
            f"Path {module_path} for model_name {module_name} does not exist (err = {e})."
        )
        sys.exit(-1)

    return model


def load_dataset_config(dataset_name):
    dataset_config = str_to_class(
        "direct.data.datasets_config", dataset_name + "Config"
    )
    return dataset_config


def build_operators(cfg) -> (Callable, Callable):
    # Get the operators
    forward_operator = str_to_class(f"direct.data.transforms", cfg.forward_operator)
    backward_operator = str_to_class(f"direct.data.transforms", cfg.backward_operator)
    return forward_operator, backward_operator


def setup_logging(machine_rank, output_directory, run_name, cfg_filename, cfg, debug):
    # Setup logging
    log_file = (
        output_directory / f"log_{machine_rank}_{communication.get_local_rank()}.txt"
    )

    direct.utils.logging.setup(
        use_stdout=communication.is_main_process() or debug,
        filename=log_file,
        log_level=("INFO" if not debug else "DEBUG"),
    )
    logger.info(f"Machine rank: {machine_rank}.")
    logger.info(f"Local rank: {communication.get_local_rank()}.")
    logger.info(f"Logging: {log_file}.")
    logger.info(f"Saving to: {output_directory}.")
    logger.info(f"Run name: {run_name}.")
    logger.info(f"Config file: {cfg_filename}.")
    logger.info(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()}.")
    logger.info(f"Environment information: {collect_env.get_pretty_env_info()}.")
    logger.info(f"DIRECT version: {direct.__version__}.")  # noqa
    git_hash = direct.utils.git_hash()
    logger.info(f"Git hash: {git_hash if git_hash else 'N/A'}.")  # noqa
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}.")


def load_models_into_environment_config(cfg_from_file):
    # Load the configuration for the models
    cfg = {"model": cfg_from_file.model}

    if "additional_models" in cfg_from_file:
        cfg = {**cfg, **cfg_from_file.additional_models}
    # Parse config of additional models
    # TODO: Merge this with the normal model config loading.
    models_config = {}
    models = {}
    for curr_model_name in cfg:
        if "model_name" not in cfg[curr_model_name]:
            logger.error(f"Model {curr_model_name} has no model_name.")
            sys.exit(-1)

        curr_model_cfg = cfg[curr_model_name]
        model_name = curr_model_cfg.model_name
        models[curr_model_name] = load_model_from_name(model_name)

        models_config[curr_model_name] = OmegaConf.merge(
            load_model_config_from_name(model_name), curr_model_cfg
        )

    models_config = OmegaConf.merge(models_config)
    return models, models_config


def initialize_models_from_config(
    cfg, models, forward_operator, backward_operator, device
):
    # Create the model
    logger.info("Building models.")
    # TODO(jt): Model name is not used here.
    additional_models = {}
    _model = None
    for k, v in cfg.additional_models.items():
        # Remove model_name key
        curr_model = models[k]
        curr_model_cfg = {kk: vv for kk, vv in v.items() if kk != "model_name"}
        additional_models[k] = curr_model(**curr_model_cfg)

    # MODEL SHOULD LOAD MRI RECONSTRUCTION INSTEAD AND USE A FUNCTOOLS PARTIAL TO PASS THE OPERATORS
    # the_real_model = models["model"](**{k: v for k, v in cfg.model.items() if k != "model_name"})
    model = MRIReconstruction(
        models["model"], forward_operator, backward_operator, 2, **cfg.model
    ).to(device)

    # Log total number of parameters
    count_parameters({"model": model, **additional_models})
    return model, additional_models


def setup_engine(
    cfg,
    device,
    model,
    additional_models: dict,
    forward_operator: Optional[Union[Callable, object]] = None,
    backward_operator: Optional[Union[Callable, object]] = None,
    mixed_precision: bool = False,
):
    # Setup engine.
    # There is a bit of repetition here, but the warning provided is more descriptive
    # TODO(jt): Try to find a way to combine this with the setup above.
    model_name_short = cfg.model.model_name.split(".")[0]
    engine_name = cfg.model.model_name.split(".")[-1] + "Engine"

    try:
        engine_class = str_to_class(
            f"direct.nn.{model_name_short.lower()}.{model_name_short.lower()}_engine",
            engine_name,
        )
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(f"Engine does not exist for {cfg.model.model_name} (err = {e}).")
        sys.exit(-1)

    engine = engine_class(  # noqa
        cfg,
        model,
        device=device,
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        mixed_precision=mixed_precision,
        **additional_models,
    )
    return engine


def extract_names(cfg):
    cfg = cfg.copy()
    if isinstance(cfg, omegaconf.dictconfig.DictConfig):
        if "name" not in cfg:
            raise ValueError("`name` needs to be present in config.")
        curr_name = cfg["name"]

    elif isinstance(cfg, omegaconf.listconfig.ListConfig):
        return [extract_names(v) for v in cfg]

    else:
        raise ValueError(f"Expected DictConfig or ListConfig. Got {type(cfg)}.")

    return curr_name, cfg


def setup_common_environment(
    run_name,
    base_directory,
    cfg_filename,
    device,
    machine_rank,
    mixed_precision,
    debug=False,
):
    experiment_dir = base_directory / run_name
    if communication.get_local_rank() == 0:
        # Want to prevent multiple workers from trying to write a directory
        # This is required in the logging below
        experiment_dir.mkdir(parents=True, exist_ok=True)
    communication.synchronize()  # Ensure folders are in place.

    # Load configs from YAML file to check which model needs to be loaded.
    cfg_from_file = OmegaConf.load(cfg_filename)

    # Load the default configs to ensure type safety
    cfg = OmegaConf.structured(DefaultConfig)

    models, models_config = load_models_into_environment_config(cfg_from_file)
    cfg.model = models_config.model
    del models_config["model"]
    cfg.additional_models = models_config

    # Setup everything for training
    cfg.training = TrainingConfig
    cfg.validation = ValidationConfig
    cfg.inference = InferenceConfig

    cfg_from_file_new = cfg_from_file.copy()
    for key in cfg_from_file:
        # TODO: This does not really do a full validation.
        # BODY: This will be handeled once Hydra is implemented.
        if key in ["models", "additional_models"]:  # Still handled separately
            continue

        elif key in ["training", "validation", "inference"]:
            if not cfg_from_file[key]:
                logger.info(f"key {key} missing in config.")
                continue

            if key in ["training", "validation"]:
                dataset_cfg_from_file = extract_names(cfg_from_file[key].datasets)
                for idx, (dataset_name, dataset_config) in enumerate(
                    dataset_cfg_from_file
                ):
                    cfg_from_file_new[key].datasets[idx] = dataset_config
                    cfg[key].datasets.append(load_dataset_config(dataset_name))
            else:
                dataset_name, dataset_config = extract_names(cfg_from_file[key].dataset)
                cfg_from_file_new[key].dataset = dataset_config
                cfg[key].dataset = load_dataset_config(dataset_name)

        cfg[key] = OmegaConf.merge(cfg[key], cfg_from_file_new[key])
    # sys.exit()
    # Make configuration read only.
    # TODO(jt): Does not work when indexing config lists.
    # OmegaConf.set_readonly(cfg, True)
    setup_logging(machine_rank, experiment_dir, run_name, cfg_filename, cfg, debug)
    forward_operator, backward_operator = build_operators(cfg.physics)

    model, additional_models = initialize_models_from_config(
        cfg, models, forward_operator, backward_operator, device
    )

    engine = setup_engine(
        cfg,
        device,
        model,
        additional_models,
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        mixed_precision=mixed_precision,
    )

    environment = namedtuple(
        "environment",
        ["cfg", "experiment_dir", "engine"],
    )
    return environment(cfg, experiment_dir, engine)


def setup_training_environment(
    run_name,
    base_directory,
    cfg_filename,
    device,
    machine_rank,
    mixed_precision,
    debug=False,
):

    env = setup_common_environment(
        run_name,
        base_directory,
        cfg_filename,
        device,
        machine_rank,
        mixed_precision,
        debug=debug,
    )
    # Write config file to experiment directory.
    config_file_in_project_folder = env.experiment_dir / "config.yaml"
    logger.info(f"Writing configuration file to: {config_file_in_project_folder}.")
    if communication.is_main_process():
        with open(config_file_in_project_folder, "w") as f:
            f.write(OmegaConf.to_yaml(env.cfg))
    communication.synchronize()

    return env


def setup_testing_environment(
    run_name,
    base_directory,
    device,
    machine_rank,
    mixed_precision,
    debug=False,
):

    cfg_filename = base_directory / run_name / "config.yaml"

    if not cfg_filename.exists():
        raise OSError(f"Config file {cfg_filename} does not exist.")

    env = setup_common_environment(
        run_name,
        base_directory,
        cfg_filename,
        device,
        machine_rank,
        mixed_precision,
        debug=debug,
    )

    out_env = namedtuple(
        "environment",
        ["cfg", "engine"],
    )
    return out_env(env.cfg, env.engine)


def setup_inference_environment(
    run_name,
    base_directory,
    device,
    machine_rank,
    mixed_precision,
    debug=False,
):

    env = setup_testing_environment(
        run_name, base_directory, device, machine_rank, mixed_precision, debug=debug
    )

    out_env = namedtuple(
        "environment",
        ["cfg", "engine"],
    )
    return out_env(env.cfg, env.engine)


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, epilog=None, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """
        super().__init__(
            epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter
        )

        self.add_argument(
            "--device",
            type=str,
            default="cuda",
            help='Which device to train on. Set to "cuda" to use the GPU.',
        )
        self.add_argument(
            "--seed", default=42, type=int, help="Seed for random number generators."
        )
        self.add_argument(
            "--num-workers", type=int, default=4, help="Number of workers."
        )
        self.add_argument(
            "--mixed-precision", help="Use mixed precision.", action="store_true"
        )
        self.add_argument("--debug", help="Set debug mode true.", action="store_true")

        self.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
        self.add_argument("--num-machines", type=int, default=1, help="# of machines.")
        self.add_argument(
            "--machine-rank",
            type=int,
            default=0,
            help="the rank of this machine (unique per machine).",
        )
        self.add_argument(
            "--initialization-images",
            help="Path to images which will be used as initialization to the model. "
            "The filenames assumed to be the same as the images themselves. If these are h5 files, "
            "the key to read in the h5 has to be set in the configuration in the dataset.input_image_key.",
            required=False,
            nargs="+",
            type=pathlib.Path,
        )
        self.add_argument(
            "--initialization-kspace",
            help="Path to kspace which will be used as initialization to the model. "
            "The filenames assumed to be the same as the images themselves. If these are h5 files, "
            "the key to read in the h5 has to be set in the configuration in the dataset.input_image_key.",
            required=False,
            nargs="+",
            type=pathlib.Path,
        )
        self.add_argument(
            "--noise",
            help="Path to json file mapping relative filename to noise estimates. "
            "Path to training and validation data",
            required=False,
            nargs="+",
            type=pathlib.Path,
        )

        # Taken from: https://github.com/facebookresearch/detectron2/blob/bd2ea475b693a88c063e05865d13954d50242857/detectron2/engine/defaults.py#L49 # noqa
        # PyTorch still may leave orphan processes in multi-gpu training.
        # Therefore we use a deterministic way to obtain port,
        # so that users are aware of orphan processes by seeing the port occupied.
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
        self.add_argument(
            "--dist-url",
            default=f"tcp://127.0.0.1:{port}",
            help="initialization URL for pytorch distributed backend. See "
            "https://pytorch.org/docs/stable/distributed.html for details.",
        )

        self.set_defaults(**overrides)
