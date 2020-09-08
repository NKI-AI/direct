# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import os
import sys
import torch
import direct.utils.logging

from direct.config.defaults import DefaultConfig, TrainingConfig, InferenceConfig
from direct.nn.rim.mri_models import MRIReconstruction
from direct.utils import communication, str_to_class, count_parameters
from collections import namedtuple


from omegaconf import OmegaConf

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


def load_models_from_config_file(cfg_from_file):
    cfg = {"model": cfg_from_file.model}

    if "additional_models" in cfg_from_file:
        cfg = {**cfg, **cfg_from_file.additional_models}
    # Parse config of additional models
    # TODO(jt): Merge this with the normal model config loading.
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
    return models_config, models


def load_dataset_config(dataset):
    dataset_name = dataset.name
    dataset_config = str_to_class(
        "direct.data.datasets_config", dataset_name + "Config"
    )
    return dataset_config


def build_operators(cfg):
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
        use_stdout=communication.get_local_rank() == 0 or debug,
        filename=log_file,
        log_level=("INFO" if not debug else "DEBUG"),
    )
    logger.info(f"Machine rank: {machine_rank}.")
    logger.info(f"Local rank: {communication.get_local_rank()}.")
    logger.info(f"Logging: {log_file}.")
    logger.info(f"Saving to: {output_directory}.")
    logger.info(f"Run name: {run_name}.")
    logger.info(f"Config file: {cfg_filename}.")
    logger.info(f"Python version: {sys.version.strip()}.")
    logger.info(f"PyTorch version: {torch.__version__}.")  # noqa
    logger.info(f"DIRECT version: {direct.__version__}.")  # noqa
    git_hash = direct.utils.git_hash()
    logger.info(f"Git hash: {git_hash if git_hash else 'N/A'}.")  # noqa
    logger.info(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()}.")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}.")


def load_models_into_environment_config(cfg_from_file):
    # Load the configuration for the models
    models_cfg, models = load_models_from_config_file(cfg_from_file)

    # Load the default configs to ensure type safety
    base_cfg = OmegaConf.structured(DefaultConfig)
    base_cfg.model = models_cfg.model
    del models_cfg["model"]
    base_cfg.additional_models = models_cfg

    return base_cfg, models


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
    cfg, device, model, additional_models: dict, mixed_precision: bool = False
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

    engine = engine_class(
        cfg,
        model,
        device=device,
        mixed_precision=mixed_precision,
        **additional_models,
    )
    return engine


def setup_common_environment(
    base_cfg,
    cfg_from_file,
    models,
    device,
    machine_rank,
    output_directory,
    run_name,
    cfg_filename,
    mixed_precision,
    debug,
):

    # Populate environment config with config from file
    cfg = OmegaConf.merge(base_cfg, cfg_from_file)

    # Make configuration read only.
    # TODO(jt): Does not work when indexing config lists.
    # OmegaConf.set_readonly(cfg, True)
    setup_logging(machine_rank, output_directory, run_name, cfg_filename, cfg, debug)

    forward_operator, backward_operator = build_operators(cfg.modality)
    model, additional_models = initialize_models_from_config(
        cfg, models, forward_operator, backward_operator, device
    )

    engine = setup_engine(
        cfg, device, model, additional_models, mixed_precision=mixed_precision
    )

    return forward_operator, backward_operator, engine, cfg


def setup_training_environment(
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
    base_cfg, models = load_models_into_environment_config(cfg_from_file)

    # Setup everything for training
    base_cfg.training = TrainingConfig
    # Parse the proper specific config for the datasets:
    base_cfg.training.datasets = [
        load_dataset_config(dataset) for dataset in base_cfg.training.datasets
    ]
    base_cfg.validation.datasets = [
        load_dataset_config(dataset) for dataset in base_cfg.validation.datasets
    ]

    # Make configuration read only.
    # TODO(jt): Does not work when indexing config lists.
    # OmegaConf.set_readonly(cfg, True)

    forward_operator, backward_operator, engine, cfg = setup_common_environment(
        base_cfg,
        cfg_from_file,
        models,
        device,
        machine_rank,
        experiment_dir,
        run_name,
        cfg_filename,
        mixed_precision,
        debug,
    )

    # Check if the file exists in the project directory
    config_file_in_project_folder = experiment_dir / "config.yaml"
    if config_file_in_project_folder.exists():
        if dict(OmegaConf.load(config_file_in_project_folder)) != dict(cfg):
            pass
            # raise ValueError(
            #     f"This project folder exists and has a config.yaml, "
            #     f"yet this does not match with the one the model was built with."
            # )
    else:
        if communication.get_local_rank() == 0:
            with open(config_file_in_project_folder, "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
        communication.synchronize()

    environment = namedtuple(
        "environment",
        ["cfg", "experiment_dir", "forward_operator", "backward_operator", "engine"],
    )
    return environment(cfg, experiment_dir, forward_operator, backward_operator, engine)


def setup_inference_environment(
    run_name,
    base_directory,
    output_directory,
    device,
    machine_rank,
    mixed_precision,
    debug=False,
):

    cfg_filename = base_directory / run_name / "config.yaml"
    if not cfg_filename.exists():
        raise OSError(f"Config file {cfg_filename} does not exist.")

    # Load configs from YAML file to check which model needs to be loaded.
    cfg_from_file = OmegaConf.load(cfg_filename)
    base_cfg, models = load_models_into_environment_config(cfg_from_file)
    base_cfg.inference = InferenceConfig
    base_cfg.inference.dataset = load_dataset_config(base_cfg.inference.dataset)

    forward_operator, backward_operator, engine, cfg = setup_common_environment(
        base_cfg,
        cfg_from_file,
        models,
        device,
        machine_rank,
        output_directory,
        run_name,
        cfg_filename,
        mixed_precision,
        debug,
    )

    environment = namedtuple(
        "environment",
        ["cfg", "output_directory", "forward_operator", "backward_operator", "engine"],
    )
    env = environment(
        cfg, output_directory, forward_operator, backward_operator, engine
    )
    return env


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
            "--name", help="Run name, if None use configs name.", default=None, type=str
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
