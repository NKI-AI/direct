# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import random
import numpy as np
import torch
import os
import sys

import direct.utils.logging
import direct.launch

from omegaconf import OmegaConf
from pprint import pformat

from direct.common.subsample import build_masking_functions
from direct.data.mri_transforms import build_mri_transforms
from direct.data.datasets import build_datasets
from direct.data.lr_scheduler import WarmupMultiStepLR
from direct.utils import communication
from direct.nn.rim.rim_engine import RIMEngine
from direct.config.defaults import DefaultConfig, TrainingConfig
from direct.nn.rim.mri_models import MRIReconstruction
from direct.utils import str_to_class

from projects.fastmri.args import Args

logger = logging.getLogger(__name__)


def setup(run_name, training_root, validation_root, base_directory,
          cfg_filename, device, num_workers, resume, machine_rank):
    experiment_dir = base_directory / run_name

    if communication.get_local_rank() == 0:
        # Want to prevent multiple workers from trying to write a directory
        # This is required in the logging below
        experiment_dir.mkdir(parents=True, exist_ok=True)
    communication.synchronize()  # Ensure folders are in place.

    # Load configs from YAML file to check which model needs to be loaded.
    cfg_from_file = OmegaConf.load(cfg_filename)
    model_name = cfg_from_file.model_name + 'Config'
    try:
        model_cfg = str_to_class(f'direct.nn.{cfg_from_file.model_name.lower()}.config', model_name)
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(f'Model configuration does not exist for {cfg_from_file.model_name} (err = {e}).')
        sys.exit(-1)

    # Load the default configs to ensure type safety
    base_cfg = OmegaConf.structured(DefaultConfig)
    base_cfg = OmegaConf.merge(base_cfg, {'model': model_cfg, 'training': TrainingConfig()})
    cfg = OmegaConf.merge(base_cfg, cfg_from_file)

    # Setup logging
    log_file = experiment_dir / f'log_{machine_rank}_{communication.get_local_rank()}.txt'
    direct.utils.logging.setup(
        use_stdout=communication.get_local_rank() == 0 or cfg.debug,
        filename=log_file,
        log_level=('INFO' if not cfg.debug else 'DEBUG')
    )
    logger.info(f'Machine rank: {machine_rank}.')
    logger.info(f'Local rank: {communication.get_local_rank()}.')
    logger.info(f'Logging: {log_file}.')
    logger.info(f'Saving to: {experiment_dir}.')
    logger.info(f'Run name: {run_name}.')
    logger.info(f'Config file: {cfg_filename}.')
    logger.info(f'Python version: {sys.version}.')
    logger.info(f'PyTorch version: {torch.__version__}.')  # noqa
    logger.info(f'CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()}.')
    logger.info(f'Configuration: {pformat(dict(cfg))}.')

    # Create the model
    logger.info('Building model.')
    model = MRIReconstruction(2, **cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Number of parameters: {n_params} ({n_params / 10.0**3:.2f}k).')
    logger.debug(model)

    # Create training and validation data
    train_mask_func, val_mask_func = build_masking_functions(**cfg.masking)
    train_transforms, val_transforms = build_mri_transforms(
        train_mask_func, val_mask_func=val_mask_func, crop=cfg.dataset.transforms.crop)

    training_data, validation_data = build_datasets(
        cfg.dataset.name, training_root, train_sensitivity_maps=None, train_transforms=train_transforms,
        validation_root=validation_root, val_sensitivity_maps=None, val_transforms=val_transforms)

    # Create the optimizers
    logger.info('Building optimizers.')
    optimizer: torch.optim.Optimizer = str_to_class('torch.optim', cfg.training.optimizer)(  # noqa
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )  # noqa

    # Build the LR scheduler, we use a fixed LR schedule step size, no adaptive training schedule.
    solver_steps = list(range(cfg.training.lr_step_size, cfg.training.num_iterations, cfg.training.lr_step_size))
    lr_scheduler = WarmupMultiStepLR(
        optimizer, solver_steps, cfg.training.lr_gamma, warmup_factor=1 / 3.,
        warmup_iters=cfg.training.lr_warmup_iter, warmup_method='linear')

    # Just to make sure.
    torch.cuda.empty_cache()

    # Setup training engine.
    engine = RIMEngine(cfg, model, device=device)

    engine.train(
        optimizer, lr_scheduler, training_data, experiment_dir,
        validation_data=validation_data, resume=resume, num_workers=num_workers)


if __name__ == '__main__':
    args = Args().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    name = args.name if args.name is not None else os.path.basename(args.cfg_file)[:-5]

    # There is no need for the launch script within one node and at most one GPU.
    if args.num_machines == 1 and args.num_gpus <= 1:
        setup(name, args.training_root, args.validation_root, args.experiment_directory,
              args.cfg_file, args.device, args.num_workers, args.resume, args.machine_rank)

    else:
        direct.launch.launch(
            setup,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(name, args.training_root, args.validation_root, args.experiment_directory,
                  args.cfg_file, args.device, args.num_workers, args.resume, args.machine_rank),
        )
