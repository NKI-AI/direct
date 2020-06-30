# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import random
import numpy as np
import torch
import os
import sys
import pathlib

import direct.launch

from direct.common.subsample import build_masking_function
from direct.data.mri_transforms import build_mri_transforms
from direct.data.datasets import build_dataset
from direct.data.lr_scheduler import WarmupMultiStepLR
from direct.environment import setup_environment, Args
from direct.utils import str_to_class


logger = logging.getLogger(__name__)


def setup_train(run_name, training_root, validation_root, base_directory,
                cfg_filename, checkpoint, device, num_workers, resume, machine_rank):

    # TODO(jt): Checkpoint is not yet passed!

    cfg, experiment_directory, forward_operator, backward_operator, engine = \
        setup_environment(run_name, base_directory, cfg_filename, device, machine_rank)

    # Create training and validation data
    # Masking configuration
    masking_config = {
        k: v for k, v in dict(cfg.masking).items() if not any([_ in k for _ in ['accelerations', 'center_fractions']])}

    val_accelerations = cfg.masking.accelerations \
        if not cfg.masking.val_accelerations else cfg.masking.val_accelerations
    val_center_fractions = cfg.masking.center_fractions \
        if not cfg.masking.val_center_fractions else cfg.masking.val_center_fractions

    train_mask_func = build_masking_function(
        **masking_config, accelerations=cfg.masking.accelerations, center_fractions=cfg.masking.center_fractions)

    val_mask_func = build_masking_function(
        **masking_config, accelerations=val_accelerations, center_fractions=val_center_fractions)

    train_transforms = build_mri_transforms(
        train_mask_func,
        crop=cfg.dataset.transforms.crop,
        image_center_crop=False,
        estimate_sensitivity_maps=cfg.dataset.transforms.estimate_sensitivity_maps,
        forward_operator=forward_operator,
        backward_operator=backward_operator
    )
    val_transforms = build_mri_transforms(
        val_mask_func,
        crop=cfg.dataset.transforms.crop,
        image_center_crop=True,
        estimate_sensitivity_maps=cfg.dataset.transforms.estimate_sensitivity_maps,
        forward_operator=forward_operator,
        backward_operator=backward_operator
    )

    # Trigger cudnn benchmark when the number of different input shapes is small.
    torch.backends.cudnn.benchmark = True

    training_data = build_dataset(
        cfg.dataset.name, training_root, sensitivity_maps=None, transforms=train_transforms)
    logger.info(f'Training data size: {len(training_data)}.')

    if validation_root:
        validation_data = build_dataset(
            cfg.dataset.name, validation_root, sensitivity_maps=None, transforms=val_transforms)
        logger.info(f'Validation data size: {len(validation_data)}.')
    else:
        logger.info(f'No validation data.')
        validation_data = None

    # Create the optimizers
    logger.info('Building optimizers.')
    optimizer: torch.optim.Optimizer = str_to_class('torch.optim', cfg.training.optimizer)(  # noqa
        engine.model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )  # noqa

    # Build the LR scheduler, we use a fixed LR schedule step size, no adaptive training schedule.
    solver_steps = list(range(cfg.training.lr_step_size, cfg.training.num_iterations, cfg.training.lr_step_size))
    lr_scheduler = WarmupMultiStepLR(
        optimizer, solver_steps, cfg.training.lr_gamma, warmup_factor=1 / 3.,
        warmup_iters=cfg.training.lr_warmup_iter, warmup_method='linear')

    # Just to make sure.
    torch.cuda.empty_cache()

    engine.train(
        optimizer, lr_scheduler, training_data, experiment_directory,
        validation_data=validation_data, resume=resume, num_workers=num_workers)


if __name__ == '__main__':
    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} training_set validation_set experiment_dir --num-gpus 8 --cfg cfg.yaml
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} training_set validation_set experiment_dir --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} training_set validation_set experiment_dir --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument('training_root', type=pathlib.Path, help='Path to the training data.')
    parser.add_argument('validation_root', type=pathlib.Path, help='Path to the validation data.')
    parser.add_argument('experiment_directory', type=pathlib.Path, help='Path to the experiment directory.')
    parser.add_argument('--resume', help='Resume training if possible.', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.name if args.name is not None else os.path.basename(args.cfg_file)[:-5]

    # TODO(jt): This can move to environment.
    # There is no need for the launch script within one node and at most one GPU.
    if args.num_machines == 1 and args.num_gpus <= 1:
        setup_train(run_name, args.training_root, args.validation_root, args.experiment_directory,
                    args.cfg_file, args.checkpoint, args.device, args.num_workers, args.resume, args.machine_rank)

    else:
        direct.launch.launch(
            setup_train,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(run_name, args.training_root, args.validation_root, args.experiment_directory,
                  args.cfg_file, args.checkpoint, args.device, args.num_workers, args.resume, args.machine_rank),
        )
