# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import random
import numpy as np
import torch
import os
import sys
import pathlib
import h5py

import direct.launch

from direct.common.subsample import build_masking_function
from direct.data.mri_transforms import build_mri_transforms
from direct.data.datasets import build_dataset
from direct.environment import setup_environment, Args
logger = logging.getLogger(__name__)


def setup_inference(run_name, data_root, base_directory, output_directory,
                    cfg_filename, checkpoint, accelerations, center_fractions,
                    device, num_workers, machine_rank):

    # TODO(jt): This is a duplicate line, check how this can be merged with train_rim.py
    # TODO(jt): Log elsewhere than for training.
    # TODO(jt): Logging is different when having multiple processes.
    cfg, experiment_directory, forward_operator, backward_operator, engine\
        = setup_environment(run_name, base_directory, cfg_filename, device, machine_rank)

    # Create training and validation data
    # Masking configuration
    if len(cfg.validation.datasets) > 1:
        logger.warning('Multiple validation datasets given. Will only evaluate the first.')

    mask_func = build_masking_function(**cfg.validation.datasets[0].transforms.masking)

    mri_transforms = build_mri_transforms(
        mask_func,
        crop=None,  # No cropping needed for testing
        image_center_crop=True,
        estimate_sensitivity_maps=cfg.training.dataset.transforms.estimate_sensitivity_maps,
        forward_operator=forward_operator,
        backward_operator=backward_operator
    )

    # Trigger cudnn benchmark when the number of different input shapes is small.
    torch.backends.cudnn.benchmark = True

    # TODO(jt): batches should have constant shapes! This works for Calgary Campinas because they are all with 256
    # slices.
    data = build_dataset(
        cfg.validation.datasets[0].name, data_root, sensitivity_maps=None, transforms=mri_transforms)
    logger.info(f'Inference data size: {len(data)}.')

    # Just to make sure.
    torch.cuda.empty_cache()

    # Run prediction
    output = engine.predict(data, experiment_directory, checkpoint_number=checkpoint, num_workers=num_workers)

    # Create output directory
    output_directory.mkdir(exist_ok=True, parents=True)

    # Only relevant for the Calgary Campinas challenge.
    # TODO(jt): This can be inferred from the configuration.
    crop = (50, -50)

    # TODO(jt): Perhaps aggregation to the main process would be most optimal here before writing.
    for idx, filename in enumerate(output):
        # The output has shape (depth, 1, height, width)
        logger.info(f'({idx + 1}/{len(output)}): Writing {output_directory / filename}...')
        reconstruction = torch.stack([_[1].rename(None) for _ in output[filename]]).numpy()[:, 0, ...].astype(np.float)
        if crop:
            reconstruction = reconstruction[slice(*crop)]

        # Only needed to fix a bug in Calgary Campinas training
        reconstruction = reconstruction / np.sqrt(np.prod(reconstruction.shape[1:]))

        with h5py.File(output_directory / filename, 'w') as f:
            f.create_dataset('reconstruction', data=reconstruction)


if __name__ == '__main__':
    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} validation_root output_directory --num-gpus 8 --cfg cfg.yaml
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} validation_root output_directory --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} validation_root output_directory --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument('validation_root', type=pathlib.Path, help='Path to the validation data.')
    parser.add_argument('experiment_directory', type=pathlib.Path, help='Path to the experiment directory.')
    parser.add_argument('output_directory', type=pathlib.Path, help='Path to the output directory.')
    parser.add_argument('--accelerations', nargs='+', default=None, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')
    parser.add_argument('--center-fractions', nargs='+', default=None, type=float,
                        help='Fraction of low-frequency ACS to be sampled. Should '
                             'have the same length as accelerations.')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.name if args.name is not None else os.path.basename(args.cfg_file)[:-5]

    direct.launch.launch(setup_inference, args.num_machines, args.num_gpus, args.machine_rank, args.dist_url,
                         run_name, args.validation_root, args.experiment_directory, args.output_directory,
                         args.cfg_file, args.checkpoint, args.accelerations, args.center_fractions, args.device,
                         args.num_workers, args.machine_rank)
