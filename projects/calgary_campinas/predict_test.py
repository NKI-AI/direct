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

from direct.common.subsample import CalgaryCampinasMaskFunc
from direct.data.mri_transforms import build_mri_transforms, Compose
from direct.data.datasets import build_dataset
from direct.environment import setup_environment, Args

logger = logging.getLogger(__name__)


class CreateSamplingMask:
    """
    Create sampling mask from a dictionary.
    """

    def __init__(self, masks_dict):
        self.masks_dict = masks_dict

    def __call__(self, sample):
        sample["sampling_mask"] = self.masks_dict[sample["filename"]][
            np.newaxis, ..., np.newaxis
        ]
        sample["acs_mask"] = torch.from_numpy(
            CalgaryCampinasMaskFunc(accelerations=[]).circular_centered_mask(
                sample["kspace"].shape[1:], 18
            )
        )

        return sample


def volume_post_processing(volume):
    # Only relevant for the Calgary Campinas challenge.
    crop = (50, -50)
    if crop:
        volume = volume[slice(*crop)]

    # Only needed to fix a bug in Calgary Campinas training
    volume = volume / np.sqrt(np.prod(volume.shape[1:]))
    return volume


def setup_inference(
    run_name,
    data_root,
    base_directory,
    output_directory,
    volume_processing_func,
    cfg_filename,
    checkpoint,
    masks,
    device,
    num_workers,
    machine_rank,
):

    # TODO(jt): This is a duplicate line, check how this can be merged with train_rim.py
    # TODO(jt): Log elsewhere than for training.
    # TODO(jt): Logging is different when having multiple processes.
    # TODO(jt): This can be merged with run_rim.py
    env = setup_environment(
        run_name, base_directory, cfg_filename, device, machine_rank
    )

    # Process all masks
    all_maps = masks.glob("*.npy")
    logger.info("Loading masks...")
    masks_dict = {
        filename.name.replace(".npy", ".h5"): np.load(filename) for filename in all_maps
    }
    logger.info(f"Loaded {len(masks_dict)} masks.")

    # Don't add the mask func, add it separately
    mri_transforms = build_mri_transforms(
        forward_operator=env.forward_operator,
        backward_operator=env.backward_operator,
        mask_func=None,
        crop=None,  # No cropping needed for testing
        image_center_crop=True,
        estimate_sensitivity_maps=env.cfg.training.dataset.transforms.estimate_sensitivity_maps,
    )

    mri_transforms = Compose([CreateSamplingMask(masks_dict), mri_transforms])

    # Trigger cudnn benchmark when the number of different input shapes is small.
    torch.backends.cudnn.benchmark = True

    # TODO(jt): batches should have constant shapes! This works for Calgary Campinas because they are all with 256
    # slices.
    if len(env.cfg.validation.datasets) > 1:
        logger.warning("Multiple datasets given. Will only predict the first.")

    data = build_dataset(
        env.cfg.validation.dataset[0].name,
        data_root,
        sensitivity_maps=None,
        transforms=mri_transforms,
    )
    logger.info(f"Inference data size: {len(data)}.")

    # Just to make sure.
    torch.cuda.empty_cache()

    # Run prediction
    output = env.engine.predict(
        data, env.experiment_dir, checkpoint_number=checkpoint, num_workers=num_workers,
    )

    # Create output directory
    output_directory.mkdir(exist_ok=True, parents=True)

    # TODO(jt): Perhaps aggregation to the main process would be most optimal here before writing.
    for idx, filename in enumerate(output):
        # The output has shape (depth, 1, height, width)
        logger.info(
            f"({idx + 1}/{len(output)}): Writing {output_directory / filename}..."
        )
        reconstruction = (
            torch.stack([_[1].rename(None) for _ in output[filename]])
            .numpy()[:, 0, ...]
            .astype(np.float)
        )
        reconstruction = volume_processing_func(reconstruction)

        with h5py.File(output_directory / filename, "w") as f:
            f.create_dataset("reconstruction", data=reconstruction)


if __name__ == "__main__":
    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} test_root output_directory --num-gpus 8 --cfg cfg.yaml
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} test_root output_directory --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} test_root output_directory --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument(
        "test_root", type=pathlib.Path, help="Path to the validation data."
    )
    parser.add_argument(
        "experiment_dir", type=pathlib.Path, help="Path to the experiment directory.",
    )
    parser.add_argument(
        "output_directory", type=pathlib.Path, help="Path to the output directory."
    )
    parser.add_argument("--masks", type=pathlib.Path, help="Path to the masks.")
    parser.add_argument("--checkpoint", type=int, required=True, help="Checkpoint number.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = (
        args.name if args.name is not None else os.path.basename(args.cfg_file)[:-5]
    )

    direct.launch.launch(
        setup_inference,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        run_name,
        args.test_root,
        args.experiment_dir,
        args.output_directory,
        volume_post_processing,
        args.cfg_file,
        args.checkpoint,
        args.masks,
        args.device,
        args.num_workers,
        args.machine_rank,
    )
