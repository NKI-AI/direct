# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import random
import numpy as np
import torch
import sys
import pathlib
import h5py
import os


import direct.launch

from functools import partial

from direct.common.subsample import build_masking_function
from direct.data.mri_transforms import build_mri_transforms
from direct.data.datasets import build_dataset
from direct.environment import setup_inference_environment, Args
from direct.utils.io import read_list

logger = logging.getLogger(__name__)


def setup_inference(
    run_name,
    cfg_file,
    data_root,
    base_directory,
    output_directory,
    filenames_filter,
    checkpoint,
    device,
    num_workers,
    machine_rank,
    volume_processing_func=None,
    mixed_precision=False,
    debug=False,
):
    env = setup_inference_environment(
        run_name,
        cfg_file,
        base_directory,
        output_directory,
        device,
        machine_rank,
        mixed_precision,
        debug,
    )

    mask_func = build_masking_function(**env.cfg.inference.dataset.transforms.masking)

    # TODO: Disable cropping, this must be somewhere else
    env.cfg.inference.dataset.transforms.crop = None
    env.cfg.inference.dataset.transforms.image_center_crop = False

    partial_build_mri_transforms = partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=mask_func,
    )
    mri_transforms = partial_build_mri_transforms(
        **env.cfg.inference.dataset.transforms
    )

    # Trigger cudnn benchmark when the number of different input masks_dict is small.
    torch.backends.cudnn.benchmark = True

    # TODO(jt): batches should have constant masks_dict! This works for Calgary Campinas because they are all with 256
    # slices.
    if filenames_filter:
        filenames_filter = [data_root / _ for _ in read_list(filenames_filter)]

    data = build_dataset(
        env.cfg.inference.dataset.name,
        root=data_root,
        filenames_filter=filenames_filter,
        sensitivity_maps=None,
        text_description="inference",
        kspace_context=env.cfg.inference.dataset.kspace_context,
        transforms=mri_transforms,
    )
    logger.info(f"Inference data size: {len(data)}.")

    # Just to make sure.
    torch.cuda.empty_cache()

    # Run prediction
    output = env.engine.predict(
        data,
        base_directory / run_name,
        checkpoint_number=checkpoint,
        num_workers=num_workers,
    )

    # Create output directory
    output_directory.mkdir(exist_ok=True, parents=True)

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
        if volume_processing_func:
            reconstruction = volume_processing_func(reconstruction)

        with h5py.File(output_directory / filename, "w") as f:
            f.create_dataset("reconstruction", data=reconstruction)


if __name__ == "__main__":
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a lot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    # Remove warnings from named tensors being experimental
    os.environ["PYTHONWARNINGS"] = "ignore"

    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> [--other-flags]
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument(
        "data_root", type=pathlib.Path, help="Path to the output directory."
    )
    parser.add_argument(
        "output_directory", type=pathlib.Path, help="Path to the output directory."
    )
    parser.add_argument(
        "experiment_directory",
        type=pathlib.Path,
        help="Path to the directory with checkpoints and config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Number of an existing checkpoint.",
    )
    parser.add_argument(
        "--validation-index",
        type=int,
        required=True,
        help="This is the index of the validation set in the config, e.g., 0 will select the first validation set.",
    )

    parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    parser.add_argument("--name", help="Run name.", required=True, type=str)
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for inference. "
        "Only use it to overwrite the standard loading of the config in the project directory.",
        required=False,
        type=pathlib.Path,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    volume_processing_func = None

    direct.launch.launch(
        setup_inference,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        args.name,
        args.cfg_file,
        args.data_root,
        args.experiment_directory,
        args.output_directory,
        args.filenames_filter,
        args.checkpoint,
        args.device,
        args.num_workers,
        args.machine_rank,
        volume_processing_func,
        args.mixed_precision,
        args.debug,
    )
