# coding=utf-8
# Copyright (c) DIRECT Contributors

import functools
import logging
import os
import pathlib
import sys

import torch

from direct.cli.utils import file_or_url
from direct.common.subsample import build_masking_function
from direct.environment import Args
from direct.inference import build_inference_transforms, setup_inference_save_to_h5
from direct.launch import launch
from direct.utils import set_all_seeds

logger = logging.getLogger(__name__)


def _get_transforms(validation_index, env):
    dataset_cfg = env.cfg.validation.datasets[validation_index]
    mask_func = build_masking_function(**dataset_cfg.transforms.masking)
    transforms = build_inference_transforms(env, mask_func, dataset_cfg)
    return dataset_cfg, transforms


if __name__ == "__main__":
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a lot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    epilog = f"""
        Examples
        --------
        Run on single machine:
            1.  $ python3 predict_val.py <output_directory>
                    --checkpoint <checkpoint_path_or_url> --cfg <cfg_file_path_or_url>
                    --data-root <val_data_root> --validation-index <validation_set_index> [--other-flags]
            OR
            2.  $ python3 predict_val.py <output_directory> --checkpoint <checkpoint_path_or_url>
                    --experiment-directory <experiment_directory_containing_config.yaml>
                    --data-root <val_data_root> --validation-index <validation_set_index> [--other-flags]

        Run on multiple machines:
            (machine0)$ {sys.argv[0]} python3 predict_val.py <output_directory>
                --checkpoint <checkpoint_path_or_url> --cfg <cfg_file_path_or_url>
                --data-root <val_data_root> --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} python3 predict_val.py <val_data_root> <output_directory>
                --checkpoint <checkpoint_path_or_url> --cfg <cfg_file_path_or_url>
                --data-root <val_data_root> --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        Notes
        -----
        * If --experiment-directory is passed and --cfg is not, then the experiment_directory should contain the
        config file (named `config.yaml`).
        * If none of --experiment-directory or --cfg are passed, then the output_directory should contain the
        config file (named `config.yaml`).
        """

    parser = Args(epilog=epilog)
    parser.add_argument("output_directory", type=pathlib.Path, help="Path to the DoIterationOutput directory.")
    parser.add_argument("--data-root", type=pathlib.Path, help="Path to the data directory.")
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=file_or_url,
        required=True,
        help="Checkpoint to a model. This can be a path to a local file or an URL. "
        "If a URL is given the checkpoint will first be downloaded to the environmental variable "
        "`DIRECT_MODEL_DOWNLOAD_DIR` (default=current directory).",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file. Can be either a local file or a remote URL."
        "Only use it to overwrite the standard loading of the config in the project directory."
        "Note that if `--cfg` is not passed, `<experiment_directory>/config.yaml` will be used as "
        "a config file, so make sure it exists.",
        required=False,
        type=file_or_url,
    )
    parser.add_argument(
        "--validation-index",
        type=int,
        required=False,
        help="This is the index of the validation set in the config, e.g., 0 will select the first validation set."
        "Default value is 0.",
        default=0,
    )
    parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    parser.add_argument(
        "--experiment-directory",
        type=pathlib.Path,
        help="Path to the directory with checkpoints and config file saved as `config.yaml`."
        "Here will also be saved the output logs. If not passed, output_directory will be used.",
        required=False,
    )
    parser.add_argument(
        "--name",
        dest="name",
        help="Run name if this is different than the config in the experiment directory.",
        required=False,
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.experiment_directory is None:
        args.experiment_directory = args.output_directory

    set_all_seeds(args.seed)

    setup_inference_save_to_h5 = functools.partial(
        setup_inference_save_to_h5,
        functools.partial(_get_transforms, args.validation_index),
    )

    launch(
        setup_inference_save_to_h5,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        args.name,
        args.data_root,
        args.experiment_directory,
        args.output_directory,
        args.filenames_filter,
        args.checkpoint,
        args.device,
        args.num_workers,
        args.machine_rank,
        args.cfg_file,
        args.mixed_precision,
        args.debug,
    )
