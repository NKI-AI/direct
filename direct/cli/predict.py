# coding=utf-8
# Copyright (c) DIRECT Contributors

import argparse
import pathlib

from direct.cli.utils import file_or_url
from direct.environment import Args
from direct.predict import predict_from_argparse


def register_parser(parser: argparse._SubParsersAction):
    """Register wsi commands to a root parser."""

    epilog = f"""
        Examples:
        ---------
        Run on single machine:
            $ direct predict <data_root> <output_directory> <experiment_directory> --checkpoint <checkpoint> \
                            --num-gpus <num_gpus> [ --cfg <cfg_filename>.yaml --other-flags <other_flags>]

        Run on multiple machines:
            (machine0)$ direct predict <data_root> <output_directory> <experiment_dir> --checkpoint <checkpoint> \
                            --cfg <cfg_filename>.yaml --machine-rank 0 --num-machines 2 [--other-flags]
            (machine1)$ direct predict <data_root> <output_directory> <experiment_dir> --checkpoint <checkpoint> \
                            --cfg <cfg_filename>.yaml --machine-rank 1 --num-machines 2 [--other-flags]
        """
    common_parser = Args(add_help=False)
    predict_parser = parser.add_parser(
        "predict",
        help="Run inference using direct.",
        parents=[common_parser],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument("data_root", type=pathlib.Path, help="Path to the inference data directory.")
    predict_parser.add_argument("output_directory", type=pathlib.Path, help="Path to the output directory.")
    predict_parser.add_argument(
        "experiment_directory",
        type=pathlib.Path,
        help="Path to the directory with checkpoints and config.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Number of an existing checkpoint in experiment directory.",
    )
    predict_parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    predict_parser.add_argument(
        "--name",
        dest="name",
        help="Run name if this is different than the experiment directory.",
        required=False,
        type=str,
        default="",
    )
    predict_parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for inference. Can be either a local file or a remote URL."
        "Only use it to overwrite the standard loading of the config in the project directory.",
        required=False,
        type=file_or_url,
    )

    predict_parser.set_defaults(subcommand=predict_from_argparse)
