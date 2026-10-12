# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI for magnitude → phase + sensitivity map → synthetic multicoil k-space."""

from __future__ import annotations

import argparse
import pathlib

from direct.cli.utils import file_or_url
from direct.environment import Args
from direct.synthesize import synthesize_from_argparse


def register_parser(parser: argparse._SubParsersAction) -> None:
    """Register ``direct synthesize``."""
    epilog = """
        Examples:
        ---------
            $ direct synthesize <output_directory> \\
                --cfg synthesize.yaml \\
                --phase-model <flow_matching checkpoint>.pt \\
                --sensitivity-model <sensitivity_maps checkpoint>.pt \\
                --data-root <fastmri data root> \\
                --device cuda --num-gpus 1
        """
    common_parser = Args(add_help=False)
    synth_parser = parser.add_parser(
        "synthesize",
        help="Synthesize multicoil k-space from magnitude using phase + sensitivity models.",
        parents=[common_parser],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    synth_parser.add_argument("output_directory", type=pathlib.Path, help="Path to the output directory.")
    synth_parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        required=True,
        help="Path to input data (FastMRI k-space volumes or magnitude-compatible root).",
    )
    synth_parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=file_or_url,
        required=True,
        help="Config with transforms, model architectures, and synthesize.* options.",
    )
    synth_parser.add_argument(
        "--phase-model",
        dest="phase_model",
        type=file_or_url,
        required=True,
        help="Checkpoint for the phase synthesizer (e.g. PhaseFlowMatching).",
    )
    synth_parser.add_argument(
        "--sensitivity-model",
        dest="sensitivity_model",
        type=file_or_url,
        required=True,
        help="Checkpoint for SensitivityMapSynthesizer (complex maps, RSS-normalized).",
    )
    synth_parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    synth_parser.add_argument(
        "--experiment_directory",
        dest="experiment_directory",
        type=pathlib.Path,
        required=False,
        help="Optional log directory; defaults to output_directory.",
    )
    synth_parser.add_argument(
        "--name",
        dest="name",
        type=str,
        default="synthesize",
        help="Run name under experiment_directory.",
    )
    synth_parser.set_defaults(subcommand=synthesize_from_argparse)
