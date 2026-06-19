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
"""DIRECT Command-line interface.

This is the file which builds the main parser.
"""

import argparse
import sys


def main():
    """Console script for direct."""
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Direct CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular imports
    from direct.cli.predict import register_parser as register_predict_subcommand
    from direct.cli.train import register_parser as register_train_subcommand
    from direct.cli.upload import register_parser as register_upload_subcommand

    # Training images related commands.
    register_train_subcommand(root_subparsers)
    # Inference images related commands.
    register_predict_subcommand(root_subparsers)
    # Data related comments.
    register_upload_subcommand(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    sys.exit(main())
