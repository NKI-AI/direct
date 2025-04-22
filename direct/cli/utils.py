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
import argparse
import pathlib
import sys

from direct.types import FileOrUrl, PathOrString
from direct.utils.io import check_is_valid_url


def is_file(path):
    path = pathlib.Path(path)
    if path.is_file():
        return path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file or url.")


def file_or_url(path: PathOrString) -> FileOrUrl:
    if check_is_valid_url(path):
        return FileOrUrl(path)
    path = pathlib.Path(path)
    if path.is_file():
        return FileOrUrl(path)
    raise argparse.ArgumentTypeError(f"{path} is not a valid file or url.")


def check_train_val(key, name):
    if key is not None and len(key) != 2:
        sys.exit(f"--{name} has to be of the form `train_folder, validation_folder` if a validation folder is set.")
