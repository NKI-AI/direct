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
"""direct.utils.dataset module."""

from __future__ import annotations

import pathlib
import re
import urllib.parse
from typing import Any

import numpy as np

from direct.types import PathOrString
from direct.utils.io import check_is_valid_url, read_list

# Matches tokens such as ``15T``, ``30T``, ``055T``, ``70T`` in filenames
# (e.g. CMRxRecon2025: ``..._UIH_15T_umr670_...``).
_FIELD_STRENGTH_TOKEN = re.compile(r"(?:^|[^0-9A-Za-z])(\d+)T(?:[^0-9A-Za-z]|$)", re.IGNORECASE)


def parse_field_strength_tesla(filename: str | pathlib.Path) -> float | None:
    """Parse magnetic field strength in Tesla from an ``XT`` token in ``filename``.

    Digits before ``T`` are interpreted with a decimal point before the last digit when
    there is more than one digit:

    * ``7T`` / ``70T`` → 7.0 T
    * ``15T`` → 1.5 T
    * ``30T`` → 3.0 T
    * ``055T`` → 0.55 T

    Args:
        filename: Path or basename that may contain a field-strength token.

    Returns:
        Field strength in Tesla, or ``None`` if no ``XT`` token is present.
    """
    name = pathlib.Path(filename).name
    matches = _FIELD_STRENGTH_TOKEN.findall(name)
    if not matches:
        return None

    digits = matches[-1]
    if not digits or int(digits) == 0:
        return None

    return float(int(digits)) / float(10 ** max(len(digits) - 1, 0))


def maybe_attach_field_strength(sample: dict[str, Any]) -> dict[str, Any]:
    """Attach ``field_strength`` to ``sample`` when the filename encodes an ``XT`` token.

    If no token is found, ``sample`` is left unchanged (key is not added).

    Args:
        sample: Dataset sample; expects optional ``filename`` key.

    Returns:
        The same sample dict, possibly with ``field_strength`` as a length-1 ``np.ndarray``.
    """
    value = parse_field_strength_tesla(sample.get("filename", ""))
    if value is not None:
        sample["field_strength"] = np.array([value], dtype=np.float64)
    return sample


def get_filenames_for_datasets_from_config(cfg, files_root: PathOrString, data_root: PathOrString):
    """Given a configuration object it returns a list of filenames.

    Args:
        cfg: cfg-object cfg object having property lists having the relative paths compared to files root.
        files_root: Files root.
        data_root: Data root.

    Returns:
        The result.
    """
    if "filenames_lists" not in cfg:
        return None
    lists = cfg.filenames_lists
    return get_filenames_for_datasets(lists, files_root, data_root)


def get_filenames_for_datasets(lists: list[PathOrString], files_root: PathOrString, data_root: PathOrString):
    """Given lists of filenames of data points, concatenate these into a large list of full filenames.

    Args:
        lists: Lists.
        files_root: Files root.
        data_root: Data root.

    Returns:
        The result.
    """
    # Build the path, know that files_root can also be a URL
    is_url = check_is_valid_url(files_root)

    filter_filenames = []
    for curr_list in lists:
        if not is_url:
            path_to_list = pathlib.Path(files_root) / curr_list
        else:
            # The path needs to be extended / and '...' needs to be parsed. The urljoin handles this correctly
            # Note: any query arguments are dropped. So any temporary keys such as ?Q=XYZ will not be added to the URL.
            path_to_list = urllib.parse.urljoin(str(files_root), str(curr_list))

        filter_filenames += [pathlib.Path(data_root) / pathlib.Path(_) for _ in read_list(path_to_list)]

    return filter_filenames
