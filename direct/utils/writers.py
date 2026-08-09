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
from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def write_output_to_h5(
    output: tuple[list[tuple[Any, Any, pathlib.Path]], dict[str, Any]],
    output_directory: pathlib.Path,
    volume_processing_func: Callable | None = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
) -> None:
    """Write inference output to h5 files, and the aggregated metrics to a json file.

    Parameters
    ----------
    output: tuple
        Two-tuple of (volumes, metrics). The volumes are a list of (data, sampling_mask, filename) entries,
        where data is either a torch.Tensor of shape [depth, num_channels, ...], or, if a registration model
        is used, a three-tuple of (volume, registration_volume, displacement_field). The metrics are a
        dictionary with keys filenames and values the computed inference metrics.
    output_directory: pathlib.Path
    volume_processing_func: callable
        Function which postprocesses the volume array before saving.
    output_key: str
        Name of key to save the output to.
    create_dirs_if_needed: bool
        If true, the output directory and all its parents will be created.

    Notes
    -----
    Currently only num_channels = 1 is supported. If you run this function with more channels the first one
    will be used.
    """
    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)

    metrics = output[1]

    with open(output_directory / "metrics_inference.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))

    for idx, (data, sampling_mask, filename) in enumerate(output[0]):
        if isinstance(filename, pathlib.PosixPath):
            filename = filename.name

        logger.info(f"({idx + 1}/{len(output[0])}): Writing {output_directory / filename}...")

        if isinstance(data, tuple):
            volume, registration_volume, displacement_field = data
        else:
            volume = data
            registration_volume = None

        reconstruction = volume.numpy()[:, 0, ...].astype(np.float32)
        if registration_volume is not None:
            registration_volume = registration_volume.numpy()[:, 0, ...].astype(np.float32)
            displacement_field = displacement_field.numpy().astype(np.float32)

        if sampling_mask is not None:
            sampling_mask = sampling_mask.numpy()[:, 0, ...].astype(np.float32)

        if volume_processing_func:
            reconstruction = volume_processing_func(reconstruction)

        with h5py.File(output_directory / filename, "w") as f:
            f.create_dataset(output_key, data=reconstruction)
            if sampling_mask is not None:
                f.create_dataset("sampling_mask", data=sampling_mask)
            if registration_volume is not None:
                f.create_dataset("registration_volume", data=registration_volume)
                f.create_dataset("displacement_field", data=displacement_field)
