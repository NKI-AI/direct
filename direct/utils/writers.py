# coding=utf-8
# Copyright (c) DIRECT Contributors

import json
import logging
import pathlib
from typing import Callable, DefaultDict, Dict, Optional, Union

import h5py  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


def write_output_to_h5(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
) -> None:
    """Write dictionary with keys filenames and values torch tensors to h5 files.

    Parameters
    ----------
    output: dict
        Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
        where num_channels is typically 1 for MRI.
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