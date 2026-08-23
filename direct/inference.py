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
"""direct.inference module."""

import logging
import pathlib
import sys
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from omegaconf import DictConfig

from direct.data.datasets import build_dataset_from_input
from direct.data.mri_transforms import build_mri_transforms
from direct.environment import setup_inference_environment
from direct.types import FileOrUrl, PathOrString
from direct.utils import chunks, dict_flatten, remove_keys
from direct.utils.io import read_list
from direct.utils.writers import write_output_to_h5

logger = logging.getLogger(__name__)


def setup_inference_save_to_h5(
    get_inference_settings: Callable,
    run_name: str,
    data_root: PathOrString | None,
    base_directory: PathOrString,
    output_directory: PathOrString,
    filenames_filter: list[PathOrString] | None,
    checkpoint: FileOrUrl,
    device: str,
    num_workers: int,
    machine_rank: int,
    cfg_file: PathOrString | None = None,
    process_per_chunk: int | None = None,
    mixed_precision: bool = False,
    debug: bool = False,
    is_validation: bool = False,
) -> None:
    """This function contains most of the logic in DIRECT required to launch a multi-gpu / multi-node inference process.

    It saves predictions as `.h5` files.

    Args:
        get_inference_settings: Callable object to create inference dataset and environment.
        run_name: Experiment run name. Can be an empty string.
        data_root: Path of the directory of the data if applicable for dataset. Can be None.
        base_directory: Path to directory where where inference logs will be stored. If `run_name` is not an empty string,
            `base_directory / run_name` will be used.
        output_directory: Path to directory where output data will be saved.
        filenames_filter: List of filenames to include in the dataset (if applicable). Can be None.
        checkpoint: Checkpoint to a model. This can be a path to a local file or an URL.
        device: Device name.
        num_workers: Number of workers.
        machine_rank: Machine rank.
        cfg_file: Path to configuration file. If None, will search in `base_directory`.
        process_per_chunk: Processes per chunk number.
        mixed_precision: If True, mixed precision will be allowed. Default is ``False``.
        debug: If True, debug information will be displayed. Default is ``False``.
        is_validation: If True, will use settings (e.g. `batch_size` & `crop`) of `validation` in config. Otherwise it will
            use `inference` settings. Default is ``False``.

    Returns:
        The result.
    """
    env = setup_inference_environment(
        run_name, pathlib.Path(base_directory), device, machine_rank, mixed_precision, cfg_file, debug=debug
    )

    dataset_cfg, transforms = get_inference_settings(env)

    # Trigger cudnn benchmark when the number of different input masks_dict is small.
    torch.backends.cudnn.benchmark = True
    if data_root and filenames_filter:
        filenames_filter = [data_root / _ for _ in read_list(filenames_filter)]

    filenames_chunks: list[Any]
    if not process_per_chunk:
        filenames_chunks = [filenames_filter]
    else:
        filenames_chunks = list(chunks(filenames_filter or [], process_per_chunk))

    logger.info(f"Predicting dataset and saving in: {output_directory}.")

    if is_validation:
        batch_size, crop = env.cfg.validation.batch_size, env.cfg.validation.crop  # type: ignore
    else:
        batch_size, crop = env.cfg.inference.batch_size, env.cfg.inference.crop  # type: ignore

    for curr_filenames_filter in filenames_chunks:
        output = inference_on_environment(
            env=env,
            data_root=data_root,
            dataset_cfg=dataset_cfg,
            transforms=transforms,
            experiment_path=pathlib.Path(base_directory) / run_name,
            checkpoint=checkpoint,
            num_workers=num_workers,
            filenames_filter=curr_filenames_filter,
            batch_size=batch_size,
            crop=crop,
        )

        # Perhaps aggregation to the main process would be most optimal here before writing.
        # The current way this write the volumes for each process.
        write_output_to_h5(
            output,  # ty: ignore[invalid-argument-type]
            pathlib.Path(output_directory),
            output_key="reconstruction",
        )


def build_inference_transforms(env, mask_func: Callable | None, dataset_cfg: DictConfig) -> Any:
    """Builds inference transforms."""
    partial_build_mri_transforms = partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=mask_func,
    )
    transforms = partial_build_mri_transforms(**dict_flatten(remove_keys(dataset_cfg.transforms, "masking")))  # ty: ignore[invalid-argument-type]
    return transforms


def inference_on_environment(
    env,
    data_root: PathOrString | None,
    dataset_cfg: DictConfig,
    transforms: Callable,
    experiment_path: PathOrString,
    checkpoint: FileOrUrl,
    num_workers: int = 0,
    filenames_filter: list[PathOrString] | None = None,
    batch_size: int = 1,
    crop: str | None = None,
) -> dict | defaultdict:
    """Performs inference on environment.

    Args:
        env: Env.
        data_root: Path of the directory of the data if applicable for dataset. Can be None.
        dataset_cfg: Configuration containing inference dataset settings.
        transforms: Dataset transformations object.
        experiment_path: Path to directory where where inference logs will be stored.
        checkpoint: Checkpoint to a model. This can be a path to a local file or an URL.
        num_workers: Number of workers.
        filenames_filter: List of filenames to include in the dataset (if applicable). Can be None. Default is ``None``.
        batch_size: Inference batch size. Default is ``1``.
        crop: Inference crop type. Can be `header` or None. Default is ``None``.

    Returns:
        The result.
    """

    logger.warning("pass_h5s and pass_dictionaries is not yet supported for inference.")

    kwargs = {}
    if data_root is not None:
        kwargs.update({"data_root": data_root})
        if filenames_filter:
            kwargs.update({"filenames_filter": filenames_filter})

    dataset = build_dataset_from_input(transforms=transforms, dataset_config=dataset_cfg, **kwargs)

    if len(dataset) <= 0:  # ty: ignore[invalid-argument-type]
        logger.info("Inference dataset is empty. Terminating inference...")
        sys.exit(-1)

    logger.info("Inference data size: %s.", len(dataset))  # ty: ignore[invalid-argument-type]

    # Run prediction
    output = env.engine.predict(
        dataset,
        experiment_path,
        checkpoint=checkpoint,
        num_workers=num_workers,
        batch_size=batch_size,
        crop=crop,
    )
    return output
