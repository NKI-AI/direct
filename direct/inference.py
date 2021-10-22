# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import sys
from functools import partial
from typing import Callable, Optional

import torch

from direct.data.datasets import build_dataset_from_input
from direct.data.mri_transforms import build_mri_transforms
from direct.environment import setup_inference_environment
from direct.utils import chunks, remove_keys
from direct.utils.io import read_list
from direct.utils.writers import write_output_to_h5

logger = logging.getLogger(__name__)


def setup_inference_save_to_h5(
    get_inference_settings,
    run_name,
    data_root,
    base_directory,
    output_directory,
    filenames_filter,
    checkpoint,
    device,
    num_workers: int,
    machine_rank: int,
    process_per_chunk: Optional[int] = None,
    volume_processing_func: Callable = None,
    mixed_precision: bool = False,
    debug: bool = False,
):
    """
    This function contains most of the logic in DIRECT required to launch a multi-gpu / multi-node inference process.

    Parameters
    ----------
    get_inference_settings : Callable
    run_name :
    data_root :
    base_directory :
    output_directory :
    filenames_filter :
    checkpoint :
    device :
    num_workers :
    machine_rank :
    process_per_chunk :
    volume_processing_func :
    mixed_precision :
    debug :

    Returns
    -------
    None
    """
    env = setup_inference_environment(run_name, base_directory, device, machine_rank, mixed_precision, debug=debug)

    if env.cfg.checkpoint.checkpoint_url is not None:
        try:
            if checkpoint is not None:
                logger.warning(f"'checkpoint_url' is not null. This will ignore checkpoint value {checkpoint}.")

            checkpoint = "download"
            logger.info(
                f"Attempting to download checkpoint from {env.cfg.checkpoint.checkpoint_url} "
                f"as {f'model_{checkpoint}.pt'}."
            )
            torch.hub.load_state_dict_from_url(
                env.cfg.checkpoint.checkpoint_url,
                model_dir=base_directory,
                file_name=f"model_{checkpoint}.pt",
                progress=False,
            )
            logger.info(
                f"Successfully downloaded checkpoint from {env.cfg.checkpoint.checkpoint_url}. "
                f"Saved temporarily to {base_directory}."
            )
        except Exception as exc:
            logger.info(
                f"Could not download checkpoint from {env.cfg.checkpoint.checkpoint_url}. Make sure that"
                f"the url contains a valid torch state_dict. Exiting with error message: {exc}"
            )
            sys.exit(-1)

    dataset_cfg, transforms = get_inference_settings(env)

    # Trigger cudnn benchmark when the number of different input masks_dict is small.
    torch.backends.cudnn.benchmark = True

    if filenames_filter:
        filenames_filter = [data_root / _ for _ in read_list(filenames_filter)]

    if not process_per_chunk:
        filenames_filter = [filenames_filter]
    else:
        filenames_filter = list(chunks(filenames_filter, process_per_chunk))

    logger.info(f"Predicting dataset and saving in: {output_directory}.")
    for curr_filenames_filter in filenames_filter:
        output = inference_on_environment(
            env=env,
            data_root=data_root,
            dataset_cfg=dataset_cfg,
            transforms=transforms,
            experiment_path=base_directory / run_name,
            checkpoint=checkpoint,
            num_workers=num_workers,
            filenames_filter=curr_filenames_filter,
        )

        if env.cfg.checkpoint.checkpoint_url is not None:
            import os
            # Delete downloaded checkpoint.
            logger.info(f"Removing {f'model_{checkpoint}.pt'} from {base_directory}...")
            os.remove(base_directory / f"model_{checkpoint}.pt")

        # Perhaps aggregation to the main process would be most optimal here before writing.
        # The current way this write the volumes for each process.
        write_output_to_h5(
            output,
            output_directory,
            volume_processing_func,
            output_key="reconstruction",
        )


def build_inference_transforms(env, mask_func, dataset_cfg):
    partial_build_mri_transforms = partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=mask_func,
    )
    transforms = partial_build_mri_transforms(**remove_keys(dataset_cfg.transforms, "masking"))
    return transforms


def inference_on_environment(
    env,
    data_root,
    dataset_cfg,
    transforms,
    experiment_path,
    checkpoint,
    num_workers=0,
    filenames_filter=None,
):

    logger.warning("pass_h5s and pass_dictionaries is not yet supported for inference.")

    initial_images = None
    initial_kspaces = None
    pass_dictionaries = None

    dataset = build_dataset_from_input(
        transforms,
        dataset_cfg,
        initial_images,
        initial_kspaces,
        filenames_filter,
        data_root,
        pass_dictionaries,
    )

    if len(dataset) <= 0:
        logger.info("Inference dataset is empty. Terminating inference...")
        sys.exit(-1)

    logger.info(f"Inference data size: {len(dataset)}.")

    # Run prediction
    output = env.engine.predict(
        dataset,
        experiment_path,
        checkpoint_number=checkpoint,
        num_workers=num_workers,
    )
    return output
