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
import functools
import logging
import os

import torch

from direct.common.subsample import build_masking_function
from direct.inference import build_inference_transforms, setup_inference_save_to_h5
from direct.launch import launch
from direct.utils import set_all_seeds

logger = logging.getLogger(__name__)


def _get_transforms(env):
    dataset_cfg = env.cfg.inference.dataset
    masking = dataset_cfg.transforms.masking  # Can be None
    mask_func = None if masking is None else build_masking_function(**masking)
    transforms = build_inference_transforms(env, mask_func, dataset_cfg)
    return dataset_cfg, transforms


setup_inference_save_to_h5 = functools.partial(
    setup_inference_save_to_h5,
    functools.partial(_get_transforms),
)


def predict_from_argparse(args: argparse.Namespace):
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a lot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    set_all_seeds(args.seed)
    experiment_directory = (
        args.experiment_directory if args.experiment_directory is not None else args.output_directory
    )

    launch(
        setup_inference_save_to_h5,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        args.name,
        args.data_root,
        experiment_directory,
        args.output_directory,
        args.filenames_filter,
        args.checkpoint,
        args.device,
        args.num_workers,
        args.machine_rank,
        args.cfg_file,
        None,
        args.mixed_precision,
        args.debug,
        False,
    )
