# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np
import pathlib

from typing import Callable, Dict, Optional, Any

from direct.data.h5_data import H5SliceData
from direct.utils import str_to_class

import logging
logger = logging.getLogger(__name__)


class FastMRIDataset(H5SliceData):
    def __init__(self,
                 root: pathlib.Path,
                 transform: Optional[Callable] = None,
                 dataset_description: Optional[Dict[Any, Any]] = None,
                 pass_mask: bool = False, **kwargs) -> None:
        super().__init__(
            root=root, dataset_description=dataset_description,
            metadata=None, extra_keys=None if not pass_mask else ('mask',), **kwargs)
        if self.sensitivity_maps is not None:
            raise NotImplementedError(f'Sensitivity maps are not supported in the current '
                                      f'{self.__class__.__name__} class.')

        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        sample['sensitivity_map'] = np.ones_like(sample['kspace'])

        if self.transform:
            sample = self.transform(sample)

        return sample


def build_datasets(dataset_name, training_root: pathlib.Path, train_sensitivity_maps=None, train_transforms=None,
                   validation_root=None, val_sensitivity_maps=None, val_transforms=None):
    logger.info(f'Building dataset for {dataset_name}.')
    dataset_class: Callable = str_to_class('direct.data.datasets', dataset_name + 'Dataset')

    train_data = dataset_class(
        root=training_root,
        dataset_description=None,
        transform=train_transforms,
        sensitivity_maps=train_sensitivity_maps,
        pass_mask=False)
    logger.info(f'Train data size: {len(train_data)}.')

    if validation_root:
        val_data = dataset_class(
            root=validation_root,
            dataset_description=None,
            transform=val_transforms,
            sensitivity_maps=val_sensitivity_maps,
            pass_mask=False)

        logger.info(f'Validation data size: {len(val_data)}.')

        return train_data, val_data

    return train_data
