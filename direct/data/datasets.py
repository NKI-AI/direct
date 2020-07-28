# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np
import pathlib
import warnings

from typing import Callable, Dict, Optional, Any

from direct.data.h5_data import H5SliceData
from direct.utils import str_to_class

from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)


class FastMRIDataset(H5SliceData):
    def __init__(self,
                 root: pathlib.Path,
                 transform: Optional[Callable] = None,
                 dataset_description: Optional[Dict[Any, Any]] = None,
                 pass_mask: bool = False,
                 pass_header: bool = False, **kwargs) -> None:

        extra_keys = ['mask'] if pass_mask else []
        self.pass_header = pass_header
        if pass_header:
            try:
                import ismrmd
            except ImportError:
                raise ImportError(f'ISMRMD Library not available. Will not be able to parse ISMRMD headers. '
                                  f'Install pyxb and ismrmrd-python from https://github.com/ismrmrd/ismrmrd-python '
                                  f'if you wish to parse the headers.')

            extra_keys.append('ismrmrd_header')

        super().__init__(
            root=root,
            dataset_description=dataset_description,
            metadata=None,
            extra_keys=tuple(extra_keys),
            *kwargs)
        if self.sensitivity_maps is not None:
            raise NotImplementedError(f'Sensitivity maps are not supported in the current '
                                      f'{self.__class__.__name__} class.')

        self.transform = transform

        self.__header_cache = {}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        if self.pass_header:
            sample.update(self.parse_header(idx, sample['ismrmd_header']))
            del sample['ismrmrd_header']

        if self.transform:
            sample = self.transform(sample)

        return sample

    def parse_header(self, idx, xml_header):
        if idx in self.__header_cache:
            return self.__header_cache[idx]
        else:
            header = ismrmrd.xsd.CreateFromDocument(xml_header) # noqa
            raise NotImplementedError('Parsing FastMRI headers are not yet implemented. '
                                      'Acquisition parameters can be obtained from the header.')


class CalgaryCampinasDataset(H5SliceData):
    def __init__(self,
                 root: pathlib.Path,
                 transform: Optional[Callable] = None,
                 dataset_description: Optional[Dict[Any, Any]] = None,
                 pass_mask: bool = False, **kwargs) -> None:
        super().__init__(
            root=root, dataset_description=dataset_description,
            metadata=None, extra_keys=None, **kwargs)

        if self.sensitivity_maps is not None:
            raise NotImplementedError(f'Sensitivity maps are not supported in the current '
                                      f'{self.__class__.__name__} class.')

        # Sampling rate in the slice-encode direction
        self.sampling_rate_slice_encode: float = 0.85
        self.transform = transform
        self.pass_mask: bool = pass_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        kspace = sample['kspace']

        # TODO: use broadcasting function.
        if self.pass_mask:
            # # In case the data is already masked, the sampling mask can be recovered by finding the zeros.
            # This needs to be done in the primary function!
            # sampling_mask = ~(np.abs(kspace).sum(axis=(0, -1)) == 0)
            sample['mask'] = (sample['mask'] * np.ones(kspace.shape).astype(np.int32))[..., np.newaxis]

        kspace = kspace[..., ::2] + 1j * kspace[..., 1::2]   # Convert real-valued to complex-valued data.
        num_z = kspace.shape[1]
        kspace[:, int(np.ceil(num_z * self.sampling_rate_slice_encode)):, :] = 0. + 0. * 1j

        # Downstream code expects the coils to be at the first axis.
        # TODO: When named tensor support is more solid, this could be circumvented.
        sample['kspace'] = np.ascontiguousarray(kspace.transpose(2, 0, 1))

        if self.transform:
            sample = self.transform(sample)
        return sample


def build_dataset(
        dataset_name,
        root: pathlib.Path,
        sensitivity_maps: Optional[pathlib.Path] = None,
        transforms=None,
        text_description=None, **kwargs) -> Dataset:
    """

    Parameters
    ----------
    dataset_name : str
        Name of dataset class (without `Dataset`) in direct.data.datasets.
    root : pathlib.Path
        Root path to the data for the dataset class.
    sensitivity_maps : pathlib.Path
        Path to sensitivity maps.
    transforms : object
        Transformation object
    text_description : str
        Description of dataset, can be used for logging.

    Returns
    -------
    Dataset
    """

    logger.info(f'Building dataset for {dataset_name}.')
    dataset_class: Callable = str_to_class('direct.data.datasets', dataset_name + 'Dataset')
    logger.debug(f'Dataset class: {dataset_class}.')

    dataset = dataset_class(
        root=root,
        dataset_description=None,
        transform=transforms,
        sensitivity_maps=sensitivity_maps,
        pass_mask=False,
        text_description=text_description, **kwargs)

    logger.debug(f'Dataset:\n{dataset}')

    return dataset
