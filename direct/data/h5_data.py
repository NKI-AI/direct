# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
import numpy as np
import h5py

from torch.utils.data import Dataset
from typing import Dict, Callable, Optional, Any, Tuple, Union
from collections import OrderedDict

from direct.utils import cast_as_path, DirectClass
from direct.utils.io import read_json
from direct.types import PathOrString

import logging
logger = logging.getLogger(__name__)


class H5SliceData(DirectClass, Dataset):
    """
    A PyTorch Dataset class which outputs k-space slices based on the h5 dataformat.
    """

    def __init__(self, root: pathlib.Path,
                 dataset_description: Optional[Dict[PathOrString, Any]] = None,
                 metadata: Optional[Dict[PathOrString, Dict]] = None,
                 sensitivity_maps: Optional[PathOrString] = None,
                 extra_keys: Optional[Tuple] = None,
                 text_description: Optional[str] = None) -> None:
        """
        Initialize the dataset. The dataset can remove spike noise and empty slices.

        Parameters
        ----------
        root : pathlib.Path
            Root directory to data.
        metadata : dict
            If given, this dictionary will be passed to the output transform.
        sensitivity_maps : [pathlib.Path, None]
            Path to sensitivity maps, or None.
        extra_keys : Tuple
            Add extra keys in h5 file to output.
        text_description : str
            Description of dataset, can be useful for logging.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.root = pathlib.Path(root)

        self.metadata = metadata

        self.dataset_description = dataset_description
        self.text_description = text_description
        self.data = []

        self.volume_indices = OrderedDict()
        current_slice_number = 0  # This is required to keep track of where a volume is in the dataset
        if isinstance(dataset_description, (pathlib.Path, str)):
            examples = read_json(dataset_description)
            for filename in examples:
                num_slices = examples[filename]['num_slices']
                # ignore_slices = examples[filename].get('ignore_slices', [])
                # TODO: Slices can, and should be able to be ignored (for instance too many empty ones)
                ignore_slices = []
                for idx in range(num_slices):
                    if idx not in ignore_slices:
                        self.data.append((filename, idx))
                self.volume_indices[filename] = range(current_slice_number, current_slice_number + num_slices)
                current_slice_number += num_slices

        elif not dataset_description:
            self.logger.info(f'No dataset description given, parsing directory {self.root} for h5 files. '
                             f'It is recommended you create such a file, as this will speed up processing.')
            filenames = list(self.root.glob('*.h5'))
            self.logger.info(f'Using {len(filenames)} h5 files in {self.root}.')

            for idx, filename in enumerate(filenames):
                if idx % (len(filenames) // 5) or len(filenames) == (idx + 1):
                    self.logger.info(f'Parsing: {(idx + 1) / len(filenames) * 100:.2f}%.')
                try:
                    kspace = h5py.File(filename, 'r')['kspace']
                except OSError as e:
                    self.logger.warning(f'{filename} failed with OSError: {e}. Skipping...')
                    continue

                num_slices = kspace.shape[0]
                self.data += [(filename, idx) for idx in range(num_slices)]
                self.volume_indices[filename] = range(current_slice_number, current_slice_number + num_slices)
                current_slice_number += num_slices
        else:
            raise ValueError(f'Expected `Path` or `str` for `dataset_description`, got {type(dataset_description)}')

        self.sensitivity_maps = cast_as_path(sensitivity_maps)
        self.extra_keys = extra_keys

        if self.text_description:
            self.logger.info(f'Dataset description: {self.text_description}.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename, slice_no = self.data[idx]
        filename = pathlib.Path(filename)
        metadata = None if not self.metadata else self.metadata[filename.name]

        extra_data = {}

        with h5py.File(filename, 'r') as data:
            kspace = data['kspace'][slice_no]
            if self.extra_keys:
                for extra_key in self.extra_keys:
                    extra_data[extra_key] = data[extra_key][()]

        if kspace.ndim == 2:  # Singlecoil data does not always have coils at the first axis.
            kspace = kspace[np.newaxis, ...]

        sample = {'kspace': kspace,
                  'filename': filename.name,
                  'slice_no': slice_no}

        # If the sensitivity maps exist, load these
        if self.sensitivity_maps:
            with h5py.File(self.sensitivity_maps / filename.name, 'r') as sens:
                sensitivity_map = sens['sensitivity_map'][slice_no]
            sample['sensitivity_map'] = sensitivity_map

        if metadata is not None:
            sample['metadata'] = metadata

        sample.update(extra_data)

        return sample
