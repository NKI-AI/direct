# coding=utf-8
# Copyright (c) DIRECT Contributors
import h5py
import logging
import numpy as np
import pathlib
import re
from collections import OrderedDict
from torch.utils.data import Dataset
from typing import Dict, Optional, Any, Tuple, List, Union

from direct.types import PathOrString
from direct.utils import cast_as_path, DirectModule

logger = logging.getLogger(__name__)


class H5SliceData(DirectModule, Dataset):
    """
    A PyTorch Dataset class which outputs k-space slices based on the h5 dataformat.
    """

    def __init__(
        self,
        root: pathlib.Path,
        filenames_filter: Union[Optional[List[PathOrString]], None] = None,
        regex_filter: Optional[str] = None,
        dataset_description: Optional[Dict[PathOrString, Any]] = None,
        metadata: Optional[Dict[PathOrString, Dict]] = None,
        sensitivity_maps: Optional[PathOrString] = None,
        extra_keys: Optional[Tuple] = None,
        pass_attrs: bool = False,
        text_description: Optional[str] = None,
        kspace_context: Optional[int] = None,
        pass_dictionaries: Optional[Dict[str, Dict]] = None,
        pass_h5s: Optional[Dict[str, List]] = None,
        slice_data: Optional[Union[slice, bool]] = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        root : pathlib.Path
            Root directory to data.
        filenames_filter : List
            List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
            on the root. If set, will skip searching for files in the root.
        regex_filter : str
            Regular expression filter on the absolute filename. Will be applied after any filenames filter.
        metadata : dict
            If given, this dictionary will be passed to the output transform.
        sensitivity_maps : [pathlib.Path, None]
            Path to sensitivity maps, or None.
        extra_keys : Tuple
            Add extra keys in h5 file to output.
        pass_attrs : bool
            Pass the attributes saved in the h5 file.
        text_description : str
            Description of dataset, can be useful for logging.
        pass_dictionaries : dict
            Pass a dictionary of dictionaries, e.g. if {"name": {"filename_0": val}}, then to `filename_0`s sample dict,
            a key with name `name` and value `val` will be added.
        pass_h5s : dict
            Pass a dictionary of paths. If {"name": path} is given then to the sample of `filename` the same slice
            of path / filename will be added to the sample dictionary and will be asigned key `name`. This can first
            instance be convenient when you want to pass sensitivity maps as well. So for instance:

            >>> pass_h5s = {"sensitivity_map": "/data/sensitivity_maps"}

            will add to each output sample a key `sensitivity_map` with value a numpy array containing the same slice
            of /data/sensitivity_maps/filename.h5 as the one of the original filename filename.h5.
        slice_data : Optional[slice]
            If set, for instance to slice(50,-50) only data within this slide will be added to the dataset. This
            is for instance convenient in the validation set of the public Calgary-Campinas dataset as the first 50
            and last 50 slices are excluded in the evaluation.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.root = pathlib.Path(root)
        self.filenames_filter = filenames_filter

        self.metadata = metadata

        self.dataset_description = dataset_description
        self.text_description = text_description

        self.data: List[type] = []

        self.volume_indices: OrderedDict[str, int] = OrderedDict()

        if self.filenames_filter:
            self.logger.info(f"Attempting to load {len(self.filenames_filter)} "
                             f"filenames from list.")
            filenames = self.filenames_filter
        else:
            self.logger.info(f"Parsing directory {self.root} for h5 files.")
            filenames = list(self.root.glob("*.h5"))

        if regex_filter:
            filenames = [_ for _ in filenames if re.match(regex_filter, str(_))]

        self.logger.info(f"Using {len(filenames)} h5 files in {self.root}.")

        self.parse_filenames_data(
            filenames, extra_h5s=pass_h5s, filter_slice=slice_data
        )  # Collect information on the image masks_dict.
        self.pass_h5s = pass_h5s

        self.sensitivity_maps = cast_as_path(sensitivity_maps)
        self.pass_attrs = pass_attrs
        self.extra_keys = extra_keys
        self.pass_dictionaries = pass_dictionaries

        self.kspace_context = kspace_context if kspace_context else 0
        self.ndim = 2 if self.kspace_context == 0 else 3

        if self.text_description:
            self.logger.info(f"Dataset description: {self.text_description}.")

    def parse_filenames_data(self, filenames, extra_h5s=None, filter_slice=None):
        current_slice_number = 0  # This is required to keep track of where a volume is in the dataset

        for idx, filename in enumerate(filenames):
            if len(filenames) < 5 or idx % (len(filenames) // 5) == 0 or len(filenames) == (idx + 1):
                self.logger.info(f"Parsing: {(idx + 1) / len(filenames) * 100:.2f}%.")
            try:
                kspace = h5py.File(filename, "r")["kspace"]
                self.verify_extra_h5_integrity(filename, kspace.shape, extra_h5s=extra_h5s)

            except OSError as e:
                self.logger.warning(f"{filename} failed with OSError: {e}. Skipping...")
                continue

            num_slices = kspace.shape[0]
            if not filter_slice:
                self.data += [(filename, _) for _ in range(num_slices)]

            elif isinstance(filter_slice, slice):
                admissible_indices = range(*filter_slice.indices(num_slices))
                self.data += [(filename, _) for _ in range(num_slices) if _ in admissible_indices]
                num_slices = len(admissible_indices)

            else:
                raise NotImplementedError

            self.volume_indices[filename] = range(current_slice_number, current_slice_number + num_slices)

            current_slice_number += num_slices

    @staticmethod
    def verify_extra_h5_integrity(image_fn, image_shape, extra_h5s):
        if not extra_h5s:
            return

        for key in extra_h5s:
            h5_key, path = extra_h5s[key]
            extra_fn = path / image_fn.name
            try:
                with h5py.File(extra_fn, "r") as f:
                    shape = f[h5_key].shape
            except (OSError, TypeError) as e:
                raise ValueError(f"Reading of {extra_fn} for key {h5_key} failed: {e}.")

            # TODO: This is not so trivial to do it this way, as the shape depends on context
            # if image_shape != shape:
            #     raise ValueError(f"{extra_fn} and {image_fn} has different shape. "
            #                      f"Got {shape} and {image_shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename: Union[str, pathlib.Path]
        slice_no: Optional[int]

        filename, slice_no = self.data[idx]  # type: ignore
        filename = pathlib.Path(filename)
        metadata = None if not self.metadata else self.metadata[filename.name]

        kspace, extra_data = self.get_slice_data(
            filename, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys
        )

        if kspace.ndim == 2:  # Singlecoil data does not always have coils at the first axis.
            kspace = kspace[np.newaxis, ...]

        sample = {"kspace": kspace, "filename": filename.name, "slice_no": slice_no}

        # If the sensitivity maps exist, load these
        if self.sensitivity_maps:
            sensitivity_map, _ = self.get_slice_data(self.sensitivity_maps / filename.name, slice_no)
            sample["sensitivity_map"] = sensitivity_map

        if metadata is not None:
            sample["metadata"] = metadata

        sample.update(extra_data)

        if self.pass_dictionaries:
            for key in self.pass_dictionaries:
                if key in sample:
                    raise ValueError(f"Trying to add key {key} to sample dict, but this key already exists.")
                sample[key] = self.pass_dictionaries[key][filename.name]

        if self.pass_h5s:
            for key, (h5_key, path) in self.pass_h5s.items():
                curr_slice, _ = self.get_slice_data(path / filename.name, slice_no, key=h5_key)
                if key in sample:
                    raise ValueError(f"Trying to add key {key} to sample dict, but this key already exists.")
                sample[key] = curr_slice

        return sample

    def get_slice_data(self, filename, slice_no, key="kspace", pass_attrs=False, extra_keys=None):
        extra_data = {}
        if not filename.exists():
            raise OSError(f"{filename} does not exist.")

        try:
            data = h5py.File(filename, "r")
        except Exception as e:
            raise Exception(f"Reading filename {filename} caused exception: {e}")

        if self.kspace_context == 0:
            curr_data = data[key][slice_no]
        else:
            # This can be useful for getting stacks of slices.
            num_slices = self.get_num_slices(filename)
            curr_data = data[key][
                max(0, slice_no - self.kspace_context) : min(slice_no + self.kspace_context + 1, num_slices),
            ]
            curr_shape = curr_data.shape
            if curr_shape[0] < num_slices - 1:
                if slice_no - self.kspace_context < 0:
                    new_shape = list(curr_shape).copy()
                    new_shape[0] = self.kspace_context - slice_no
                    curr_data = np.concatenate(
                        [np.zeros(new_shape, dtype=curr_data.dtype), curr_data],
                        axis=0,
                    )
                if self.kspace_context + slice_no > num_slices - 1:
                    new_shape = list(curr_shape).copy()
                    new_shape[0] = slice_no + self.kspace_context - num_slices + 1
                    curr_data = np.concatenate(
                        [curr_data, np.zeros(new_shape, dtype=curr_data.dtype)],
                        axis=0,
                    )
            # Move the depth axis to the second spot.
            curr_data = np.swapaxes(curr_data, 0, 1)

        if pass_attrs:
            extra_data["attrs"] = dict(data.attrs)

        if extra_keys:
            for extra_key in self.extra_keys:
                if extra_key == "attrs":
                    raise ValueError("attrs need to be passed by setting `pass_attrs = True`.")
                extra_data[extra_key] = data[extra_key][()]
        data.close()
        return curr_data, extra_data

    def get_num_slices(self, filename):
        num_slices = self.volume_indices[filename].stop - self.volume_indices[filename].start
        return num_slices
