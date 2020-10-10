# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
import numpy as np
import h5py
import warnings

from torch.utils.data import Dataset
from typing import Dict, Optional, Any, Tuple, List
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

    def __init__(
        self,
        root: pathlib.Path,
        filenames_filter: Optional[List[PathOrString]] = None,
        dataset_description: Optional[Dict[PathOrString, Any]] = None,
        metadata: Optional[Dict[PathOrString, Dict]] = None,
        sensitivity_maps: Optional[PathOrString] = None,
        extra_keys: Optional[Tuple] = None,
        pass_attrs: bool = False,
        text_description: Optional[str] = None,
        kspace_context: Optional[int] = None,
        pass_dictionaries: Optional[Dict[str, Dict]] = None,
        pass_h5s: Optional[Dict[str, List]] = None,
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
            Pass a list of dictionaries, e.g. if {"name": {"filename_0": val}}, then to `filename_0`s sample dict, a key
            with name `name` and value `val` will be added.
        pass_h5s : dict
            # TODO Improve description
            Pass a list of dictionaries, e.g. if {"name": path}, then path will be parsed and this key added.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.root = pathlib.Path(root)
        self.filenames_filter = filenames_filter

        self.metadata = metadata

        self.dataset_description = dataset_description
        self.text_description = text_description

        self.data = []

        self.volume_indices = OrderedDict()

        if self.filenames_filter:
            self.logger.info(
                f"Attempting to load {len(filenames_filter)} filenames from list."
            )
            filenames = filenames_filter
        else:
            self.logger.info(
                f"No dataset description given, parsing directory {self.root} for h5 files. "
                f"It is recommended you create such a file, as this will speed up processing."
            )
            filenames = list(self.root.glob("*.h5"))
        self.logger.info(f"Using {len(filenames)} h5 files in {self.root}.")

        self.parse_filenames_data(filenames, extra_h5s=pass_h5s)  # Collect information on the image masks_dict.
        self.pass_h5s = pass_h5s

        self.sensitivity_maps = cast_as_path(sensitivity_maps)
        self.pass_attrs = pass_attrs
        self.extra_keys = extra_keys
        self.pass_dictionaries = pass_dictionaries

        self.kspace_context = kspace_context if kspace_context else 0
        self.ndim = 2 if self.kspace_context == 0 else 3

        if self.text_description:
            self.logger.info(f"Dataset description: {self.text_description}.")

    def parse_filenames_data(self, filenames, extra_h5s=None):
        current_slice_number = (
            0  # This is required to keep track of where a volume is in the dataset
        )

        for idx, filename in enumerate(filenames):
            if (
                len(filenames) < 5
                or idx % (len(filenames) // 5) == 0
                or len(filenames) == (idx + 1)
            ):
                self.logger.info(f"Parsing: {(idx + 1) / len(filenames) * 100:.2f}%.")
            try:
                kspace = h5py.File(filename, "r")["kspace"]
                self.verify_extra_h5_integrity(filename, kspace.shape, extra_h5s=extra_h5s)

            except OSError as e:
                self.logger.warning(f"{filename} failed with OSError: {e}. Skipping...")
                continue

            num_slices = kspace.shape[0]
            self.data += [(filename, idx) for idx in range(num_slices)]
            self.volume_indices[filename] = range(
                current_slice_number, current_slice_number + num_slices
            )
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
        filename, slice_no = self.data[idx]
        filename = pathlib.Path(filename)
        metadata = None if not self.metadata else self.metadata[filename.name]

        kspace, extra_data = self.get_slice_data(
            filename, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys
        )

        if (
            kspace.ndim == 2
        ):  # Singlecoil data does not always have coils at the first axis.
            kspace = kspace[np.newaxis, ...]

        sample = {"kspace": kspace, "filename": filename.name, "slice_no": slice_no}

        # If the sensitivity maps exist, load these
        if self.sensitivity_maps:
            sensitivity_map, _ = self.get_slice_data(
                self.sensitivity_maps / filename.name, slice_no
            )
            sample["sensitivity_map"] = sensitivity_map

        if metadata is not None:
            sample["metadata"] = metadata

        sample.update(extra_data)

        if self.pass_dictionaries:
            for key, value in self.pass_dictionaries.values():
                if key in sample:
                    raise ValueError(
                        f"Trying to add key {key} to sample dict, but this key already exists."
                    )
                sample[key] = value[filename]

        if self.pass_h5s:
            for key, (h5_key, path) in self.pass_h5s.items():
                curr_slice, _ = self.get_slice_data(path / filename.name, slice_no, key=h5_key)
                if key in sample:
                    raise ValueError(
                        f"Trying to add key {key} to sample dict, but this key already exists."
                    )
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
                max(0, slice_no - self.kspace_context) : min(
                    slice_no + self.kspace_context + 1, num_slices
                ),
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
                    raise ValueError(
                        f"attrs need to be passed by setting `pass_attrs = True`."
                    )
                extra_data[extra_key] = data[extra_key][()]
        data.close()
        return curr_data, extra_data

    def get_num_slices(self, filename):
        num_slices = (
            self.volume_indices[filename].stop - self.volume_indices[filename].start
        )
        return num_slices
