# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np
import pathlib
import bisect

from typing import Callable, Dict, Optional, Any, List
from functools import lru_cache

from direct.data.h5_data import H5SliceData
from direct.utils import str_to_class, remove_keys
from direct.types import PathOrString


from torch.utils.data import Dataset, IterableDataset


try:
    import ismrmrd
except ImportError:
    raise ImportError(
        f"ISMRMD Library not available. Will not be able to parse ISMRMD headers. "
        f"Install pyxb and ismrmrd-python from https://github.com/ismrmrd/ismrmrd-python "
        f"if you wish to parse the headers."
    )

import logging

logger = logging.getLogger(__name__)


class FastMRIDataset(H5SliceData):
    def __init__(
        self,
        root: pathlib.Path,
        transform: Optional[Callable] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        regex_filter: Optional[str] = None,
        pass_mask: bool = False,
        pass_max: bool = True,
        initial_images: Optional[List[pathlib.Path]] = None,
        initial_images_key: Optional[str] = None,
        noise_data: Optional[Dict] = None,
        pass_h5s: Optional[Dict] = None,
        **kwargs,
    ) -> None:

        # TODO: Clean up Dataset class such that only **kwargs need to get parsed.
        # BODY: Additional keysneeded for this dataset can be popped if needed.
        self.pass_mask = pass_mask
        extra_keys = ["mask"] if pass_mask else []
        extra_keys.append("ismrmrd_header")

        super().__init__(
            root=root,
            filenames_filter=filenames_filter,
            regex_filter=regex_filter,
            metadata=None,
            extra_keys=tuple(extra_keys),
            pass_attrs=pass_max,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )
        if self.sensitivity_maps is not None:
            raise NotImplementedError(
                f"Sensitivity maps are not supported in the current "
                f"{self.__class__.__name__} class."
            )

        # TODO: Make exclusive or to give error when one of the two keys is not set.
        # TODO: Convert into mixin, and add support to main image
        # TODO: Such a support would also work for the sensitivity maps
        self.initial_images = initial_images
        self.initial_images_key = initial_images_key

        if self.initial_images:
            self.initial_images = {k.name: k for k in self.initial_images}

        self.noise_data = noise_data

        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        if self.pass_attrs:
            sample["scaling_factor"] = sample["attrs"]["max"]
            del sample["attrs"]

        sample.update(self.parse_header(sample["ismrmrd_header"]))
        del sample["ismrmrd_header"]
        # Some images have strange behavior.
        image_shape = sample["kspace"].shape
        if (
            image_shape[-1] < sample["reconstruction_size"][-2]
        ):  # reconstruction size is (x, y, z)
            sample["reconstruction_size"] = (image_shape[-1], image_shape[-1], 1)

        if self.pass_mask:
            # mask should be shape (1, h, w, 1) mask provided is only w
            kspace_shape = sample["kspace"].shape
            sampling_mask = sample["mask"]

            # Mask needs to be padded.
            sampling_mask[: sample["padding_left"]] = 0
            sampling_mask[sample["padding_right"] :] = 0

            sampling_mask = sampling_mask.reshape(1, -1)
            del sample["mask"]

            sample["sampling_mask"] = self.__broadcast_mask(kspace_shape, sampling_mask)
            sample["acs_mask"] = self.__broadcast_mask(
                kspace_shape, self.__get_acs_from_fastmri_mask(sampling_mask)
            )

        # Explicitly zero-out the outer parts of kspace which are padded
        sample["kspace"] = self.explicit_zero_padding(
            sample["kspace"], sample["padding_left"], sample["padding_right"]
        )

        if self.transform:
            sample = self.transform(sample)

        if self.noise_data:
            sample["loglikelihood_scaling"] = self.noise_data[sample["slice_no"]]

        return sample

    @staticmethod
    def explicit_zero_padding(kspace, padding_left, padding_right):
        if padding_left > 0:
            kspace[..., 0:padding_left] = 0 + 0 * 1j
        if padding_right > 0:
            kspace[..., padding_right:] = 0 + 0 * 1j

        return kspace

    @staticmethod
    def __get_acs_from_fastmri_mask(mask):
        l = r = mask.shape[-1] // 2
        while mask[..., r]:
            r += 1
        while mask[..., l]:
            l -= 1
        acs_mask = np.zeros_like(mask)
        acs_mask[:, l + 1 : r] = 1
        return acs_mask

    def __broadcast_mask(self, kspace_shape, mask):
        if self.ndim == 2:
            mask = np.broadcast_to(mask, [kspace_shape[1], mask.shape[-1]])
            mask = mask[np.newaxis, ..., np.newaxis]
        elif self.ndim == 3:
            mask = np.broadcast_to(mask, [kspace_shape[2], mask.shape[-1]])
            mask = mask[np.newaxis, np.newaxis, ..., np.newaxis]
        return mask

    @lru_cache(maxsize=None)
    def parse_header(self, xml_header):
        # Borrowed from: https://github.com/facebookresearch/fastMRI/blob/57c0a9ef52924d1ffb30d7b7a51d022927b04b23/fastmri/data/mri_data.py#L136
        header = ismrmrd.xsd.CreateFromDocument(xml_header)  # noqa
        encoding = header.encoding[0]

        encoding_size = (
            encoding.encodedSpace.matrixSize.x,
            encoding.encodedSpace.matrixSize.y,
            encoding.encodedSpace.matrixSize.z,
        )
        reconstruction_size = (
            encoding.reconSpace.matrixSize.x,
            encoding.reconSpace.matrixSize.y,
            encoding.reconSpace.matrixSize.z,
        )
        encoding_limits_center = encoding.encodingLimits.kspace_encoding_step_1.center
        encoding_limits_max = encoding.encodingLimits.kspace_encoding_step_1.maximum + 1
        padding_left = encoding_size[1] // 2 - encoding_limits_center
        padding_right = padding_left + encoding_limits_max

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": encoding_size,
            "reconstruction_size": reconstruction_size,
        }
        return metadata


class CalgaryCampinasDataset(H5SliceData):
    def __init__(
        self,
        root: pathlib.Path,
        transform: Optional[Callable] = None,
        regex_filter: Optional[str] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        pass_mask: bool = False,
        crop_outer_slices: bool = False,
        pass_h5s: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            filenames_filter=filenames_filter,
            regex_filter=regex_filter,
            metadata=None,
            extra_keys=None,
            slice_data=slice(50, -50) if crop_outer_slices else False,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )

        if self.sensitivity_maps is not None:
            raise NotImplementedError(
                f"Sensitivity maps are not supported in the current "
                f"{self.__class__.__name__} class."
            )

        # Sampling rate in the slice-encode direction
        self.sampling_rate_slice_encode: float = 0.85
        self.transform = transform
        self.pass_mask: bool = pass_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        kspace = sample["kspace"]

        # TODO: use broadcasting function.
        if self.pass_mask:
            # # In case the data is already masked, the sampling mask can be recovered by finding the zeros.
            # This needs to be done in the primary function!
            # sampling_mask = ~(np.abs(kspace).sum(axis=(0, -1)) == 0)
            sample["mask"] = (sample["mask"] * np.ones(kspace.shape).astype(np.int32))[
                ..., np.newaxis
            ]

        kspace = (
            kspace[..., ::2] + 1j * kspace[..., 1::2]
        )  # Convert real-valued to complex-valued data.
        num_z = kspace.shape[1]
        kspace[:, int(np.ceil(num_z * self.sampling_rate_slice_encode)) :, :] = (
            0.0 + 0.0 * 1j
        )

        # Downstream code expects the coils to be at the first axis.
        # TODO: When named tensor support is more solid, this could be circumvented.
        sample["kspace"] = np.ascontiguousarray(kspace.transpose(2, 0, 1))

        if self.transform:
            sample = self.transform(sample)
        return sample


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments
    ---------
    datasets : sequence
        List of datasets to be concatenated

    From pytorch 1.5.1: torch.utils.data.ConcatDataset
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super().__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(
                d, IterableDataset
            ), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.logger = logging.getLogger(type(self).__name__)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = (
            idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        )
        return self.datasets[dataset_idx][sample_idx]


def build_dataset(
    name,
    root: pathlib.Path,
    filenames_filter: Optional[List[PathOrString]] = None,
    sensitivity_maps: Optional[pathlib.Path] = None,
    transforms: Optional[Any] = None,
    text_description: Optional[str] = None,
    kspace_context: Optional[int] = 0,
    **kwargs,
) -> Dataset:
    """

    Parameters
    ----------
    dataset_name : str
        Name of dataset class (without `Dataset`) in direct.data.datasets.
    root : pathlib.Path
        Root path to the data for the dataset class.
    filenames_filter : List
        List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
        on the root. If set, will skip searching for files in the root.
    sensitivity_maps : pathlib.Path
        Path to sensitivity maps.
    transforms : object
        Transformation object
    text_description : str
        Description of dataset, can be used for logging.
    kspace_context : int
        If set, output will be of shape -kspace_context:kspace_context.

    Returns
    -------
    Dataset
    """

    # TODO: Maybe only **kwargs are fine.
    logger.info(f"Building dataset for: {name}.")
    dataset_class: Callable = str_to_class("direct.data.datasets", name + "Dataset")
    logger.debug(f"Dataset class: {dataset_class}.")
    dataset = dataset_class(
        root=root,
        filenames_filter=filenames_filter,
        transform=transforms,
        sensitivity_maps=sensitivity_maps,
        text_description=text_description,
        kspace_context=kspace_context,
        **kwargs,
    )

    logger.debug(f"Dataset:\n{dataset}")

    return dataset


def build_dataset_from_input(
    transforms,
    dataset_config,
    initial_images,
    initial_kspaces,
    filenames_filter,
    data_root,
    pass_dictionaries,
):
    pass_h5s = None
    if initial_images is not None and initial_kspaces is not None:
        raise ValueError(
            f"initial_images and initial_kspaces are mutually exclusive. "
            f"Got {initial_images} and {initial_kspaces}."
        )

    if initial_images:
        pass_h5s = {"initial_image": (dataset_config.input_image_key, initial_images)}

    if initial_kspaces:
        pass_h5s = {
            "initial_kspace": (dataset_config.input_kspace_key, initial_kspaces)
        }

    dataset = build_dataset(
        root=data_root,
        filenames_filter=filenames_filter,
        transforms=transforms,
        pass_h5s=pass_h5s,
        pass_dictionaries=pass_dictionaries,
        **remove_keys(dataset_config, ["transforms", "lists"]),
    )
    return dataset
