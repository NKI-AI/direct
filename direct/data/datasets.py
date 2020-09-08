# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np
import pathlib
import bisect

from typing import Callable, Dict, Optional, Any, List
from functools import lru_cache

from direct.data.h5_data import H5SliceData
from direct.utils import str_to_class
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
        dataset_description: Optional[Dict[Any, Any]] = None,
        pass_mask: bool = False,
        pass_header: bool = True,
        **kwargs,
    ) -> None:

        extra_keys = ["mask"] if pass_mask else []
        self.pass_header = pass_header
        if pass_header:
            extra_keys.append("ismrmrd_header")

        super().__init__(
            root=root,
            filenames_filter=filenames_filter,
            dataset_description=dataset_description,
            metadata=None,
            extra_keys=tuple(extra_keys),
            **kwargs,
        )
        if self.sensitivity_maps is not None:
            raise NotImplementedError(
                f"Sensitivity maps are not supported in the current "
                f"{self.__class__.__name__} class."
            )

        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        if self.pass_header:
            sample.update(self.parse_header(sample["ismrmrd_header"]))
            del sample["ismrmrd_header"]
            # Some images have strange behavior.
            image_shape = sample["kspace"].shape

            if (
                image_shape[-1] < sample["reconstruction_size"][-2]
            ):  # reconstruction size is (x, y, z)
                # warnings.warn(
                #     f"Encountered {sample['filename']} with header reconstruction size {sample['reconstruction_size']}, "
                #     f" yet matrix size is {image_shape}, this is a known issue in the FastMRI dataset."
                # )
                sample["reconstruction_size"] = (image_shape[-1], image_shape[-1], 1)

        if self.transform:
            sample = self.transform(sample)

        return sample

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
        filenames_filter: Optional[List[PathOrString]] = None,
        dataset_description: Optional[Dict[Any, Any]] = None,
        pass_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            filenames_filter=filenames_filter,
            dataset_description=dataset_description,
            metadata=None,
            extra_keys=None,
            **kwargs,
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
    dataset_name,
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

    logger.info(f"Building dataset for: {dataset_name}.")
    dataset_class: Callable = str_to_class(
        "direct.data.datasets", dataset_name + "Dataset"
    )
    logger.debug(f"Dataset class: {dataset_class}.")

    dataset = dataset_class(
        root=root,
        filenames_filter=filenames_filter,
        dataset_description=None,
        transform=transforms,
        sensitivity_maps=sensitivity_maps,
        pass_mask=False,
        text_description=text_description,
        kspace_context=kspace_context,
        **kwargs,
    )

    logger.debug(f"Dataset:\n{dataset}")

    return dataset
