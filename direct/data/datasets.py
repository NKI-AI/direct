# coding=utf-8
# Copyright (c) DIRECT Contributors

import bisect
import contextlib
import pathlib
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from direct.data.fake import FakeMRIData
from direct.data.h5_data import H5SliceData
from direct.types import PathOrString
from direct.utils import remove_keys, str_to_class

try:
    import ismrmrd
except ImportError:
    raise ImportError(
        "ISMRMD Library not available. Will not be able to parse ISMRMD headers. "
        "Install pyxb and ismrmrd-python from https://github.com/ismrmrd/ismrmrd-python "
        "if you wish to parse the headers."
    )

import logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class FakeMRIBlobsDataset(Dataset):
    """
    A PyTorch Dataset class which outputs random fake k-space
    images which reconstruct into Gaussian blobs.
    """

    def __init__(
        self,
        sample_size: int,
        num_coils: int,
        spatial_shape: Union[List[int], Tuple[int]],
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        filenames: Optional[Union[List[str], str]] = None,
        pass_attrs: Optional[bool] = None,
        text_description: Optional[str] = None,
        kspace_context: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Dataset initialisation.

        Parameters
        ----------
        sample_size: int
            Size of the dataset.
        num_coils: int
            Number of coils for the fake k-space data.
        spatial_shape: List or Tuple of ints.
            Shape of the reconstructed fake data. Should be (height, width) or (slice, height, width), corresponding
            to ndim = 2 and ndim = 3.
        transform: Optional[Callable]
            A list of transforms to be performed on the generated samples. Default is None.
        seed: int
            Seed. Default is None.
        filenames: List of strings or string.
            Names for the generated samples. If string is given, a number order starting from "00001" is appended
            to the name of each sample.
        pass_attrs: bool
            Pass the attributes of the generated sample.
        text_description: str
            Description of dataset, can be useful for logging.
        kspace_context: bool
            If true corresponds to 3D reconstruction, else reconstruction is 2D.
        """

        self.logger = logging.getLogger(type(self).__name__)

        if len(spatial_shape) not in [2, 3]:
            raise NotImplementedError(
                f"Currently FakeDataset is implemented only for 2D or 3D data."
                f"Spatial shape must have 2 or 3 dimensions. Got shape {spatial_shape}."
            )
        self.sample_size = sample_size
        self.num_coils = num_coils
        self.spatial_shape = spatial_shape
        self.transform = transform
        self.pass_attrs = pass_attrs if pass_attrs is not None else True
        self.text_description = text_description
        if self.text_description:
            self.logger.info("Dataset description: {text_description}.", text_description=self.text_description)

        self.generator: Callable = FakeMRIData(
            ndim=len(self.spatial_shape),
            blobs_n_samples=kwargs.get("blobs_n_samples", None),
            blobs_cluster_std=kwargs.get("blobs_cluster_std", None),
        )
        self.volume_indices: Dict[str, range] = {}

        self.rng = np.random.RandomState()

        with temp_seed(self.rng, seed):
            # size = sample_size * num_slices if data is 3D
            self.data = [
                (filename, slice_no, seed)
                for (filename, seed) in zip(
                    self.parse_filenames_data(filenames),
                    list(self.rng.choice(a=range(int(1e5)), size=self.sample_size, replace=False)),
                )  # ensure reproducibility
                for slice_no in range(self.spatial_shape[0] if len(spatial_shape) == 3 else 1)
            ]
        self.kspace_context = kspace_context if kspace_context else 0
        self.ndim = 2 if self.kspace_context == 0 else 3

        if self.kspace_context != 0:
            raise NotImplementedError("3D reconstruction is not yet supported with FakeMRIBlobsDataset.")

    def parse_filenames_data(self, filenames):

        if filenames is None:
            filenames = ["sample"]

        if isinstance(filenames, str):
            filenames = [filenames]

        if len(filenames) != self.sample_size:
            filenames = [filenames[0] + f"{_:05}" for _ in range(1, self.sample_size + 1)]

        current_slice_number = 0
        for idx, filename in enumerate(filenames):
            if len(filenames) < 5 or idx % (len(filenames) // 5) == 0 or len(filenames) == (idx + 1):
                self.logger.info("Parsing: {progress:.2f}%.", progress=(idx + 1) / len(filenames) * 100)

            num_slices = self.spatial_shape[0] if len(self.spatial_shape) == 3 else 1
            self.volume_indices[filename] = range(current_slice_number, current_slice_number + num_slices)
            current_slice_number += num_slices

        return filenames

    @staticmethod
    def _get_metadata(metadata):
        encoding_size = metadata["encoding_size"]
        reconstruction_size = metadata["reconstruction_size"]
        metadata = {
            "encoding_size": encoding_size,
            "reconstruction_size": reconstruction_size,
        }
        return metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename, slice_no, sample_seed = self.data[idx]

        sample = self.generator(
            sample_size=1,
            num_coils=self.num_coils,
            spatial_shape=self.spatial_shape,
            name=[filename],
            seed=sample_seed,
        )[0]
        sample["kspace"] = sample["kspace"][slice_no]

        if "attrs" in sample:
            metadata = self._get_metadata(sample["attrs"])
            sample.update(metadata)

            if self.pass_attrs:
                sample["scaling_factor"] = sample["attrs"]["max"]

            del sample["attrs"]

        sample["slice_no"] = slice_no
        if sample["kspace"].ndim == 2:  # Singlecoil data does not always have coils at the first axis.
            sample["kspace"] = sample["kspace"][np.newaxis, ...]

        if self.transform:
            sample = self.transform(sample)

        return sample


class FastMRIDataset(H5SliceData):
    """
    FastMRI challenge dataset.
    """

    def __init__(
        self,
        root: pathlib.Path,
        transform: Optional[Callable] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        regex_filter: Optional[str] = None,
        pass_mask: bool = False,
        pass_max: bool = True,
        initial_images: Union[List[pathlib.Path], None] = None,
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
                f"Sensitivity maps are not supported in the current " f"{self.__class__.__name__} class."
            )

        # TODO: Make exclusive or to give error when one of the two keys is not set.
        # TODO: Convert into mixin, and add support to main image
        # TODO: Such a support would also work for the sensitivity maps
        self.initial_images_key = initial_images_key
        self.initial_images = {}

        if initial_images:
            self.initial_images = {k.name: k for k in initial_images}

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
        if image_shape[-1] < sample["reconstruction_size"][-2]:  # reconstruction size is (x, y, z)
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
            sample["acs_mask"] = self.__broadcast_mask(kspace_shape, self.__get_acs_from_fastmri_mask(sampling_mask))

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
        left = right = mask.shape[-1] // 2
        while mask[..., right]:
            right += 1
        while mask[..., left]:
            left -= 1
        acs_mask = np.zeros_like(mask)
        acs_mask[:, left + 1 : right] = 1
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
        # Borrowed from: https://github.com/facebookresearch/fastMRI/blob/\
        # 57c0a9ef52924d1ffb30d7b7a51d022927b04b23/fastmri/data/mri_data.py#L136
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
    """
    Calgary-Campinas challenge dataset.
    """

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
            slice_data=slice(50, -50) if crop_outer_slices else None,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )

        if self.sensitivity_maps is not None:
            raise NotImplementedError(
                f"Sensitivity maps are not supported in the current " f"{self.__class__.__name__} class."
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
            sample["mask"] = (sample["mask"] * np.ones(kspace.shape).astype(np.int32))[..., np.newaxis]

        kspace = kspace[..., ::2] + 1j * kspace[..., 1::2]  # Convert real-valued to complex-valued data.
        num_z = kspace.shape[1]
        kspace[:, int(np.ceil(num_z * self.sampling_rate_slice_encode)) :, :] = 0.0 + 0.0 * 1j

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
        out_sequence, total = [], 0
        for item in sequence:
            length = len(item)
            out_sequence.append(length + total)
            total += length
        return out_sequence

    def __init__(self, datasets):
        super().__init__()
        if len(datasets) <= 0:
            raise AssertionError("datasets should not be an empty iterable")
        self.datasets = list(datasets)
        for dataset in self.datasets:
            if isinstance(dataset, IterableDataset):
                raise AssertionError("ConcatDataset does not support IterableDataset")
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.logger = logging.getLogger(type(self).__name__)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def build_dataset(
    name: str,
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
    name : str
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
    logger.info("Building dataset for: {name}.", name=name)
    dataset_class: Callable = str_to_class("direct.data.datasets", name + "Dataset")
    logger.debug("Dataset class: {dataset_class}.", dataset_class=dataset_class)
    dataset = dataset_class(
        root=root,
        filenames_filter=filenames_filter,
        transform=transforms,
        sensitivity_maps=sensitivity_maps,
        text_description=text_description,
        kspace_context=kspace_context,
        **kwargs,
    )

    logger.debug("Dataset:\n{dataset}", dataset=dataset)

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
    """
    Parameters
    ----------
    transforms : object, Callable
        Transformation object.
    dataset_config: Dataset configuration file
    initial_images: pathlib.Path
        Path to initial_images.
    initial_kspaces: pathlib.Path
        Path to initial kspace images.
    filenames_filter : List
        List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
        on the root. If set, will skip searching for files in the root.
    data_root : pathlib.Path
        Root path to the data for the dataset class.
    pass_dictionaries:

    Returns
    -------
    Dataset
    """
    pass_h5s = None
    if initial_images is not None and initial_kspaces is not None:
        raise ValueError(
            f"initial_images and initial_kspaces are mutually exclusive. "
            f"Got {initial_images} and {initial_kspaces}."
        )

    if initial_images:
        pass_h5s = {"initial_image": (dataset_config.input_image_key, initial_images)}

    if initial_kspaces:
        pass_h5s = {"initial_kspace": (dataset_config.input_kspace_key, initial_kspaces)}

    dataset = build_dataset(
        root=data_root,
        filenames_filter=filenames_filter,
        transforms=transforms,
        pass_h5s=pass_h5s,
        pass_dictionaries=pass_dictionaries,
        **remove_keys(dataset_config, ["transforms", "lists"]),
    )
    return dataset
