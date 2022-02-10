# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_blobs

logger = logging.getLogger(__name__)


class FakeMRIData:
    """
    Generates fake 2D or 3D MRI data by generating random 2D or 3D images of gaussian blobs.
    """

    def __init__(
        self,
        ndim: int = 2,
        blobs_n_samples: Optional[int] = None,
        blobs_cluster_std: Optional[float] = None,
    ) -> None:
        """

        Parameters
        ----------
        ndim: int
        blobs_n_samples: Optional[int], default is None.
        blobs_cluster_std: Optional[float], default is None.
        """

        if ndim not in [2, 3]:
            raise NotImplementedError(f"Currently FakeMRIData is not implemented for {ndim}D data.")

        self.ndim = ndim
        self.blobs_n_samples = blobs_n_samples
        self.blobs_cluster_std = blobs_cluster_std

        self.logger = logging.getLogger(type(self).__name__)

    def get_kspace(
        self,
        spatial_shape: Union[List[int], Tuple[int, ...]],
        num_coils: int,
    ) -> np.array:
        """
        Parameters
        ----------
        spatial_shape: List of ints or tuple of ints.
        num_coils: int
        """

        samples, centers, classes = self.make_blobs(spatial_shape, num_coils)

        image = self._get_image_from_samples(samples, classes, num_coils, spatial_shape)

        if num_coils > 1:
            image = self._make_coil_data(image, samples, centers, classes)

        kspace = fft(image)

        return kspace[np.newaxis, ...] if self.ndim == 2 else kspace.transpose(1, 0, 2, 3)

    def set_attrs(self, sample: Dict) -> Dict:
        """
        Sets metadata attributes of sample.
        """

        attrs = dict()
        attrs["norm"] = np.linalg.norm(sample["reconstruction_rss"])
        attrs["max"] = np.max(sample["reconstruction_rss"])
        attrs["encoding_size"] = sample["kspace"].shape[-self.ndim :]
        attrs["reconstruction_size"] = sample["reconstruction_rss"].shape[-self.ndim :]

        return attrs

    def make_blobs(
        self,
        spatial_shape: Union[List[int], Tuple[int, ...]],
        num_coils: int,
        seed: Optional[int] = None,
    ) -> np.array:
        """
        Generates gaussian blobs in 'num_coils' classes and scales them the interval
        [0, slice] x [0, heihgt] x [0, width].
        """

        # Number of samples to be converted to an image
        n_samples = self.blobs_n_samples if self.blobs_n_samples else np.prod(list(spatial_shape)) // self.ndim
        cluster_std = self.blobs_cluster_std if self.blobs_cluster_std is not None else 0.1

        samples, classes, centers = make_blobs(
            n_samples=n_samples,
            n_features=self.ndim,
            centers=num_coils,
            cluster_std=cluster_std,
            center_box=(0, 1),
            random_state=seed,
            return_centers=True,
        )

        scaled_samples = scale_data(data=samples, shape=spatial_shape)
        scaled_centers = scale_data(data=centers, other=samples, shape=spatial_shape)

        return scaled_samples, scaled_centers, classes

    def _get_image_from_samples(self, samples, classes, num_coils, spatial_shape):
        image = np.zeros(tuple([num_coils] + list(spatial_shape)))
        for coil_idx in range(num_coils):

            if self.ndim == 2:
                image[
                    coil_idx, samples[np.where(classes == coil_idx), 0], samples[np.where(classes == coil_idx), 1]
                ] = 1  # assign 1 to each pixel

            elif self.ndim == 3:
                image[
                    coil_idx,
                    samples[np.where(classes == coil_idx), 0],
                    samples[np.where(classes == coil_idx), 1],
                    samples[np.where(classes == coil_idx), 2],
                ] = 1

        return image

    def _make_coil_data(self, image, samples, centers, classes):
        return self._interpolate_clusters(image, samples, centers, classes)

    def _interpolate_clusters(self, image, samples, centers, classes):
        weights = self._calculate_interpolation_weights(samples, centers, classes)
        if image.ndim == 3:
            image = image.transpose(1, 2, 0).dot(weights.T).transpose(2, 0, 1)
        elif image.ndim == 4:
            image = image.transpose(1, 2, 3, 0).dot(weights.T).transpose(3, 0, 1, 2)

        return image

    @staticmethod
    def _calculate_interpolation_weights(samples, centers, classes):
        n_classes = np.unique(classes).shape[0]
        interpolation_weights = np.zeros((n_classes, n_classes))
        for idx_i in range(n_classes):
            for idx_j in range(n_classes):
                interpolation_weights[idx_i, idx_j] = (
                    1 / np.linalg.norm(samples[classes == idx_i] - centers[idx_j], axis=0).mean()
                )

            interpolation_weights[idx_i] /= interpolation_weights[idx_i].sum()

        return interpolation_weights

    def __call__(
        self,
        sample_size: int = 1,
        num_coils: int = 1,
        spatial_shape: Union[List[int], Tuple[int, ...]] = (100, 100),
        name: Union[str, List[str]] = "fake_mri_sample",
        seed: Optional[int] = None,
        root: Optional[pathlib.Path] = None,
    ) -> List[Dict]:

        """
        Returns (and saves if save_as_h5 is True) fake mri samples in the form of gaussian blobs.

        Parameters
        ----------
        sample_size: int
            Size of the samples.
        num_coils: int
            Number of simulated coils.
        spatial_shape: List of ints or Tuple of ints.
            Must be (slice, height, width) or (height, width).
        name: String or list of strings.
            Name of file.
        save_as_h5: bool
            If set to True samples will be saved on root.
        root: pathlib.Path, Optional
            Root to save data. To be used with save_as_h5=True

        Returns:
        --------
            sample: dict or list of dicts
                Contains:
                    "kspace": np.array of shape (slice, num_coils, height, width)
                    "reconstruction_rss": np. array of shape (slice, height, width)
                    If spatial_shape is of shape 2 (height, width), slice=1.
        """

        if len(spatial_shape) != self.ndim:
            raise ValueError(f"Spatial shape must have {self.ndim} dimensions. Got shape {spatial_shape}.")

        sample: List[Dict] = [dict() for _ in range(sample_size)]

        if isinstance(name, str):
            name = [name]

        if len(name) != sample_size:
            name = [name[0] + f"{_:04}" for _ in range(1, sample_size + 1)]

        for idx in range(sample_size):

            sample[idx]["kspace"] = self.get_kspace(spatial_shape, num_coils)
            sample[idx]["reconstruction_rss"] = root_sum_of_squares(sample[idx]["kspace"], coil_dim=1)
            sample[idx]["attrs"] = self.set_attrs(sample[idx])
            sample[idx]["filename"] = name[idx]

        return sample  # if sample_size > 1 else sample[0]


def scale_data(data, shape, other=None):
    """
    Scales data to (0,1) and then to shape.
    If other is specified, data is scaled based on other to (0,1) and then to shape.
    """
    if other is None:
        other = data
    scaled_data = (data - other.min(0)) / (other.max(0) - other.min(0)) * (np.array(shape) - 1)
    scaled_data = np.round(scaled_data).astype(int)

    return scaled_data


def fft(data, dims=(-2, -1)):
    """
    Fast Fourier Transform.
    """
    data = np.fft.ifftshift(data, dims)
    out = np.fft.fft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def ifft(data, dims=(-2, -1)):
    """
    Inverse Fast Fourier Transform.
    """
    data = np.fft.ifftshift(data, dims)
    out = np.fft.ifft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def root_sum_of_squares(kspace_data, coil_dim=1):
    """
    Root Sum of Squares Estimate, given kspace data.
    """
    return np.sqrt((np.abs(ifft(kspace_data)) ** 2).sum(coil_dim))
