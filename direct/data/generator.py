import logging

from sklearn.datasets import make_blobs
import numpy as np
import h5py

import pathlib
from typing import Dict, List, Optional, Tuple, Union

import os
logger = logging.getLogger(__name__)


class FakeMRIDataGenerator:

    def __init__(
        self,
        ndim: int = 2,
        blobs_n_samples: Optional[int] = None,
        blobs_cluster_std: Optional[float] = None,
    ) -> None:

        if ndim not in [2, 3]:
            raise NotImplementedError(f"Currently FakeMRIDataGenerator is not implemented for {ndim}D data.")

        self.ndim = ndim
        self.blobs_n_samples = blobs_n_samples
        self.blobs_cluster_std = blobs_cluster_std

        self.logger = logging.getLogger(type(self).__name__)

    def get_kspace(
        self,
        spatial_shape: Union[Tuple[int], List[int]],
        num_coils: int,
        seed: Optional[int] = None,
    ) -> np.array:

        samples, centers, classes = self.make_blobs(spatial_shape, num_coils)

        image = self._get_image_from_samples(samples, classes, num_coils, spatial_shape)

        if num_coils > 1:
            image = self._make_coil_data(image, samples, centers, classes)

        kspace = fft(image)

        return kspace[np.newaxis, ...] if self.ndim == 2 else kspace.transpose(1, 0, 2, 3)

    def get_attrs(self, sample: Dict) -> Dict:

        attrs = dict()
        attrs["norm"] = np.linalg.norm(sample["reconstruction_rss"])
        attrs["max"] = np.max(sample["reconstruction_rss"])
        attrs["encoding_size"] = sample["kspace"].shape[-2:]
        attrs["reconstruction_size"] = sample["reconstruction_rss"].shape[-2:]

        return attrs

    def make_blobs(
        self,
        spatial_shape: Union[List[int], Tuple[int]],
        num_coils: int,
        seed: Optional[int] = None,
    ) -> np.array:

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

        scaled_samples, scaled_centers = self._scale_data(samples, centers, spatial_shape)

        return scaled_samples, scaled_centers, classes

    def get_reconstruction_rss_from_kspace(
        self,
        kspace: np.array,
        coil_dim: int = 1
    ) -> np.array:

        return root_sum_of_squares(kspace, coil_dim)

    def _get_image_from_samples(self, samples, classes, num_coils, spatial_shape):

        image = np.zeros(tuple([num_coils] + list(spatial_shape)))

        for coil_idx in range(num_coils):

            if image.ndim == 3:
                image[
                    coil_idx,
                    samples[np.where(classes == coil_idx), 0],
                    samples[np.where(classes == coil_idx), 1]
                ] = 1  # assign 1 to each pixel

            elif image.ndim == 4:
                image[
                    coil_idx,
                    samples[np.where(classes == coil_idx), 0],
                    samples[np.where(classes == coil_idx), 1],
                    samples[np.where(classes == coil_idx), 2]
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

    def _calculate_interpolation_weights(self, samples, centers, classes):

        n_classes = np.unique(classes).shape[0]

        interpolation_weights = np.zeros((n_classes, n_classes))

        for i in range(n_classes):
            for j in range(n_classes):
                interpolation_weights[i, j] = 1 / np.linalg.norm(samples[classes == i] - centers[j], axis=0).mean()

            interpolation_weights[i] /= interpolation_weights[i].sum()

        return interpolation_weights

    def _scale_data(self, samples, centers, shape):

        scaled_samples = (samples - samples.min(0)) / (samples.max(0) - samples.min(0)) * (np.array(shape) - 1)
        scaled_samples = np.round(scaled_samples).astype(int)

        scaled_centers = (centers - samples.min(0)) / (samples.max(0) - samples.min(0)) * (np.array(shape) - 1)
        scaled_centers = np.round(scaled_centers).astype(int)

        return scaled_samples, scaled_centers

    def save_as_h5(self, sample, name, root):

        for idx in range(len(sample)):

            if len(name) < 5 or idx % (len(name) // 5) == 0 or len(name) == (idx + 1):
                self.logger.info(f"Storing samples: {(idx + 1) / len(name) * 100:.2f}%.")

            with h5py.File(root + name[idx] + ".h5", 'w') as h5_file:
                h5_file.create_dataset("kspace", data=sample[idx]["kspace"], dtype='c8')
                h5_file.create_dataset("reconstruction_rss", data=sample[idx]["reconstruction_rss"], dtype='f4')

                h5_file.attrs.create("max", data=sample[idx]["attrs"]["max"])
                h5_file.attrs.create("norm", data=sample[idx]["attrs"]["norm"])
                h5_file.attrs.create("reconstruction_size", data=sample[idx]["attrs"]["reconstruction_size"])
                h5_file.attrs.create("encoding_size", data=sample[idx]["attrs"]["encoding_size"])
                h5_file.attrs.create("filename", data=name[idx])

                h5_file.close()

    def __call__(
        self,
        sample_size: int = 1,
        num_coils: int = 1,
        spatial_shape: Union[List[int], Tuple[int]] = (100, 100),
        name: Union[str, List[str]] = "fake_mri_sample",
        seed: Optional[int] = None,
        save_as_h5: bool = False,
        root: Optional[pathlib.Path] = None,
    ) -> Dict:

        """
        Returns (and saves if save_as_h5 is True) fake mri samples.

        Parameters:
        -----------
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
            sample: dict
                Contains:
                    "kspace": np.array of shape (slice, num_coils, height, width)
                    "reconstruction_rss": np. array of shape (slice, height, width)
                    If spatial_shape is of shape 2 (height, width), slice=1.
        """

        if len(spatial_shape) != self.ndim:
            raise ValueError(f"Spatial shape must have {self.ndim} dimensions. Got shape {spatial_shape}.")

        sample = [dict() for _ in range(sample_size)]

        if isinstance(name, str):
            name = [name]

        if len(name) != sample_size:
            name = [name[0] + f'{_:04}' for _ in range(1, sample_size + 1)]

        for idx in range(sample_size):

            sample[idx]["kspace"] = self.get_kspace(spatial_shape, num_coils)
            sample[idx]["reconstruction_rss"] = self.get_reconstruction_rss_from_kspace(
                sample[idx]["kspace"],
                coil_dim=1
            )
            sample[idx]["attrs"] = self.get_attrs(sample[idx])
            sample[idx]["filename"] = name[idx]

        if save_as_h5:

            if root is None:
                root = "./"
            if not os.path.exists(root):
                os.makedirs(root)

            self.save_as_h5(sample, name, root)

        return sample if sample_size > 1 else sample[0]

def fft(data, dims=(-2, -1)):
    data = np.fft.ifftshift(data, dims)
    out = np.fft.fft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def ifft(data, dims=(-2, -1)):
    data = np.fft.ifftshift(data, dims)
    out = np.fft.ifft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def root_sum_of_squares(data, coil_dim=1):
    return np.sqrt((np.abs(ifft(data)) ** 2).sum(coil_dim))
