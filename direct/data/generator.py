from sklearn.datasets import make_blobs
import numpy as np
import h5py

import os
import pathlib
from typing import Dict, List, Tuple, Union


class FakeMRIDataGenerator:
    """
    Generates fake MRI data in the form of Gaussian Blobs. When called provides H5 files with stored
    "kspace" and "reconstruction_rss" images.

    Parameters:
    -----------
        root: pathlib.Path
            Where to store samples.
        ndim: int
            Dimension of samples.
        blobs_cluster_std: float
            Standard Deviation of the Gaussian Blobs.
    """

    def __init__(
        self,
        root: pathlib.Path = "./",
        ndim: int = 2,
        blobs_cluster_std: float = 0.05
    ) -> None:

        if ndim != 2:
            raise NotImplementedError(f"Currently FakeMRIDataGenerator is not implemented for {ndim}D data.")

        if not os.path.exists(root):
            os.makedirs(root)

        self.root = root
        self.ndim = ndim
        self.blobs_cluster_std = blobs_cluster_std

    def get_kspace(
        self,
        spatial_shape: Union[Tuple[int], List[int]],
        num_coils: int
    ) -> np.array:

        samples, centers, classes = self.make_blobs(spatial_shape, num_coils)

        image = self._get_image_from_samples(samples, classes, num_coils, spatial_shape)

        if num_coils > 1:
            image = self._interpolate_clusters(image, samples, centers, classes)

        kspace = fft(image)

        return kspace[np.newaxis, ...]

    def get_attrs(self, sample : Dict) -> Dict:

        attrs = dict()
        attrs["norm"] = np.linalg.norm(sample["reconstruction_rss"])
        attrs["max"] = np.max(sample["reconstruction_rss"])
        attrs["encoding_size"] = sample["kspace"].shape[-2:]
        attrs["reconstruction_size"] = sample["reconstruction_rss"].shape[-2:]

        return attrs

    def make_blobs(
        self,
        spatial_shape: Union[List[int], Tuple[int]],
        num_coils: int
    ) -> np.array:

        height, width = spatial_shape
        # Number of samples to be converted to an image
        n_samples = (height * width) // 2

        samples, classes, centers = make_blobs(
            n_samples=n_samples,
            n_features=self.ndim,
            centers=num_coils,
            cluster_std=self.blobs_cluster_std,
            center_box=(0, 1),
            return_centers=True,
        )

        scaled_samples, scaled_centers = self._scale_data(samples, centers, (height, width))

        return scaled_samples, scaled_centers, classes

    def get_reconstruction_rss_from_kspace(
        self,
        kspace : np.array,
        coil_dim : int = 0
    ) -> np.array:

        return root_sum_of_squares(kspace, coil_dim)[np.newaxis, ...]

    def _get_image_from_samples(self, samples, classes, num_coils, spatial_shape):

        image = np.zeros((num_coils, spatial_shape[0], spatial_shape[1]))

        for coil_idx in range(num_coils):
            image[coil_idx, samples[np.where(classes == coil_idx), 0], samples[
                np.where(classes == coil_idx), 1]] = 1  # assign 1 to each pixel
        # image *= np.random.rand(*image.shape)
        return image

    def _interpolate_clusters(self, image, samples, centers, classes):

        weights = self._calculate_interpolation_weights(samples, centers, classes)

        image = image.transpose(1, 2, 0).dot(weights.T).transpose(2, 0, 1)

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

        scaled_samples = (samples - samples.min(0)) / (samples.max(0) - samples.min(0)) * np.array(
            [(shape[0] - 1), (shape[1] - 1)])
        scaled_samples = np.round(scaled_samples).astype(int)

        scaled_centers = (centers - samples.min(0)) / (samples.max(0) - samples.min(0)) * np.array(
            [(shape[0] - 1), (shape[1] - 1)])
        scaled_centers = np.round(scaled_centers).astype(int)

        return scaled_samples, scaled_centers


    def __call__(
        self,
        size : int = 1,
        num_coils : int = 1,
        spatial_shape : Union[List[int], Tuple[int]] = (100, 100),
        name : Union[str, List[str]] = "fake_mri_sample",
        save_as_h5 : bool = True,
    ) -> Dict:

        if len(spatial_shape) != self.ndim:
            raise NotImplementedError(f"Currently FakeMRIDataGenerator is not implemented for {len(spatial_shape)}D data.")

        sample = [dict() for _ in range(size)]

        if isinstance(name, str):
            name = [name]

        if len(name) != size:
            name = [name[0] + f'{_:04}' for _ in range(1, size+1)]

        for ind in range(size):
            sample[ind]["kspace"] = self.get_kspace(spatial_shape, num_coils)
            sample[ind]["reconstruction_rss"] = self.get_reconstruction_rss_from_kspace(sample[ind]["kspace"], coil_dim=0)
            sample[ind]["attrs"] = self.get_attrs(sample[ind])

            if save_as_h5:
                with h5py.File(self.root + name[ind] + ".h5", 'w') as h5_file:
                    h5_file.create_dataset("kspace", data=sample[ind]["kspace"], dtype='c8')
                    h5_file.create_dataset("reconstruction_rss", data=sample[ind]["reconstruction_rss"], dtype='f4')

                    h5_file.attrs.create("max", data=sample[ind]["attrs"]["max"])
                    h5_file.attrs.create("norm", data=sample[ind]["attrs"]["norm"])
                    h5_file.attrs.create("reconstruction_size", data=sample[ind]["attrs"]["reconstruction_size"])
                    h5_file.attrs.create("encoding_size", data=sample[ind]["attrs"]["encoding_size"])
                    h5_file.attrs.create("filename", data=name[ind])

                    h5_file.close()

        return sample


def fft(data_numpy, dims=(-2, -1)):

    data_numpy = np.fft.ifftshift(data_numpy, dims)
    out_numpy = np.fft.fft2(data_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, dims)

    return out_numpy

def ifft(data_numpy, dims=(-2, -1)):

    data_numpy = np.fft.ifftshift(data_numpy, dims)
    out_numpy = np.fft.ifft2(data_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, dims)

    return out_numpy

def root_sum_of_squares(data, coil_dim=0):

    return np.sqrt((np.abs(ifft(data)) ** 2).sum(coil_dim))
