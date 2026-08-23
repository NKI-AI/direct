# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""direct.data.fake module."""

import logging
import pathlib

import numpy as np
from sklearn.datasets import make_blobs

from direct.data.sens import simulate_sensitivity_maps

logger = logging.getLogger(__name__)


class FakeMRIData:
    """Generates fake 2D or 3D MRI data by generating random 2D or 3D images of gaussian blobs."""

    def __init__(
        self,
        ndim: int = 2,
        blobs_n_samples: int | None = None,
        blobs_cluster_std: float | None = None,
    ) -> None:
        """Inits :class:`FakeMRIData`.

        Args:
            ndim: Dimension of samples. Can be ``2`` or 3. Default is ``2``.
            blobs_n_samples: The total number of points equally divided among clusters. Default is ``None``.
            blobs_cluster_std: Standard deviation of the clusters. Default is ``None``.

        Returns:
            ``None``.
        """

        if ndim not in [2, 3]:
            raise NotImplementedError(f"Currently FakeMRIData is not implemented for {ndim}D data.")

        self.ndim = ndim
        self.blobs_n_samples = blobs_n_samples
        self.blobs_cluster_std = blobs_cluster_std

        self.logger = logging.getLogger(type(self).__name__)

    def get_kspace(
        self,
        spatial_shape: list[int] | tuple[int, ...],
        num_coils: int,
    ) -> np.ndarray:
        """Get kspace.

        Args:
                    spatial_shape: Spatial shape.
                    num_coils: Num coils.

        Returns:
            The result.
        """

        samples = self.make_blobs(spatial_shape, num_coils)

        image = self._get_image_from_samples(samples, spatial_shape)
        image = image[None]
        if num_coils > 1:
            sens_maps = simulate_sensitivity_maps(spatial_shape[-2:], num_coils)

            image = image * (sens_maps if self.ndim == 2 else sens_maps[:, None])

        kspace = fft(image)

        return kspace[np.newaxis, ...] if self.ndim == 2 else kspace.transpose(1, 0, 2, 3)

    def set_attrs(self, sample: dict) -> dict:
        """Sets metadata attributes of sample.

        Args:
            sample: Sample.

        Returns:
            The result.
        """

        attrs = {}
        attrs["norm"] = np.linalg.norm(sample["reconstruction_rss"])
        attrs["max"] = np.max(sample["reconstruction_rss"])
        attrs["encoding_size"] = sample["kspace"].shape[-self.ndim :]
        attrs["reconstruction_size"] = sample["reconstruction_rss"].shape[-self.ndim :]

        return attrs

    def make_blobs(
        self,
        spatial_shape: list[int] | tuple[int, ...],
        num_coils: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates gaussian blobs in ``'num_coils'`` classes and scales them the interval.

        [0, slice] x [0, height] x [0, width].

        Args:
            spatial_shape: Spatial shape.
            num_coils: Num coils.
            seed: Seed.

        Returns:
            The result.
        """

        # Number of samples to be converted to an image
        n_samples = self.blobs_n_samples if self.blobs_n_samples else np.prod(list(spatial_shape)) // self.ndim
        cluster_std = self.blobs_cluster_std if self.blobs_cluster_std is not None else 0.1

        samples, _, _ = make_blobs(
            n_samples=n_samples,
            n_features=self.ndim,
            centers=num_coils,
            cluster_std=cluster_std,
            center_box=(0, 1),
            random_state=seed,
            return_centers=True,
        )

        samples = scale_data(data=samples, shape=spatial_shape)

        return samples

    @staticmethod
    def _get_image_from_samples(samples, spatial_shape):
        """Get image from samples.

        Args:
            samples: Samples.
            spatial_shape: Spatial shape.

        Returns:
            ``None``.
        """
        image = np.zeros(list(spatial_shape))
        image[tuple(np.split(samples, len(spatial_shape), axis=-1))] = 1

        return image + 0.0j

    def __call__(
        self,
        sample_size: int = 1,
        num_coils: int = 1,
        spatial_shape: list[int] | tuple[int, ...] = (100, 100),
        name: str | list[str] = "fake_mri_sample",
        seed: int | None = None,
        root: pathlib.Path | None = None,
    ) -> list[dict]:
        """Returns fake mri samples in the form of gaussian blobs.

        Args:
            sample_size: Size of the samples.
            num_coils: Number of simulated coils.
            spatial_shape: Must be ``(slice, height, width)`` or ``(height, width)``.
            name: Name of file.
            root: Root to save data. To be used with save_as_h5=``True``

        Returns:
            --------
            Contains: ``"kspace"``: np.array of shape ``(slice, num_coils, height, width)`` ``"reconstruction_rss"``:
                np. array of shape ``(slice, height, width)`` If spatial_shape is of shape ``2`` ``(height, width)``,
                slice=1.
        """

        if len(spatial_shape) != self.ndim:
            raise ValueError(f"Spatial shape must have {self.ndim} dimensions. Got shape {spatial_shape}.")

        sample: list[dict] = [{} for _ in range(sample_size)]

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


def scale_data(data, shape):
    """Scales data to (0,1) and then to shape.

    Args:
        data: Data.
        shape: Shape.

    Returns:
        ``None``.
    """

    scaled_data = (data - data.min(0)) / (data.max(0) - data.min(0)) * (np.array(shape) - 1)
    scaled_data = np.round(scaled_data).astype(int)

    return scaled_data


def fft(data, dims=(-2, -1)):
    """Fast Fourier Transform.

    Args:
        data: Data.
        dims: Dims.

    Returns:
        ``None``.
    """
    data = np.fft.ifftshift(data, dims)
    out = np.fft.fft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def ifft(data, dims=(-2, -1)):
    """Inverse Fast Fourier Transform.

    Args:
        data: Data.
        dims: Dims.

    Returns:
        ``None``.
    """
    data = np.fft.ifftshift(data, dims)
    out = np.fft.ifft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def root_sum_of_squares(kspace_data, coil_dim=1):
    """Root Sum of Squares Estimate, given kspace data.

    Args:
        kspace_data: Kspace data.
        coil_dim: Coil dim.

    Returns:
        ``None``.
    """
    return np.sqrt((np.abs(ifft(kspace_data)) ** 2).sum(coil_dim))
