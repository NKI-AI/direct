# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""Random elastic deformation utilities for registration data augmentation."""

from typing import Any

import numpy as np
import torch

from direct.types import TransformKey
from direct.utils import DirectModule


def _deform_random_grid(*args, **kwargs) -> list[np.ndarray]:
    """Lazy import wrapper for :func:`elasticdeform.deform_random_grid`.

    ``elasticdeform`` wheels may be incompatible with NumPy 2.x.

    Returns
    -------
    list[np.ndarray]
        Deformed images returned by ``elasticdeform``.
    """
    try:
        from elasticdeform import deform_random_grid
    except ImportError as exc:
        raise ImportError(
            "elasticdeform is required for random elastic deformation. "
            "Install a NumPy-2-compatible build, e.g. "
            "`pip install --no-binary=elasticdeform elasticdeform`, "
            "or avoid registration_simulate_reference=ELASTIC."
        ) from exc
    return deform_random_grid(*args, **kwargs)


def random_elastic_deformation(
    image: torch.Tensor,
    sigma: float = 2.0,
    points: int = 3,
    order: int = 3,
    prefilter: bool = True,
    rotate: float | None = None,
    zoom: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply elastic deformation with a random deformation grid.

    This generates a random, square deformation grid with displacements
    sampled from a normal distribution with standard deviation ``sigma``.
    The deformation is then applied to the images.

    Parameters
    ----------
    image : torch.Tensor
        Image to deform with shape (batch, height, width).
    sigma : float
        Standard deviation of the normal distribution for the random displacements. Default: 2.0.
    points : int
        Number of points of the random deformation grid. Default: 3.
    order : int
        Interpolation order. Can be {0, 1, 2, 3, 4}. Default: 3.
    prefilter : bool
        If True the input will be pre-filtered with a spline filter. Default: True.
    rotate : float, optional
        Angle in degrees to rotate the output. Default: None.
    zoom : float, optional
        Scale factor to zoom the output. Default: None.
    seed : int, optional
        Random seed for reproducibility. Default: None.

    Returns
    -------
    torch.Tensor
        Deformed image with shape (batch, height, width).
    """
    if seed is not None:
        np.random.seed(seed)

    deformed_image = _deform_random_grid(
        [*image.numpy()],
        sigma=sigma,
        points=points,
        order=order,
        prefilter=prefilter,
        rotate=rotate,
        zoom=zoom,
    )

    return torch.from_numpy(np.array(deformed_image))


class RandomElasticDeformationModule(DirectModule):
    """Transform module that applies random elastic deformation to an image."""

    def __init__(
        self,
        image_key: TransformKey,
        target_key: TransformKey = TransformKey.REFERENCE_IMAGE,
        sigma: float = 2.0,
        points: int = 3,
        order: int = 3,
        prefilter: bool = True,
        rotate: float | None = None,
        zoom: float | None = None,
        use_seed: bool | None = None,
    ) -> None:
        """Inits :class:`RandomElasticDeformationModule`.

        Parameters
        ----------
        image_key : TransformKey
            Key of the image to deform.
        target_key : TransformKey
            Key of the deformed image. Default: TransformKey.REFERENCE_IMAGE.
        sigma : float
            Standard deviation of the normal distribution for the random displacements. Default: 2.0.
        points : int
            Number of points of the random deformation grid. Default: 3.
        order : int
            Interpolation order. Can be {0, 1, 2, 3, 4}. Default: 3.
        prefilter : bool
            If True the input will be pre-filtered with a spline filter. Default: True.
        rotate : float, optional
            Angle in degrees to rotate the output. Default: None.
        zoom : float, optional
            Scale factor to zoom the output. Default: None.
        use_seed : bool, optional
            Whether to use a random seed for reproducibility. Default: None.
        """
        super().__init__()

        self.sigma = sigma
        self.points = points
        self.order = order
        self.prefilter = prefilter
        self.rotate = rotate
        self.zoom = zoom
        self.use_seed = use_seed

        self.image_key = image_key
        self.target_key = target_key

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply random elastic deformation to the configured image key.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing the image to deform.

        Returns
        -------
        dict[str, Any]
            Dictionary with the deformed image stored under ``target_key``.
        """
        image = data[self.image_key]

        deformed_image = random_elastic_deformation(
            image,
            self.sigma,
            self.points,
            self.order,
            self.prefilter,
            self.rotate,
            self.zoom,
            self.use_seed,
        )

        data[self.target_key] = deformed_image.to(image.device)

        return data
