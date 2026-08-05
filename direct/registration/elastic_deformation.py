from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from elasticdeform import deform_random_grid

from direct.types import TransformKey
from direct.utils import DirectModule


def random_elastic_deformation(
    image: torch.Tensor,
    sigma: float = 2.0,
    points: int = 3,
    order: int = 3,
    prefilter=True,
    rotate: Optional[float] = None,
    zoom: Optional[float] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Elastic deformation with a random deformation grid

    This generates a random, square deformation grid with displacements
    sampled from from a normal distribution with standard deviation `sigma`.
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
    zoom : float. optional
        Scale factor to zoom the output. Default: None.

    Returns
    -------
    torch.Tensor
        Deformed image with shape (batch, height, width).
    """
    if seed is not None:
        np.random.seed(seed)

    deformed_image = deform_random_grid(
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
    """Module for applying random elastic deformation to an image.

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
    zoom : float. optional
        Scale factor to zoom the output. Default: None.
    seed : int, optional
        Random seed for reproducibility. Default: None.
    """

    def __init__(
        self,
        image_key: TransformKey,
        target_key: TransformKey = TransformKey.REFERENCE_IMAGE,
        sigma: float = 2.0,
        points: int = 3,
        order: int = 3,
        prefilter=True,
        rotate: Optional[float] = None,
        zoom: Optional[float] = None,
        use_seed: Optional[bool] = None,
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
        zoom : float. optional
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
        """Forward pass of the random elastic deformation module.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing the image to deform with key 'image' of

        Returns
        -------
        dict[str, Any]
            Dictionary containing the deformed image with key 'image'.
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
