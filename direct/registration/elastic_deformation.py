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

"""Random elastic deformation utilities for registration data augmentation.

The deformation matches ``elasticdeform.deform_random_grid`` / ``deform_grid``
``(BSD-licensed; Gijs van Tulder / SciPy Contributors)``. Affine rotate/zoom helpers
below are adapted from that package; the sampling backend uses SciPy instead of
the NumPy-1-era C extension so DIRECT stays NumPy-2 compatible without that
dependency.
"""

from typing import Any

import numpy as np
import torch
from scipy import ndimage

from direct.types import TransformKey
from direct.utils import DirectModule


def _compute_rotation_zoom_affine(
    angle: float | None = None,
    zoom: float | None = None,
    center: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build a 3x3 affine for 2D rotate/zoom about ``center``.

    Adapted from ``elasticdeform.deform_grid._compute_rotation_zoom_affine``.

    Args:
        angle: Angle.
        zoom: Zoom.
        center: Center.

    Returns:
        The result.
    """
    affine = None
    if center is not None:
        a = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]], dtype=np.float64)
        affine = a if affine is None else np.dot(a, affine)
    if angle:
        theta = np.radians(angle)
        a = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
            dtype=np.float64,
        )
        affine = a if affine is None else np.dot(a, affine)
    if zoom:
        a = np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, 1]], dtype=np.float64)
        affine = a if affine is None else np.dot(a, affine)
    if center is not None:
        a = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float64)
        affine = a if affine is None else np.dot(a, affine)
    return affine


def _apply_rotation_and_zoom(
    rotate: float | None,
    zoom: float | None,
    inverse_affine: np.ndarray | None,
    output_shape: list[int] | tuple[int, ...],
) -> np.ndarray | None:
    """Compose rotate/zoom into a 2x3 inverse affine.

    Adapted from ``elasticdeform.deform_grid._apply_rotation_and_zoom``.

    Args:
        rotate: Rotate.
        zoom: Zoom.
        inverse_affine: Inverse affine.
        output_shape: Output shape.

    Returns:
        The result.
    """
    if rotate is None and zoom is None:
        return inverse_affine
    if len(output_shape) != 2:
        raise ValueError("Zoom and rotate is only implemented for 2D images.")
    rotate = -float(rotate or 0)
    zoom_factor = 1 / float(zoom or 1)
    new_inverse_affine = _compute_rotation_zoom_affine(
        angle=rotate, zoom=zoom_factor, center=np.array(output_shape, dtype=np.float64) / 2 - 0.5
    )
    assert new_inverse_affine is not None
    if inverse_affine is not None:
        base_inverse_affine = np.eye(3, dtype=np.float64)
        base_inverse_affine[:-1, :] = inverse_affine
        return np.dot(new_inverse_affine, base_inverse_affine)[:2, :]
    return new_inverse_affine[:2, :]


def _deform_grid_2d(
    image: np.ndarray,
    displacement: np.ndarray,
    *,
    order: int,
    prefilter: bool,
    rotate: float | None,
    zoom: float | None,
) -> np.ndarray:
    """Apply a coarse displacement grid to a 2D image.

    Matches ``elasticdeform.deform_grid`` for 2D inputs ``(constant border mode)``.

    Args:
        image: Image.
        displacement: Displacement.
        order: Order.
        prefilter: Prefilter.
        rotate: Rotate.
        zoom: Zoom.

    Returns:
        The result.
    """
    height, width = image.shape
    points_y, points_x = displacement.shape[1], displacement.shape[2]

    # Upsample coarse displacements; numerically matches elasticdeform's spline
    # interpolation of the control grid onto the output lattice.
    dy = ndimage.zoom(displacement[0], (height / points_y, width / points_x), order=3)
    dx = ndimage.zoom(displacement[1], (height / points_y, width / points_x), order=3)

    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )

    inverse_affine = _apply_rotation_and_zoom(rotate, zoom, None, (height, width))
    if inverse_affine is not None:
        ones = np.ones_like(grid_y)
        stacked = np.stack([grid_y.ravel(), grid_x.ravel(), ones.ravel()], axis=0)
        transformed = inverse_affine @ stacked
        sample_y = transformed[0].reshape(height, width) + dy
        sample_x = transformed[1].reshape(height, width) + dx
    else:
        sample_y = grid_y + dy
        sample_x = grid_x + dx

    return ndimage.map_coordinates(
        image,
        np.stack([sample_y, sample_x], axis=0),
        order=order,
        mode="constant",
        prefilter=prefilter,
    )


def _deform_random_grid(
    images: list[np.ndarray],
    *,
    sigma: float = 25.0,
    points: int = 3,
    order: int = 3,
    prefilter: bool = True,
    rotate: float | None = None,
    zoom: float | None = None,
) -> list[np.ndarray]:
    """Random elastic deformation matching ``elasticdeform.deform_random_grid``.

    Uses the global NumPy RNG (``numpy.random.randn``) so seeding via
    ``numpy.random.seed`` matches elasticdeform bit-for-bit on the displacement.
    All images share one displacement field, as in elasticdeform.

    Args:
        images: Images.
        sigma: Sigma.
        points: Points.
        order: Order.
        prefilter: Prefilter.
        rotate: Rotate.
        zoom: Zoom.

    Returns:
        The result.
    """
    if not images:
        raise ValueError("Expected at least one image to deform.")
    height, width = images[0].shape[-2:]
    if any(img.ndim != 2 or img.shape != (height, width) for img in images):
        raise ValueError("All images must be 2D arrays with the same shape.")

    displacement = np.random.randn(2, points, points) * sigma
    return [
        _deform_grid_2d(
            image,
            displacement,
            order=order,
            prefilter=prefilter,
            rotate=rotate,
            zoom=zoom,
        ).astype(image.dtype, copy=False)
        for image in images
    ]


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

    Behaviour matches ``elasticdeform.deform_random_grid`` for the same seed,
    ``sigma``, ``points``, ``order``, ``prefilter``, ``rotate``, and ``zoom``.

    Args:
        image: Image to deform with shape ``(batch, height, width)``.
        sigma: Standard deviation of the normal distribution for the random displacements. Default is ``2.0``.
        points: Number of points of the random deformation grid. Default is ``3``.
        order: Interpolation order. Can be {``0``, ``1``, ``2``, ``3``, ``4``}. Default is ``3``.
        prefilter: If ``True`` the input will be pre-filtered with a spline filter. Default is ``True``.
        rotate: Angle in degrees to rotate the output. Default is ``None``.
        zoom: Scale factor to zoom the output. Default is ``None``.
        seed: Random seed for reproducibility. Default is ``None``.

    Returns:
        Deformed image with shape ``(batch, height, width)``.
    """
    if seed is not None:
        np.random.seed(seed)

    image_np = image.detach().cpu().numpy()
    deformed_image = _deform_random_grid(
        [*image_np],
        sigma=sigma,
        points=points,
        order=order,
        prefilter=prefilter,
        rotate=rotate,
        zoom=zoom,
    )

    return torch.from_numpy(np.array(deformed_image)).to(dtype=image.dtype)


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

        Args:
            image_key: Key of the image to deform.
            target_key: Key of the deformed image. Default is :attr:`~direct.types.TransformKey.REFERENCE_IMAGE`.
            sigma: Standard deviation of the normal distribution for the random displacements. Default is ``2.0``.
            points: Number of points of the random deformation grid. Default is ``3``.
            order: Interpolation order. Can be {``0``, ``1``, ``2``, ``3``, ``4``}. Default is ``3``.
            prefilter: If ``True`` the input will be pre-filtered with a spline filter. Default is ``True``.
            rotate: Angle in degrees to rotate the output. Default is ``None``.
            zoom: Scale factor to zoom the output. Default is ``None``.
            use_seed: Whether to use a random seed for reproducibility. Default is ``None``.

        Returns:
            ``None``.
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

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        """Apply random elastic deformation to the configured image key.

        Args:
            sample: Dictionary containing the image to deform.

        Returns:
            Dictionary with the deformed image stored under ``target_key``.
        """
        image = sample[self.image_key]

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

        sample[self.target_key] = deformed_image.to(image.device)

        return sample
