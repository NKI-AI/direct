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

"""Simulation and reuse of complex MRI receive-coil sensitivity maps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

SensitivityMode = Literal["birdcage", "surface", "biot_savart", "empirical"]


def simulate_coil_sensitivity_maps(
    shape: Sequence[int],
    num_coils: int,
    *,
    mode: SensitivityMode = "surface",
    reference_maps: Tensor | None = None,
    coil_radius: float = 1.5,
    coil_size: float = 0.55,
    falloff_power: float = 1.5,
    phase_strength: float = 0.7,
    angular_jitter: float = 0.08,
    biot_savart_segments: int = 96,
    seed: int | None = None,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Generate or reuse complex 2D MRI receive-coil sensitivity maps.

    The analytical modes place coils around a normalized field of view spanning
    ``[-1, 1]`` along both axes. ``empirical`` mode resizes real sensitivity
    maps and applies SVD coil compression when fewer output coils are requested.

    Args:
        shape: Spatial output shape ``(x, y)``.
        num_coils: Number of output coils.
        mode: Map-generation method:

            * ``"birdcage"`` reproduces the standard inverse-distance model
              used by SigPy.
            * ``"surface"`` generates randomized localized surface-coil maps.
            * ``"biot_savart"`` numerically integrates circular loop fields.
            * ``"empirical"`` reuses maps supplied through ``reference_maps``.
        reference_maps: Complex maps with shape ``(coils, x, y)`` or real-view
            maps with shape ``(coils, x, y, 2)``. Required in empirical mode.
        coil_radius: Radius of the coil centers relative to the normalized FOV.
        coil_size: Radius of each circular loop in Biot-Savart mode.
        falloff_power: Distance falloff exponent in surface mode.
        phase_strength: Strength of smooth spatial phase in surface mode.
        angular_jitter: Maximum angular displacement as a fraction of uniform
            inter-coil spacing in surface mode.
        biot_savart_segments: Number of line segments used for each coil loop.
        seed: Optional random seed for reproducible surface-coil maps.
        normalize: If true, enforce pointwise root-sum-of-squares normalization.
        dtype: Real output dtype, either ``torch.float32`` or ``torch.float64``.
        device: Output device. Defaults to CPU.

    Returns:
        Sensitivity maps with shape ``(num_coils, x, y, 2)``, where the final
        dimension stores real and imaginary components.

    Raises:
        TypeError: If ``dtype`` or ``reference_maps`` has an unsupported type.
        ValueError: If an argument or input shape is invalid.
    """
    _validate_inputs(
        shape=shape,
        num_coils=num_coils,
        mode=mode,
        reference_maps=reference_maps,
        coil_radius=coil_radius,
        coil_size=coil_size,
        falloff_power=falloff_power,
        phase_strength=phase_strength,
        angular_jitter=angular_jitter,
        biot_savart_segments=biot_savart_segments,
        dtype=dtype,
    )

    spatial_shape = (int(shape[0]), int(shape[1]))
    output_device = torch.device("cpu" if device is None else device)

    if mode == "empirical":
        assert reference_maps is not None
        maps = _prepare_empirical_maps(
            reference_maps,
            shape=spatial_shape,
            num_coils=num_coils,
            dtype=dtype,
            device=output_device,
        )
    else:
        grid_x, grid_y = _coordinate_grid(
            spatial_shape,
            dtype=dtype,
            device=output_device,
        )
        if mode == "birdcage":
            maps = _birdcage_maps(grid_x, grid_y, num_coils, coil_radius)
        elif mode == "surface":
            maps = _surface_maps(
                grid_x,
                grid_y,
                num_coils=num_coils,
                coil_radius=coil_radius,
                falloff_power=falloff_power,
                phase_strength=phase_strength,
                angular_jitter=angular_jitter,
                seed=seed,
            )
        else:
            maps = _biot_savart_maps(
                grid_x,
                grid_y,
                num_coils=num_coils,
                coil_radius=coil_radius,
                coil_size=coil_size,
                num_segments=biot_savart_segments,
            )

    if normalize:
        maps = _rss_normalize(maps)

    return torch.view_as_real(maps)


def simulate_coil_sensitivity_maps_batch(
    magnitude: Tensor,
    num_coils: int,
    **kwargs,
) -> Tensor:
    """Broadcast simulated maps across a magnitude batch.

    Args:
        magnitude: Images ``(B, 1, H, W)``.
        num_coils: Number of output coils.
        **kwargs: Forwarded to :func:`simulate_coil_sensitivity_maps`.

    Returns:
        Maps with shape ``(B, num_coils, H, W, 2)`` on ``magnitude``'s device.
    """
    maps = simulate_coil_sensitivity_maps(
        magnitude.shape[-2:],
        num_coils,
        dtype=torch.float32,
        device="cpu",
        **kwargs,
    )
    return maps.to(device=magnitude.device, dtype=torch.float32)[None].expand(
        magnitude.shape[0], -1, -1, -1, -1
    ).contiguous()


def _validate_inputs(
    *,
    shape: Sequence[int],
    num_coils: int,
    mode: str,
    reference_maps: Tensor | None,
    coil_radius: float,
    coil_size: float,
    falloff_power: float,
    phase_strength: float,
    angular_jitter: float,
    biot_savart_segments: int,
    dtype: torch.dtype,
) -> None:
    if len(shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in shape
    ):
        raise ValueError(f"shape must contain two positive integers, got {shape}.")
    if not isinstance(num_coils, int) or isinstance(num_coils, bool) or num_coils <= 0:
        raise ValueError(f"num_coils must be a positive integer, got {num_coils}.")
    if mode not in {"birdcage", "surface", "biot_savart", "empirical"}:
        raise ValueError(f"Unsupported sensitivity-map mode: {mode!r}.")
    if mode == "empirical" and reference_maps is None:
        raise ValueError("reference_maps is required when mode='empirical'.")
    if reference_maps is not None and not isinstance(reference_maps, Tensor):
        raise TypeError("reference_maps must be a torch.Tensor.")
    if coil_radius <= 1.0:
        raise ValueError("coil_radius must be greater than 1.0.")
    if coil_size <= 0.0:
        raise ValueError("coil_size must be positive.")
    if falloff_power <= 0.0:
        raise ValueError("falloff_power must be positive.")
    if phase_strength < 0.0:
        raise ValueError("phase_strength must be non-negative.")
    if angular_jitter < 0.0:
        raise ValueError("angular_jitter must be non-negative.")
    if biot_savart_segments < 8:
        raise ValueError("biot_savart_segments must be at least 8.")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64.")


def _coordinate_grid(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    x = torch.linspace(-1.0, 1.0, shape[0], dtype=dtype, device=device)
    y = torch.linspace(-1.0, 1.0, shape[1], dtype=dtype, device=device)
    return torch.meshgrid(x, y, indexing="ij")


def _birdcage_maps(
    grid_x: Tensor,
    grid_y: Tensor,
    num_coils: int,
    coil_radius: float,
) -> Tensor:
    angles = (
        2.0
        * math.pi
        * torch.arange(num_coils, dtype=grid_x.dtype, device=grid_x.device)
        / num_coils
    )
    maps = []
    for angle in angles:
        delta_x = grid_x - coil_radius * torch.cos(angle)
        delta_y = grid_y - coil_radius * torch.sin(angle)
        distance = torch.sqrt(delta_x.square() + delta_y.square()).clamp_min(1e-6)
        phase = torch.atan2(delta_x, -delta_y) - angle
        maps.append(torch.polar(distance.reciprocal(), phase))
    return torch.stack(maps)


def _surface_maps(
    grid_x: Tensor,
    grid_y: Tensor,
    *,
    num_coils: int,
    coil_radius: float,
    falloff_power: float,
    phase_strength: float,
    angular_jitter: float,
    seed: int | None,
) -> Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    angle_noise = 2.0 * torch.rand(num_coils, generator=generator) - 1.0
    gains = 0.9 + 0.2 * torch.rand(num_coils, generator=generator)
    phase_offsets = 2.0 * math.pi * torch.rand(num_coils, generator=generator)

    angle_step = 2.0 * math.pi / num_coils
    angles = angle_step * torch.arange(num_coils) + (
        angular_jitter * angle_step * angle_noise
    )
    angles = angles.to(dtype=grid_x.dtype, device=grid_x.device)
    gains = gains.to(dtype=grid_x.dtype, device=grid_x.device)
    phase_offsets = phase_offsets.to(dtype=grid_x.dtype, device=grid_x.device)

    maps = []
    for angle, gain, phase_offset in zip(angles, gains, phase_offsets):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        coil_x = coil_radius * cos_angle
        coil_y = coil_radius * sin_angle

        distance_squared = (
            (grid_x - coil_x).square() + (grid_y - coil_y).square()
        )
        magnitude = gain / distance_squared.clamp_min(1e-6).pow(
            falloff_power / 2.0
        )

        radial = grid_x * cos_angle + grid_y * sin_angle
        tangential = -grid_x * sin_angle + grid_y * cos_angle
        phase = (
            phase_offset
            + math.pi * phase_strength * tangential
            + 0.15 * math.pi * radial * tangential
        )
        maps.append(torch.polar(magnitude, phase))

    return torch.stack(maps)


def _biot_savart_maps(
    grid_x: Tensor,
    grid_y: Tensor,
    *,
    num_coils: int,
    coil_radius: float,
    coil_size: float,
    num_segments: int,
) -> Tensor:
    dtype = grid_x.dtype
    device = grid_x.device
    phi = (
        2.0
        * math.pi
        * torch.arange(num_segments, dtype=dtype, device=device)
        / num_segments
    )
    dphi = 2.0 * math.pi / num_segments
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    zeros = torch.zeros_like(grid_x)
    field_points = torch.stack((grid_x, grid_y, zeros), dim=-1)
    angles = (
        2.0
        * math.pi
        * torch.arange(num_coils, dtype=dtype, device=device)
        / num_coils
    )

    maps = []
    for angle in angles:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        center = torch.stack(
            (
                coil_radius * cos_angle,
                coil_radius * sin_angle,
                torch.zeros((), dtype=dtype, device=device),
            )
        )
        tangent = torch.stack(
            (
                -sin_angle,
                cos_angle,
                torch.zeros((), dtype=dtype, device=device),
            )
        )
        z_axis = torch.tensor((0.0, 0.0, 1.0), dtype=dtype, device=device)

        loop_points = center + coil_size * (
            cos_phi[:, None] * tangent + sin_phi[:, None] * z_axis
        )
        line_elements = coil_size * dphi * (
            -sin_phi[:, None] * tangent + cos_phi[:, None] * z_axis
        )

        displacement = field_points[..., None, :] - loop_points
        distance_cubed = displacement.square().sum(dim=-1).clamp_min(1e-12).pow(1.5)
        expanded_elements = line_elements.view(1, 1, num_segments, 3).expand_as(
            displacement
        )
        field = (
            torch.cross(expanded_elements, displacement, dim=-1)
            / distance_cubed[..., None]
        ).sum(dim=-2)

        # By reciprocity, receive sensitivity is proportional to B1- = Bx - i By.
        maps.append(torch.complex(field[..., 0], -field[..., 1]))

    return torch.stack(maps)


def _prepare_empirical_maps(
    reference_maps: Tensor,
    *,
    shape: tuple[int, int],
    num_coils: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if reference_maps.is_complex():
        if reference_maps.ndim != 3:
            raise ValueError(
                "Complex reference_maps must have shape (coils, x, y)."
            )
        maps = reference_maps
    else:
        if reference_maps.ndim != 4 or reference_maps.shape[-1] != 2:
            raise ValueError(
                "Real reference_maps must have shape (coils, x, y, 2)."
            )
        components = reference_maps.to(device=device, dtype=dtype).contiguous()
        maps = torch.view_as_complex(components)

    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    maps = maps.to(device=device, dtype=complex_dtype)
    input_coils = maps.shape[0]
    if num_coils > input_coils:
        raise ValueError(
            f"Cannot obtain {num_coils} empirical coils from {input_coils} input coils."
        )

    if maps.shape[-2:] != shape:
        components = torch.view_as_real(maps).permute(0, 3, 1, 2)
        components = F.interpolate(
            components,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )
        maps = torch.view_as_complex(
            components.permute(0, 2, 3, 1).contiguous()
        )

    if num_coils < input_coils:
        flattened = maps.reshape(input_coils, -1)
        left_vectors, _, _ = torch.linalg.svd(flattened, full_matrices=False)
        compression = left_vectors[:, :num_coils].mH
        maps = (compression @ flattened).reshape(num_coils, *shape)

    return maps


def _rss_normalize(maps: Tensor) -> Tensor:
    rss = maps.abs().square().sum(dim=0, keepdim=True).sqrt()
    epsilon = torch.finfo(maps.real.dtype).eps
    return maps / rss.clamp_min(epsilon)
