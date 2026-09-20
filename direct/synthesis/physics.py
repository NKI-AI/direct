"""Physical constraints and calibration for 2D Cartesian synthesis.

Complex tensors use DIRECT's float32 real/imaginary-last convention. Coil
arrays have shape (batch, coils, height, width, 2); magnitude is (B, 1, H, W).
Sensitivity maps live in image space, even though their shape matches k-space.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from direct.data import transforms as T
from direct.data.coil_sensitivity import SensitivityMode, simulate_coil_sensitivity_maps_batch


def validate_magnitude(magnitude: torch.Tensor) -> None:
    """Validate a nonnegative float32 magnitude batch (B, 1, H, W).

    Args:
        magnitude: Image batch, on CPU or CUDA.

    Raises:
        ValueError: If shape, dtype, or values are invalid.
    """
    if magnitude.ndim != 4 or magnitude.shape[1] != 1 or min(magnitude.shape) < 1:
        raise ValueError("magnitude must have shape (B, 1, H, W) with nonempty axes.")
    if magnitude.dtype != torch.float32 or not torch.isfinite(magnitude).all() or (magnitude < 0).any():
        raise ValueError("magnitude must contain finite, nonnegative float32 values.")


def unit_complex(value: torch.Tensor) -> torch.Tensor:
    """Normalize last-axis complex pairs, mapping zero pairs to (1, 0).

    Args:
        value: Finite float32 tensor with last dimension two.

    Returns:
        Unit complex pairs with the input shape and device.
    """
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    identity = torch.zeros_like(value)
    identity[..., 0] = 1
    return torch.where(norm > 1e-8, value / norm.clamp_min(1e-8), identity)


def normalize_sensitivities(maps: torch.Tensor) -> torch.Tensor:
    """Enforce sum-of-coil power one, including outside calibration support.

    Args:
        maps: Finite float32 sensitivity maps (B, C, H, W, 2).

    Returns:
        Normalized maps. Zero-power pixels receive uniform real sensitivities.

    Raises:
        ValueError: If shape, dtype, or values are invalid.
    """
    if maps.ndim != 5 or maps.shape[-1] != 2 or min(maps.shape) < 1:
        raise ValueError("maps must have shape (B, C, H, W, 2).")
    if maps.dtype != torch.float32 or not torch.isfinite(maps).all():
        raise ValueError("maps must contain finite float32 values.")
    norm = maps.square().sum(-1).sum(1, keepdim=True).sqrt().unsqueeze(-1)
    fallback = torch.zeros_like(maps)
    fallback[..., 0] = maps.shape[1] ** -0.5
    return torch.where(norm > 1e-8, maps / norm.clamp_min(1e-8), fallback)


@dataclass
class SynthesisOutput:
    """Synthetic signals and latent factors, all on the input image grid.

    Attributes:
        kspace: Clean centered orthonormal k-space (B, C, H, W, 2).
        sensitivity_map: Normalized image-domain maps (B, C, H, W, 2).
        phase: Unit complex object phase (B, H, W, 2).
        magnitude: Unmodified conditioning magnitude (B, 1, H, W).
    """

    kspace: torch.Tensor
    sensitivity_map: torch.Tensor
    phase: torch.Tensor
    magnitude: torch.Tensor


def synthesize(magnitude: torch.Tensor, phase: torch.Tensor, maps: torch.Tensor) -> SynthesisOutput:
    r"""Compute k_c = FFT(S_c m exp(i phi)) with an exact clean RSS constraint.

    Args:
        magnitude: Nonnegative images (B, 1, H, W).
        phase: Real/imaginary phase pairs (B, H, W, 2), normalized internally.
        maps: Sensitivity maps (B, C, H, W, 2), normalized internally.

    Returns:
        Clean multicoil data whose inverse-FFT RSS equals the input magnitude.

    Raises:
        ValueError: If dimensions, devices, dtypes, or values are incompatible.
    """
    validate_magnitude(magnitude)
    expected = (magnitude.shape[0], *magnitude.shape[2:], 2)
    if phase.shape != expected or phase.dtype != torch.float32 or not torch.isfinite(phase).all():
        raise ValueError(f"phase must be a finite float32 tensor with shape {expected}.")
    maps = normalize_sensitivities(maps)
    if maps.shape[0] != expected[0] or maps.shape[2:] != expected[1:]:
        raise ValueError("maps and magnitude must share batch and spatial dimensions.")
    if phase.device != magnitude.device or maps.device != magnitude.device:
        raise ValueError("magnitude, phase, and maps must share a device.")
    phase = unit_complex(phase)
    image = magnitude[:, 0, ..., None] * phase
    coils = T.expand_operator(image, maps, dim=1)
    return SynthesisOutput(T.fft2(coils, dim=(2, 3)), maps, phase, magnitude)


def smooth_phase(magnitude: torch.Tensor, generator: torch.Generator, scale: float = 2.0) -> torch.Tensor:
    """Sample smooth phase fields plus a random global phase.

    Args:
        magnitude: Images (B, 1, H, W) specifying shape and device.
        generator: Torch generator on the same device.
        scale: Nonnegative standard scale of the coarse phase in radians.

    Returns:
        Unit complex phase (B, H, W, 2).

    Raises:
        ValueError: If scale or magnitude is invalid.
    """
    validate_magnitude(magnitude)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("phase scale must be finite and nonnegative.")
    coarse = torch.randn(magnitude.shape[0], 1, 5, 5, device=magnitude.device, generator=generator)
    angle = F.interpolate(coarse, magnitude.shape[-2:], mode="bicubic", align_corners=False) * scale
    offset = torch.rand(magnitude.shape[0], 1, 1, 1, device=magnitude.device, generator=generator) * (2 * torch.pi)
    angle = (angle + offset)[:, 0]
    return torch.stack((angle.cos(), angle.sin()), -1)


def simulated_maps(
    magnitude: torch.Tensor,
    num_coils: int,
    seed: int,
    *,
    mode: SensitivityMode = "surface",
    coil_radius: float = 1.5,
    coil_size: float = 0.55,
    falloff_power: float = 1.5,
    phase_strength: float = 0.7,
    angular_jitter: float = 0.08,
    biot_savart_segments: int = 96,
    reference_maps: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Simulate physically motivated receive-coil sensitivity maps.

    Args:
        magnitude: Images (B, 1, H, W); maps are broadcast across this batch.
        num_coils: Positive number of simulated coils.
        seed: Nonnegative simulation seed, including zero.
        mode: ``birdcage``, ``surface``, ``biot_savart``, or ``empirical``.
        coil_radius: Coil-center radius relative to the normalized FOV.
        coil_size: Loop radius in Biot-Savart mode.
        falloff_power: Distance falloff exponent in surface mode.
        phase_strength: Smooth spatial phase strength in surface mode.
        angular_jitter: Fractional angular jitter in surface mode.
        biot_savart_segments: Line segments per loop in Biot-Savart mode.
        reference_maps: Required in empirical mode, shape ``(C, H, W, 2)``.
        normalize: If true, enforce pointwise RSS normalization.

    Returns:
        Sensitivities (B, C, H, W, 2) with coil-specific magnitude and phase.

    Raises:
        ValueError: If coil count, seed, or magnitude is invalid.
    """
    validate_magnitude(magnitude)
    if num_coils < 1 or not 0 <= seed < 2**32:
        raise ValueError("num_coils must be positive and seed must be in [0, 2**32).")
    maps = simulate_coil_sensitivity_maps_batch(
        magnitude,
        num_coils,
        mode=mode,
        reference_maps=reference_maps,
        coil_radius=coil_radius,
        coil_size=coil_size,
        falloff_power=falloff_power,
        phase_strength=phase_strength,
        angular_jitter=angular_jitter,
        biot_savart_segments=biot_savart_segments,
        seed=seed,
        normalize=normalize,
    )
    return normalize_sensitivities(maps) if normalize else maps


def remove_readout_oversampling(kspace: torch.Tensor) -> torch.Tensor:
    """Remove 2x readout oversampling from centered k-space.

    Many MRI acquisitions (e.g. fastMRI) have 2x readout oversampling along
    the first spatial dimension.  This crops the readout dimension in image
    space to the central half, then transforms back to k-space.

    Args:
        kspace: Centered float32 k-space ``(B, C, H, W, 2)`` where ``H`` is the
            readout dimension.

    Returns:
        Cropped k-space ``(B, C, H//2, W, 2)`` if ``H > W``, otherwise unchanged.
    """
    height, width = kspace.shape[2:4]
    if height <= width:
        return kspace
    # Go to image space, crop readout, go back
    coil_images = T.ifft2(kspace, dim=(2, 3))
    crop_h = height // 2
    start = (height - crop_h) // 2
    coil_images_cropped = coil_images[:, :, start : start + crop_h, :, :]
    return T.fft2(coil_images_cropped, dim=(2, 3))


@torch.no_grad()
def calibration_targets(
    kspace: torch.Tensor,
    acs_size: int = 24,
    crop_readout: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract pseudo-labels from fully sampled data with DIRECT's ACS estimator.

    The strongest integrated-power reference coil fixes the local map gauge.
    Object phase is aligned to zero at the most reliable bright pixel, removing
    an unidentifiable global receiver phase. These are single-map SENSE
    approximations, not measured physical maps.

    Args:
        kspace: Fully sampled centered float32 k-space (B, C, H, W, 2).
        acs_size: Width of a centered square calibration region.
        crop_readout: If True and ``H > W``, remove 2x readout oversampling
            before calibration.  This typically improves projection NMSE on
            real data (e.g. fastMRI knee: 640×372 → 320×372).

    Returns:
        Magnitude, unit phase, maps, support weights, and per-example relative
        coil-image projection error. Error measures the single-map approximation.

    Raises:
        ValueError: If data or calibration dimensions are invalid.
    """
    from direct.data.mri_transforms import EstimateSensitivityMapModule

    if kspace.ndim != 5 or kspace.shape[-1] != 2 or kspace.dtype != torch.float32:
        raise ValueError("kspace must have shape (B, C, H, W, 2) and dtype float32.")
    if not torch.isfinite(kspace).all():
        raise ValueError("kspace must be finite.")

    if crop_readout:
        kspace = remove_readout_oversampling(kspace)

    if not 2 <= acs_size <= min(kspace.shape[2:4]):
        raise ValueError("ACS size must fit the (possibly cropped) image (at least two).")
    acs = torch.zeros_like(kspace)
    h0, w0 = [(size - acs_size + 1) // 2 for size in kspace.shape[2:4]]
    acs[:, :, h0 : h0 + acs_size, w0 : w0 + acs_size] = kspace[:, :, h0 : h0 + acs_size, w0 : w0 + acs_size]
    maps = EstimateSensitivityMapModule()({"acs_kspace": acs})["sensitivity_map"]
    coil_power = maps.square().sum((2, 3, 4))
    reference = (coil_power >= 0.999 * coil_power.amax(1, keepdim=True)).to(torch.int64).argmax(1)
    reference_map = maps[torch.arange(maps.shape[0], device=maps.device), reference]
    gauge = unit_complex(reference_map)
    maps = normalize_sensitivities(T.complex_multiplication(maps, T.conjugate(gauge[:, None])))
    coils = T.ifft2(kspace, dim=(2, 3))
    magnitude = T.root_sum_of_squares(coils, dim=1)[:, None]
    combined = T.reduce_operator(coils, maps, dim=1)
    phase = unit_complex(combined)
    # Background and weak reference-coil pixels have unreliable phase labels.
    support = magnitude / magnitude.amax((2, 3), keepdim=True).clamp_min(1e-8)
    support = support * (reference_map.square().sum(-1)[:, None] > 1e-4)
    projected = T.expand_operator(magnitude[:, 0, ..., None] * phase, maps, dim=1)
    error = (projected - coils).square().sum((1, 2, 3, 4)) / coils.square().sum((1, 2, 3, 4)).clamp_min(1e-8)
    reliability = (support[:, 0] * magnitude[:, 0]).flatten(1)
    # A fixed first pixel on the bright plateau avoids changing the gauge when
    # equivalent FFTs perturb nearly tied maxima at floating-point precision.
    plateau = reliability >= 0.999 * reliability.amax(1, keepdim=True)
    anchor = plateau.to(torch.int64).argmax(1)
    anchor_phase = phase.flatten(1, 2)[torch.arange(phase.shape[0], device=phase.device), anchor]
    phase = T.complex_multiplication(phase, T.conjugate(anchor_phase[:, None, None]))
    return {
        "magnitude": magnitude,
        "phase": phase,
        "sensitivity_map": maps,
        "weight": support,
        "projection_nmse": error,
    }


def synthesis_diagnostics(output: SynthesisOutput) -> dict[str, float]:
    """Measure physical consistency, without asserting realism.

    Args:
        output: Synthesis result.

    Returns:
        RSS relative error, map normalization error, and Fourier roundtrip error.
    """
    coils = T.ifft2(output.kspace, dim=(2, 3))
    rss = T.root_sum_of_squares(coils, dim=1)[:, None]
    denom = output.magnitude.square().sum().clamp_min(1e-12)
    power = output.sensitivity_map.square().sum(-1).sum(1)
    roundtrip = T.fft2(coils, dim=(2, 3))
    return {
        "rss_nmse": float(((rss - output.magnitude).square().sum() / denom).detach()),
        "map_power_max_error": float((power - 1).abs().max().detach()),
        "fft_nmse": float(
            ((roundtrip - output.kspace).square().sum() / output.kspace.square().sum().clamp_min(1e-12)).detach()
        ),
    }
