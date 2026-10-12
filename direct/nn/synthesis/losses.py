"""Scientifically motivated losses for magnitude-conditioned MRI synthesis.

Phase lives on the circle S¹. Regressing wrapped radians with Euclidean MSE is
discontinuous at ±π; the standard fix in circular statistics and MRI deep
learning is to embed phase as a unit complex ``u = (cos φ, sin φ)`` and use the
**chordal distance**

    ‖û − u‖²₂ = 2(1 − cos Δφ) ∈ [0, 4],

which is smooth, 2π-periodic, and equal (up to scale) to ``1 − Re(û conj(u))``.

Under approximate complex Gaussian noise, Fisher information for phase scales
as |z|², so phase errors should be weighted by signal intensity (here: RSS
magnitude × soft tissue support). Background phase is not identifiable and
must not dominate the gradient.

Complex-image regression in ℂ ≅ ℝ² is Parseval-equivalent (orthonormal FFT) to
single-image k-space MSE. Splitting into magnitude and chordal-phase terms
improves conditioning: intensity is better determined by the input than phase.

These are **calibrated pseudo-label** losses (SENSE / ACS gauge), not recovery
of unique physical fields. Magnitude alone does not identify phase; a
deterministic network trained with MSE estimates a posterior mean and can look
blurry or mediocre even when the loss is correctly specified. Distributional
models (e.g. conditional phase diffusion) address that multimodality.
"""

from __future__ import annotations

import torch

from direct.synthesis.physics import unit_complex


def _as_hw_weight(weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast weight to ``(B, H, W)`` matching a complex ``(B, H, W, 2)`` tensor."""
    if weight.ndim == 4:
        weight = weight[:, 0]
    if weight.shape != reference.shape[:3]:
        raise ValueError(f"weight shape {tuple(weight.shape)} incompatible with {tuple(reference.shape)}.")
    return weight


def snr_phase_weight(
    magnitude: torch.Tensor,
    support: torch.Tensor | None = None,
    power: float = 2.0,
) -> torch.Tensor:
    """Fisher-inspired phase weight ``∝ m^power`` on tissue support.

    Args:
        magnitude: Nonneg ``(B, 1, H, W)`` or ``(B, H, W)``.
        support: Optional soft mask ``(B, 1, H, W)`` / ``(B, H, W)`` (e.g. from
            :class:`~direct.data.mri_transforms.ExtractCalibrationTargetsModule`).
        power: Exponent; ``2`` matches asymptotic phase variance ``∝ 1/SNR``.

    Returns:
        Weight ``(B, H, W)``.
    """
    if magnitude.ndim == 4:
        magnitude = magnitude[:, 0]
    peak = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    weight = (magnitude / peak).clamp_min(0.0).pow(power)
    if support is not None:
        if support.ndim == 4:
            support = support[:, 0]
        weight = weight * support
    return weight


def weighted_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of a nonnegative per-pixel error map."""
    weight = weight.expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)


def chordal_phase_loss(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Support-weighted chordal loss on unit-complex phase.

    ``pred_phase`` / ``target_phase`` are ``(B, H, W, 2)``. Predictions are
    re-normalized so the loss always measures distance on S¹.
    """
    pred = unit_complex(pred_phase)
    target = unit_complex(target_phase)
    weight = _as_hw_weight(weight, pred)
    return weighted_mean((pred - target).square().sum(-1), weight)


def cosine_phase_loss(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """``1 − cos Δφ`` = chordal / 2; convenient for logging in [0, 2]."""
    pred = unit_complex(pred_phase)
    target = unit_complex(target_phase)
    weight = _as_hw_weight(weight, pred)
    cos = (pred * target).sum(-1).clamp(-1.0, 1.0)
    return weighted_mean(1.0 - cos, weight)


def mean_angular_error_deg(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean absolute angular error in degrees (monitor only)."""
    pred = unit_complex(pred_phase)
    target = unit_complex(target_phase)
    weight = _as_hw_weight(weight, pred)
    cos = (pred * target).sum(-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos)  # [0, π]
    return weighted_mean(angle, weight) * (180.0 / torch.pi)


def phase_gradient_loss(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Match finite-difference gradients of unit phase (local structure / B0).

    Encourages correct spatial derivatives of ``(cos φ, sin φ)`` away from
    wraps; still chordal in the embedding space, not on unwrapped radians.
    """
    pred = unit_complex(pred_phase)
    target = unit_complex(target_phase)
    weight = _as_hw_weight(weight, pred)

    def _grad_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dh = (a[:, 1:, :, :] - a[:, :-1, :, :]) - (b[:, 1:, :, :] - b[:, :-1, :, :])
        dw = (a[:, :, 1:, :] - a[:, :, :-1, :]) - (b[:, :, 1:, :] - b[:, :, :-1, :])
        err_h = dh.square().sum(-1)
        err_w = dw.square().sum(-1)
        w_h = 0.5 * (weight[:, 1:, :] + weight[:, :-1, :])
        w_w = 0.5 * (weight[:, :, 1:] + weight[:, :, :-1])
        return weighted_mean(err_h, w_h) + weighted_mean(err_w, w_w)

    return _grad_err(pred, target)


def complex_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted ‖ẑ − z‖²₂ over ℂ ≅ ℝ² (Parseval ↔ image-domain MSE)."""
    weight = _as_hw_weight(weight, pred)
    return weighted_mean((pred - target).square().sum(-1), weight)


def magnitude_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE between complex magnitudes ``|ẑ|`` and ``|z|``."""
    weight = _as_hw_weight(weight, pred)
    pred_mag = pred.square().sum(-1).clamp_min(0.0).sqrt()
    target_mag = target.square().sum(-1).clamp_min(0.0).sqrt()
    return weighted_mean((pred_mag - target_mag).square(), weight)


def phase_supervision_losses(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    magnitude: torch.Tensor,
    support: torch.Tensor | None,
    gradient_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Full phase objective used by :class:`PhaseFromMagnitudeEngine`."""
    weight = snr_phase_weight(magnitude, support, power=2.0)
    chordal = chordal_phase_loss(pred_phase, target_phase, weight)
    losses = {
        "phase_loss": chordal,
        "phase_cosine": cosine_phase_loss(pred_phase, target_phase, weight),
        "phase_mae_deg": mean_angular_error_deg(pred_phase, target_phase, weight),
    }
    if gradient_weight > 0.0:
        grad = phase_gradient_loss(pred_phase, target_phase, weight)
        losses["phase_grad_loss"] = grad
        losses["loss"] = chordal + gradient_weight * grad
    else:
        losses["loss"] = chordal
    return losses


def complex_supervision_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    magnitude: torch.Tensor,
    support: torch.Tensor | None,
    phase_weight: float = 1.0,
    magnitude_weight: float = 1.0,
    complex_weight: float = 1.0,
    gradient_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Complex + magnitude + chordal-phase objective for pipeline B."""
    weight = snr_phase_weight(magnitude, support, power=1.0)
    phase_w = snr_phase_weight(magnitude, support, power=2.0)
    cpx = complex_l2_loss(pred, target, weight)
    mag = magnitude_l2_loss(pred, target, weight)
    pred_phase = unit_complex(pred)
    target_phase = unit_complex(target)
    phase = chordal_phase_loss(pred_phase, target_phase, phase_w)
    losses = {
        "complex_loss": cpx,
        "mag_loss": mag,
        "phase_loss": phase,
        "phase_mae_deg": mean_angular_error_deg(pred_phase, target_phase, phase_w),
    }
    total = complex_weight * cpx + magnitude_weight * mag + phase_weight * phase
    if gradient_weight > 0.0:
        grad = phase_gradient_loss(pred_phase, target_phase, phase_w)
        losses["phase_grad_loss"] = grad
        total = total + gradient_weight * grad
    losses["loss"] = total
    return losses


def rss_unity_error(maps: torch.Tensor) -> torch.Tensor:
    """Mean absolute deviation of pointwise coil power from 1."""
    power = maps.square().sum(-1).sum(1)
    return (power - 1.0).abs().mean()


def _greedy_coil_permutation(pred_mag: torch.Tensor, target_mag: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Assign each target coil the unused predicted coil with highest weighted cosine."""
    batch, coils, height, width = pred_mag.shape
    weight = weight[:, None]
    pred_flat = (pred_mag * weight).reshape(batch, coils, height * width)
    target_flat = (target_mag * weight).reshape(batch, coils, height * width)
    similarity = torch.matmul(pred_flat, target_flat.transpose(-1, -2))
    pred_norm = pred_flat.norm(dim=-1).clamp_min(1e-8)
    target_norm = target_flat.norm(dim=-1).clamp_min(1e-8)
    similarity = similarity / (pred_norm[:, :, None] * target_norm[:, None, :])
    permutations = []
    scores = similarity.detach()
    for batch_idx in range(batch):
        remaining = set(range(coils))
        order = torch.empty(coils, dtype=torch.long, device=pred_mag.device)
        for target_idx in range(coils):
            column = scores[batch_idx, :, target_idx].clone()
            for pred_idx in range(coils):
                if pred_idx not in remaining:
                    column[pred_idx] = -1e9
            chosen = int(column.argmax())
            remaining.remove(chosen)
            order[target_idx] = chosen
        permutations.append(order)
    return torch.stack(permutations, dim=0)


def align_sensitivity_maps(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Permute predicted coils to the target set, then remove per-coil phase gauge."""
    if pred.shape != target.shape:
        raise ValueError("pred and target maps must share shape (B, C, H, W, 2).")
    perm = _greedy_coil_permutation(
        torch.linalg.vector_norm(pred, dim=-1),
        torch.linalg.vector_norm(target, dim=-1),
        weight,
    )
    gather = perm[:, :, None, None, None].expand(-1, -1, pred.shape[2], pred.shape[3], 2)
    matched = pred.gather(1, gather)
    support = weight[:, None, :, :, None]
    real = (matched[..., 0] * target[..., 0] + matched[..., 1] * target[..., 1]) * support[..., 0]
    imag = (matched[..., 1] * target[..., 0] - matched[..., 0] * target[..., 1]) * support[..., 0]
    z_real = real.sum((-2, -1), keepdim=True)
    z_imag = imag.sum((-2, -1), keepdim=True)
    scale = torch.sqrt(z_real.square() + z_imag.square()).clamp_min(1e-8)
    cos_g = z_real / scale
    sin_g = -z_imag / scale
    aligned_real = matched[..., 0] * cos_g - matched[..., 1] * sin_g
    aligned_imag = matched[..., 0] * sin_g + matched[..., 1] * cos_g
    return torch.stack((aligned_real, aligned_imag), dim=-1)


def map_supervision_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    magnitude: torch.Tensor,
    support: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Complex + magnitude map losses after coil matching and phase alignment."""
    weight = snr_phase_weight(magnitude, support, power=1.0)
    aligned = align_sensitivity_maps(pred, target, weight)
    complex_err = (aligned - target).square().sum(-1).sum(1)
    mag_err = (torch.linalg.vector_norm(aligned, dim=-1) - torch.linalg.vector_norm(target, dim=-1)).square().sum(1)
    complex_loss = weighted_mean(complex_err, weight)
    mag_loss = weighted_mean(mag_err, weight)
    rss_err = rss_unity_error(pred)
    return {
        "map_loss": complex_loss,
        "map_mag_loss": mag_loss,
        "map_mae": weighted_mean((aligned - target).abs().sum(-1).sum(1), weight),
        "rss_err": rss_err,
        "aligned": aligned,
        "loss": complex_loss + mag_loss,
    }


def map_smoothness_loss(maps: torch.Tensor, penalty: str = "l1") -> torch.Tensor:
    """Voxelmorph / displacement-field style spatial smoothness on complex maps.

    Treats ``(B, C, H, W, 2)`` maps as a multi-channel field ``(B, C·2, H, W)``
    and penalizes finite-difference gradients (L1 or L2), matching
    ``displacement_field_smooth_loss_l1`` used for Demons deformation fields.
    """
    from direct.functionals.smooth import SmoothLossL1, SmoothLossL2

    if maps.ndim != 5 or maps.shape[-1] != 2:
        raise ValueError(f"Expected complex maps (B, C, H, W, 2), got {tuple(maps.shape)}")
    batch, coils, height, width, complex_dim = maps.shape
    field = maps.permute(0, 1, 4, 2, 3).reshape(batch, coils * complex_dim, height, width)
    if penalty == "l1":
        return SmoothLossL1(reduction="mean")(field)
    if penalty == "l2":
        return SmoothLossL2(reduction="mean")(field)
    raise ValueError(f"Unknown smoothness penalty {penalty!r}; use 'l1' or 'l2'.")
