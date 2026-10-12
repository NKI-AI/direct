"""Distribution diagnostics and held-out reconstruction metrics for synthesis."""

import torch

from direct.data import transforms as T
from direct.functionals.challenges import fastmri_nmse, fastmri_psnr, fastmri_ssim
from direct.synthesis.physics import validate_magnitude


@torch.no_grad()
def kspace_features(kspace: torch.Tensor, radial_bins: int = 8) -> dict[str, torch.Tensor]:
    """Extract scale- and coil-permutation-invariant distribution diagnostics.

    Args:
        kspace: Centered float32 multicoil data (B, C, H, W, 2).
        radial_bins: Number of equal-width radial frequency bands.

    Returns:
        Per-example radial power fractions (B, bins), normalized coil covariance
        eigenvalues (B, C), and effective coil rank (B,). Compare distributions
        within matching anatomy, contrast, FOV, resolution and noise strata.

    Raises:
        ValueError: If data is invalid or contains an empty signal.

    Notes:
        These features can reveal mismatch; agreement does not establish that
        synthetic data is suitable for training or clinically faithful.
    """
    if kspace.ndim != 5 or kspace.shape[-1] != 2 or kspace.dtype != torch.float32:
        raise ValueError("kspace must be float32 (B, C, H, W, 2).")
    if not torch.isfinite(kspace).all() or radial_bins < 1:
        raise ValueError("Finite data and positive radial_bins are required.")
    power = kspace.square().sum(-1).sum(1)
    total = power.sum((1, 2))
    if (total <= 0).any():
        raise ValueError("Empty signals have no defined normalized spectrum.")
    height, width = kspace.shape[2:4]
    yy, xx = torch.meshgrid(
        torch.fft.fftshift(torch.fft.fftfreq(height, device=kspace.device)),
        torch.fft.fftshift(torch.fft.fftfreq(width, device=kspace.device)),
        indexing="ij",
    )
    radius = (xx.square() + yy.square()).sqrt() / (0.5**0.5)
    assignment = (radius * radial_bins).long().clamp_max(radial_bins - 1)
    bands = torch.stack([(power * (assignment == index)).sum((1, 2)) / total for index in range(radial_bins)], 1)
    coils = torch.view_as_complex(T.ifft2(kspace, dim=(2, 3)).contiguous()).flatten(2)
    covariance = coils @ coils.mH
    eigenvalues = torch.linalg.eigvalsh(covariance).real.clamp_min(0).flip(1)
    eigenvalues = eigenvalues / eigenvalues.sum(1, keepdim=True).clamp_min(1e-12)
    rank = torch.exp(-(eigenvalues * eigenvalues.clamp_min(1e-12).log()).sum(1))
    return {"radial_power": bands, "coil_eigenvalues": eigenvalues, "effective_coils": rank}


@torch.no_grad()
def reconstruction_metrics(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Average 2D fastMRI metrics over slices (one score per patient/volume).

    Args:
        prediction: Reconstructed magnitude (slices, 1, H, W).
        reference: Ground-truth magnitude, identically cropped and scaled.

    Returns:
        Mean over slices of NMSE, PSNR and SSIM. Identical images have infinite PSNR.

    Raises:
        ValueError: If inputs mismatch or the reference has no signal.

    Notes:
        Each slice is scored in 2D, then averaged. A DICOM-only synthetic
        roundtrip measures consistency, not accuracy against real acquisitions.
    """
    validate_magnitude(prediction)
    validate_magnitude(reference)
    if prediction.shape != reference.shape or min(reference.shape[-2:]) < 7 or reference.max() <= 0:
        raise ValueError("Matching nonempty reference images with spatial size >= 7 are required.")
    prediction, reference = prediction.cpu(), reference.cpu()
    num_slices = prediction.shape[0]
    totals = {"nmse": 0.0, "psnr": 0.0, "ssim": 0.0}
    for idx in range(num_slices):
        pred_s = prediction[idx : idx + 1]
        ref_s = reference[idx : idx + 1]
        totals["nmse"] += float(fastmri_nmse(ref_s, pred_s))
        totals["psnr"] += float(fastmri_psnr(ref_s, pred_s))
        totals["ssim"] += float(fastmri_ssim(ref_s, pred_s))
    return {key: value / num_slices for key, value in totals.items()}
