#!/usr/bin/env python3
"""Magnitude NIfTI/NRRD → N-coil maps (sens) + phase (FM) → k-space (+ optional R=4 mask).

Preprocessing matches multi-anatomy training: 320×320, p99 scale. For brain
NIfTI, reorient to RAS axial (FastMRI-like: anterior up), bbox-crop background,
then resample to 320×320.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom

DIRECT_ROOT = Path("/home/g.yiasemis/projects/direct")
BUILD_DIR = DIRECT_ROOT / "build" / "cp312"
SIZE = 320
ORGAN_NAMES = ("brain", "knee", "prostate", "breast", "cardiac", "unknown")


def _disable_editable_rebuild() -> None:
    pairs = (
        (BUILD_DIR / "_poisson.abi3.so", DIRECT_ROOT / "direct" / "common" / "_poisson.abi3.so"),
        (BUILD_DIR / "_gaussian.abi3.so", DIRECT_ROOT / "direct" / "common" / "_gaussian.abi3.so"),
        (BUILD_DIR / "_gaussian_fill.abi3.so", DIRECT_ROOT / "direct" / "ssl" / "_gaussian_fill.abi3.so"),
    )
    for src, dst in pairs:
        if src.is_file() and (not dst.exists() or dst.stat().st_mtime_ns < src.stat().st_mtime_ns):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    sys.meta_path = [f for f in sys.meta_path if type(f).__name__ != "MesonpyMetaFinder"]
    sys.path_importer_cache.clear()
    root = str(DIRECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_disable_editable_rebuild()

from direct.common.subsample import FastMRIRandomMaskFunc
from direct.data import transforms as T
from direct.nn.synthesis.synthesis import PhaseFlowMatching, SensitivityMapSynthesizer
from direct.synthesis.physics import synthesize


def _bbox_crop(image: np.ndarray, pad_frac: float = 0.05) -> np.ndarray:
    """Tight crop around the largest bright connected component (head / organ)."""
    from scipy.ndimage import binary_opening, label

    peak = float(image.max())
    if peak <= 0:
        return image
    # Ignore dim background / table; keep tissue.
    thr = max(0.12 * peak, float(np.percentile(image, 70)))
    mask = image > thr
    mask = binary_opening(mask, iterations=2)
    if not mask.any():
        return image
    labeled, n_feat = label(mask)
    if n_feat < 1:
        return image
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    mask = labeled == int(counts.argmax())
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(2, round((y1 - y0) * pad_frac))
    pad_x = max(2, round((x1 - x0) * pad_frac))
    y0, x0 = max(0, y0 - pad_y), max(0, x0 - pad_x)
    y1, x1 = min(image.shape[0], y1 + pad_y), min(image.shape[1], x1 + pad_x)
    return image[y0:y1, x0:x1]


def _pad_to_square(image: np.ndarray) -> np.ndarray:
    """Zero-pad the short side so the frame is square (keeps aspect ratio)."""
    h, w = image.shape
    side = max(h, w)
    pad_y = side - h
    pad_x = side - w
    return np.pad(
        image,
        ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
        mode="constant",
    )


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    """Center-crop to the largest square inside the image (keeps aspect ratio)."""
    h, w = image.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side].astype(np.float32)


def _resample_isotropic(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Isotropic zoom so the long side is ``size``, then pad/crop to ``size×size``."""
    h, w = image.shape
    scale = size / max(h, w, 1)
    if abs(scale - 1.0) > 1e-6:
        image = zoom(image, (scale, scale), order=1).astype(np.float32)
    return _pad_crop(image, size)


def _pad_crop(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    out = image
    pads = []
    for length in out.shape:
        extra = max(size - length, 0)
        pads.append((extra // 2, extra - extra // 2))
    if any(p != (0, 0) for p in pads):
        out = np.pad(out, pads, mode="constant")
    starts = [(length - size) // 2 for length in out.shape]
    return out[starts[0] : starts[0] + size, starts[1] : starts[1] + size].astype(np.float32)


def _load_volume(path: Path, modality: int | None, *, axial_ras: bool) -> tuple[np.ndarray, dict]:
    """Load magnitude volume as (H, W, D) float32. Optionally RAS-axial for brain NIfTI.

    ``info['zooms']`` is ``(dz_h, dz_w, dz_d)`` matching the returned array axes
    (for RAS axial: H=R, W=A, D=S).
    """
    info: dict = {"zooms": (1.0, 1.0, 1.0)}
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nrrd"):
        import nrrd

        vol, hdr = nrrd.read(str(path), index_order="C")
        vol = np.asarray(vol, dtype=np.float32)
        if vol.ndim == 3 and vol.shape[0] < min(vol.shape[1:]):
            vol = np.moveaxis(vol, 0, -1)
        info["orientation"] = "nrrd_as_is"
        # Best-effort spacing from NRRD space directions if present.
        try:
            spacing = hdr.get("space directions")
            if spacing is not None:
                zooms = tuple(float(np.linalg.norm(spacing[i])) for i in range(3))
                info["zooms"] = zooms
        except (TypeError, ValueError, IndexError):
            info["zooms_note"] = "space directions unreadable; assuming 1 mm"
    else:
        import nibabel as nib

        img = nib.load(str(path))
        if axial_ras:
            img = nib.as_closest_canonical(img)
            info["orientation"] = "".join(nib.orientations.aff2axcodes(img.affine))
            info["reorient"] = "as_closest_canonical_RAS"
        else:
            info["orientation"] = "".join(nib.orientations.aff2axcodes(img.affine))
        vol = np.asanyarray(img.dataobj, dtype=np.float32)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        info["zooms"] = zooms  # (R, A, S) after canonical
        if vol.ndim == 4:
            if modality is None:
                raise ValueError("4D volume requires --modality")
            vol = vol[..., modality]
        elif vol.ndim != 3:
            raise ValueError(f"Expected 3D/4D volume, got {vol.shape}")
        if axial_ras and len(info.get("orientation", "")) == 3:
            info["slice_plane"] = "axial_RAS_RA"
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(vol, 0.0, None), info


def _to_fastmri_like_view(axial_ra: np.ndarray) -> np.ndarray:
    """Map (R, A) axial plane to FastMRI-like display: anterior toward top."""
    return np.flipud(axial_ra.T).astype(np.float32)


def _resample_volume_isotropic(vol: np.ndarray, zooms: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    """Resample a 3D volume to isotropic voxels (finest spacing)."""
    spa = tuple(max(float(z), 1e-6) for z in zooms)
    target = min(spa)
    factors = tuple(s / target for s in spa)
    if all(abs(f - 1.0) < 1e-6 for f in factors):
        return vol.astype(np.float32), target
    return zoom(vol, factors, order=1).astype(np.float32), target


def load_magnitude_slice(
    path: Path,
    *,
    modality: int | None,
    slice_idx: int | None,
    organ_name: str,
    num_coils: int,
    axial_ras: bool,
    bbox_crop: bool,
) -> tuple[np.ndarray, dict]:
    stack, orient_info = _load_volume(path, modality, axial_ras=axial_ras)
    if stack.ndim != 3:
        raise ValueError(f"Expected (H,W,D) after load, got {stack.shape}")

    zooms = orient_info.get("zooms", (1.0, 1.0, 1.0))
    if axial_ras:
        # Isotropic 3D resample so in-plane mm/pixel match (OASIS often has
        # anisotropic LR spacing). Do NOT stretch when going to 320×320 —
        # pad to square after bbox instead.
        stack, iso_spa = _resample_volume_isotropic(stack, zooms)
        orient_info["isotropic_spacing_mm"] = iso_spa
        orient_info["shape_after_iso"] = list(stack.shape)
        zooms = (iso_spa, iso_spa, iso_spa)

    n_slices = stack.shape[-1]
    if slice_idx is None:
        energies = stack.reshape(-1, n_slices).sum(axis=0)
        lo, hi = n_slices // 4, max(n_slices // 4 + 1, (3 * n_slices) // 4)
        slice_idx = int(np.argmax(energies[lo:hi]) + lo)
    if not 0 <= slice_idx < n_slices:
        raise ValueError(f"slice_idx {slice_idx} out of range for D={n_slices}")

    raw = stack[:, :, slice_idx]
    if axial_ras:
        # raw is (R, A) on an isotropic square grid; anterior up.
        view = _to_fastmri_like_view(raw)
    else:
        view = raw.astype(np.float32)

    if bbox_crop:
        view = _bbox_crop(view)
        # Keep aspect: pad to square (no stretch), then isotropic resize to 320.
        view = _pad_to_square(view)
        prepared = _resample_isotropic(view, SIZE)
    else:
        prepared = _pad_crop(_pad_to_square(view), SIZE)

    peak = float(np.quantile(prepared, 0.99))
    if peak <= 0:
        raise ValueError("Slice has no positive signal after crop.")
    prepared = (prepared / peak).astype(np.float32)
    organ_id = ORGAN_NAMES.index(organ_name)
    meta = {
        "input": str(path),
        "modality_index": modality,
        "slice_idx": slice_idx,
        "n_slices": n_slices,
        "raw_shape": list(raw.shape),
        "view_shape_before_320": list(view.shape),
        "crop": SIZE,
        "bbox_crop": bbox_crop,
        "axial_ras": axial_ras,
        "scale_percentile": 0.99,
        "scale_value": peak,
        "organ_id": organ_id,
        "organ_name": organ_name,
        "num_coils": num_coils,
        **orient_info,
    }
    return prepared, meta


def _load_state(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_sens(checkpoint: Path, device: torch.device) -> SensitivityMapSynthesizer:
    model = SensitivityMapSynthesizer(
        num_filters=64,
        num_pool_layers=3,
        map_size=64,
        num_organs=6,
        organ_embed_dim=16,
        upsample_mode="nearest",
        organ_dropout_prob=0.0,
    )
    model.load_state_dict(_load_state(checkpoint), strict=True)
    return model.to(device).eval()


def load_phase(checkpoint: Path, device: torch.device) -> PhaseFlowMatching:
    model = PhaseFlowMatching(
        num_filters=128,
        num_pool_layers=4,
        num_integration_steps=50,
        eval_integration_steps=30,
        num_organs=6,
        organ_embed_dim=16,
        organ_dropout_prob=0.0,
    )
    model.load_state_dict(_load_state(checkpoint), strict=True)
    return model.to(device).eval()


def _phase_rgb(phase: np.ndarray, mag: np.ndarray) -> np.ndarray:
    hue = (np.arctan2(phase[..., 1], phase[..., 0]) + np.pi) / (2 * np.pi)
    value = mag / np.clip(mag.max(), 1e-8, None)
    sat = 0.9
    chroma = value * sat
    x = chroma * (1.0 - np.abs((hue * 6.0) % 2.0 - 1.0))
    offset = value - chroma
    h6 = hue * 6.0
    rgb = np.zeros((*hue.shape, 3), dtype=np.float32)
    zeros = np.zeros_like(hue)
    sectors = [
        (h6 < 1, (chroma, x, zeros)),
        ((h6 >= 1) & (h6 < 2), (x, chroma, zeros)),
        ((h6 >= 2) & (h6 < 3), (zeros, chroma, x)),
        ((h6 >= 3) & (h6 < 4), (zeros, x, chroma)),
        ((h6 >= 4) & (h6 < 5), (x, zeros, chroma)),
        (h6 >= 5, (chroma, zeros, x)),
    ]
    for mask, channels in sectors:
        for idx, channel in enumerate(channels):
            rgb[..., idx] = np.where(mask, channel, rgb[..., idx])
    return np.clip(rgb + offset[..., None], 0, 1)


def _kspace_display(abs_k: np.ndarray) -> np.ndarray:
    """High-contrast log |k| for plotting (percentile-stretched)."""
    y = np.log1p(abs_k.astype(np.float64))
    lo, hi = np.percentile(y, [2.0, 99.8])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def make_mask(height: int, width: int, acceleration: float, center_fraction: float, seed: int) -> torch.Tensor:
    mask_func = FastMRIRandomMaskFunc(accelerations=[acceleration], center_fractions=[center_fraction])
    # Shape: (coil=1, rows, cols, 1) — samples along width (cols = shape[-2] in mask_func? )
    # FastMRIRandom: num_cols = shape[-2], num_rows = shape[-3]
    # So shape (*, rows, cols, channels) → pass (1, H, W, 1)
    return mask_func((1, height, width, 1), seed=seed)


def plot_all(
    mag: np.ndarray,
    phase: np.ndarray,
    maps: np.ndarray,
    kspace: np.ndarray,
    rss: np.ndarray,
    kspace_u: np.ndarray,
    zf: np.ndarray,
    mask_1d: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    k_abs = np.linalg.norm(kspace, axis=-1)
    ku_abs = np.linalg.norm(kspace_u, axis=-1)
    s_abs = np.linalg.norm(maps, axis=-1)
    coils = k_abs.shape[0]
    ncols = max(coils, 6)
    # Square cells: width == 3 row-heights so aspect='equal' cannot stretch.
    cell = 2.0
    fig, axes = plt.subplots(3, ncols, figsize=(cell * ncols, cell * 3), squeeze=False)
    vmax = float(np.percentile(mag, 99.5))
    zf_vmax = float(np.percentile(zf, 99.5))

    panels = [
        (0, 0, mag, "Magnitude", "gray", 0, vmax, False),
        (0, 1, _phase_rgb(phase, mag), "Generated phase", None, None, None, True),
        (0, 2, rss, "RSS (full k)", "gray", 0, vmax, False),
        (0, 3, _kspace_display(np.sqrt((k_abs**2).sum(0))), "log|k| full", "magma", 0, 1, False),
        (0, 4, _kspace_display(np.sqrt((ku_abs**2).sum(0))), "log|k| R=4 (8% ACS)", "magma", 0, 1, False),
        (0, 5, zf, "Zero-filled RSS", "gray", 0, zf_vmax, False),
    ]

    def _show(ax, data, *, title_i: str, cmap=None, vmin=None, vmax=None, rgb: bool = False) -> None:
        if rgb:
            ax.imshow(data, aspect="equal", interpolation="nearest")
        else:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_title(title_i, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    for r, c, data, title_i, cmap, vmin, vmax_i, rgb in panels:
        _show(axes[r, c], data, title_i=title_i, cmap=cmap, vmin=vmin, vmax=vmax_i, rgb=rgb)

    smax = float(s_abs.max()) if s_abs.size else 1.0
    for c in range(coils):
        _show(axes[1, c], s_abs[c], title_i=f"|S| {c}", cmap="viridis", vmin=0, vmax=smax)
        _show(axes[2, c], _kspace_display(k_abs[c]), title_i=f"log|k| {c}", cmap="magma", vmin=0, vmax=1)

    # One-row mask visualization if space
    if ncols > coils:
        axes[2, coils].imshow(mask_1d[None, :], cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[2, coils].set_title("mask (cols)", fontsize=8)
        axes[2, coils].set_yticks([])

    for r in range(3):
        for c in range(ncols):
            if not axes[r, c].has_data() and not axes[r, c].get_title():
                axes[r, c].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    organ_name = args.organ
    if organ_name not in ORGAN_NAMES:
        raise ValueError(f"organ must be one of {ORGAN_NAMES}")
    organ_id = ORGAN_NAMES.index(organ_name)
    num_coils = int(args.num_coils)
    axial_ras = bool(args.axial_ras) if args.axial_ras is not None else (organ_name == "brain")

    input_path = Path(args.input)
    mag_np, meta = load_magnitude_slice(
        input_path,
        modality=args.modality,
        slice_idx=args.slice,
        organ_name=organ_name,
        num_coils=num_coils,
        axial_ras=axial_ras,
        bbox_crop=not args.no_bbox_crop,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sens = load_sens(Path(args.sens_ckpt), device)
    phase_net = load_phase(Path(args.phase_ckpt), device)

    mag = torch.from_numpy(mag_np)[None, None].to(device)
    organ = torch.tensor([organ_id], device=device, dtype=torch.long)

    maps = sens(mag, num_coils, organ_id=organ)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out = phase_net.sample(mag, maps, gen, num_steps=args.steps, organ_id=organ)
    out = synthesize(mag, out.phase, maps)

    coils_img = T.ifft2(out.kspace, dim=(2, 3))
    rss = T.root_sum_of_squares(coils_img, dim=1)[0].cpu().numpy()

    # Retrospective FastMRI random vertical mask: R=4, 8% center
    _, _, height, width, _ = out.kspace.shape
    mask = make_mask(height, width, args.acceleration, args.center_fraction, args.seed).to(device)
    # mask: (1, H, W, 1) → broadcast over batch/coils/complex
    while mask.ndim < out.kspace.ndim:
        mask = mask.unsqueeze(0)
    kspace_u = out.kspace * mask
    zf_coils = T.ifft2(kspace_u, dim=(2, 3))
    zf = T.root_sum_of_squares(zf_coils, dim=1)[0].cpu().numpy()
    mask_1d = mask.reshape(-1, width)[0].detach().cpu().numpy().astype(np.float32)

    mag_np = out.magnitude[0, 0].cpu().numpy()
    phase_np = out.phase[0].cpu().numpy()
    maps_np = out.sensitivity_map[0].cpu().numpy()
    kspace_np = out.kspace[0].cpu().numpy()
    kspace_u_np = kspace_u[0].cpu().numpy()

    np.savez_compressed(
        out_dir / "synthesis.npz",
        magnitude=mag_np,
        phase=phase_np,
        sensitivity_map=maps_np,
        kspace=kspace_np,
        kspace_undersampled=kspace_u_np,
        rss=rss,
        zero_filled=zf,
        sampling_mask=mask_1d,
    )
    k_cplx = kspace_np[..., 0] + 1j * kspace_np[..., 1]
    np.save(out_dir / "kspace_complex.npy", k_cplx.astype(np.complex64))
    ku_cplx = kspace_u_np[..., 0] + 1j * kspace_u_np[..., 1]
    np.save(out_dir / "kspace_undersampled_complex.npy", ku_cplx.astype(np.complex64))

    meta.update(
        {
            "sens_ckpt": str(args.sens_ckpt),
            "phase_ckpt": str(args.phase_ckpt),
            "ode_steps": args.steps,
            "seed": args.seed,
            "device": str(device),
            "acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "mask_name": "FastMRIRandom",
            "rss_l2_vs_mag": float(np.linalg.norm(rss - mag_np) / (np.linalg.norm(mag_np) + 1e-8)),
            "sampled_fraction": float(mask_1d.mean()),
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Quick magnitude QA (pixel-true aspect; no subplot stretch).
    fig_m, ax_m = plt.subplots(figsize=(4.0, 4.0))
    ax_m.imshow(mag_np, cmap="gray", aspect="equal", interpolation="nearest")
    ax_m.set_aspect("equal", adjustable="box")
    ys, xs = np.where(mag_np > 0.02 * float(mag_np.max()))
    ax_m.set_title(
        f"mag {tuple(mag_np.shape)} support=({int(ys.max() - ys.min() + 1)}, {int(xs.max() - xs.min() + 1)})"
    )
    ax_m.axis("off")
    fig_m.tight_layout()
    fig_m.savefig(out_dir / "mag_check.png", dpi=140, facecolor="white")
    plt.close(fig_m)

    plot_all(
        mag_np,
        phase_np,
        maps_np,
        kspace_np,
        rss,
        kspace_u_np,
        zf,
        mask_1d,
        out_dir / "overview.png",
        title=(
            f"{input_path.name}  slice={meta['slice_idx']}  N={num_coils}  organ={organ_name}  "
            f"R={args.acceleration} ACS={args.center_fraction:.0%}  "
            f"sens@{Path(args.sens_ckpt).stem}  phase@{Path(args.phase_ckpt).stem}"
        ),
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                **{
                    k: meta[k]
                    for k in (
                        "slice_idx",
                        "organ_name",
                        "num_coils",
                        "acceleration",
                        "center_fraction",
                        "sampled_fraction",
                        "rss_l2_vs_mag",
                        "orientation",
                    )
                    if k in meta
                },
            },
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "--nifti", dest="input", required=False, default="")
    p.add_argument("--modality", type=int, default=None)
    p.add_argument("--slice", type=int, default=None)
    p.add_argument("--organ", default="brain", choices=list(ORGAN_NAMES))
    p.add_argument("--num-coils", type=int, default=8)
    p.add_argument("--axial-ras", dest="axial_ras", action="store_true", default=None)
    p.add_argument("--no-axial-ras", dest="axial_ras", action="store_false")
    p.add_argument("--no-bbox-crop", action="store_true")
    p.add_argument("--acceleration", type=float, default=4.0)
    p.add_argument("--center-fraction", type=float, default=0.08)
    p.add_argument(
        "--sens-ckpt",
        default="/projects/direct/synthesis_diffusion/sensitivity_maps_multi_cardiac_500k/model_25000.pt",
    )
    p.add_argument(
        "--phase-ckpt",
        default="/projects/direct/synthesis_diffusion/flow_matching_multi_cardiac/model_10000.pt",
    )
    p.add_argument("--out-dir", default="/projects/direct/synthesis_diffusion/eval_fomo_oasis/P029_sub001_t1_8coil")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if not args.input:
        args.input = (
            "/projects/direct/synthesis_diffusion/eval_fomo_oasis/P029_sub001/raw/"
            "ses-01/anat/sub-001_ses-01_run-1_T1w.nii.gz"
        )
    run(args)


if __name__ == "__main__":
    main()
