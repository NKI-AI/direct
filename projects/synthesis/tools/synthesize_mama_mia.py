#!/usr/bin/env python3
"""Synthesize FastMRI-style k-space from MAMA-MIA magnitude NIfTI volumes.

Preprocessing matches the breast diffusion YAML: pad/center-crop to 320x320,
99th-percentile scale. Phase is sampled with DDIM; coil maps are Biot-Savart
loops (16 coils, same layout as FastMRI breast).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from direct.data import transforms as T
from direct.nn.synthesis.synthesis import PhaseDiffusion
from direct.synthesis.data import export_h5
from direct.synthesis.physics import simulated_maps, synthesis_diagnostics

SIZE = 320
NUM_COILS = 16
MAP_MODE = "biot_savart"


def _load_nifti(path: Path) -> tuple[np.ndarray, dict]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)
    zooms = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])
    return data, {"shape": list(data.shape), "zooms": zooms, "affine": img.affine.tolist()}


def _slice_axis(shape: tuple[int, ...]) -> int:
    """Prefer a short axis when the other two are similar (axial stack)."""
    if len(shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {shape}")
    a, b, c = shape
    pairs = [
        (0, abs(b - c) / max(b, c, 1) + (0 if a <= min(b, c) else 1)),
        (1, abs(a - c) / max(a, c, 1) + (0 if b <= min(a, c) else 1)),
        (2, abs(a - b) / max(a, b, 1) + (0 if c <= min(a, b) else 1)),
    ]
    pairs.sort(key=lambda item: (item[1], shape[item[0]]))
    return pairs[0][0]


def _pad_crop(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    out = image
    pads = []
    for length in out.shape:
        extra = max(size - length, 0)
        pads.append((extra // 2, extra - extra // 2))
    if any(p != (0, 0) for p in pads):
        out = np.pad(out, pads, mode="constant")
    starts = [(length - size) // 2 for length in out.shape]
    return out[starts[0] : starts[0] + size, starts[1] : starts[1] + size]


def preprocess_volume(volume: np.ndarray) -> tuple[np.ndarray, dict]:
    axis = _slice_axis(volume.shape)
    slices = np.moveaxis(volume, axis, 0)
    prepared = np.stack([_pad_crop(sl) for sl in slices], axis=0)
    peak = float(np.quantile(prepared, 0.99))
    if peak <= 0:
        raise ValueError("Volume has no positive signal after cropping.")
    prepared = (prepared / peak).astype(np.float32)
    meta = {
        "slice_axis": int(axis),
        "n_slices": int(prepared.shape[0]),
        "crop": SIZE,
        "scale_percentile": 0.99,
        "scale_value": peak,
    }
    return prepared, meta


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


def load_model(checkpoint: Path, device: torch.device) -> PhaseDiffusion:
    model = PhaseDiffusion(num_filters=64, num_pool_layers=4, timesteps=1000, eval_timesteps=50)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def synthesize_volume(
    magnitude: np.ndarray,
    model: PhaseDiffusion,
    device: torch.device,
    *,
    steps: int,
    seed: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mag = torch.from_numpy(magnitude[:, None].astype(np.float32))
    dummy = torch.ones(1, 1, SIZE, SIZE, dtype=torch.float32, device=device)
    maps = simulated_maps(
        dummy,
        NUM_COILS,
        seed,
        mode=MAP_MODE,
        coil_radius=1.25,
        coil_size=0.40,
        biot_savart_segments=96,
        normalize=True,
    )
    gen = torch.Generator(device=device).manual_seed(seed)
    phases, kspaces = [], []
    for start in range(0, mag.shape[0], batch_size):
        batch = mag[start : start + batch_size].to(device)
        maps_b = maps.expand(batch.shape[0], -1, -1, -1, -1)
        out = model.sample(batch, maps_b, gen, num_steps=steps)
        phases.append(out.phase.cpu())
        kspaces.append(out.kspace.cpu())
    phase = torch.cat(phases, dim=0)
    kspace = torch.cat(kspaces, dim=0)
    maps_cpu = maps.expand(mag.shape[0], -1, -1, -1, -1).cpu()
    return mag, phase, maps_cpu, kspace


def plot_volume(
    name: str,
    mag: np.ndarray,
    phase: np.ndarray,
    maps: np.ndarray,
    kspace: np.ndarray,
    rss: np.ndarray,
    out_dir: Path,
    diagnostics: dict,
) -> None:
    n_slices = mag.shape[0]
    picks = sorted({0, n_slices // 4, n_slices // 2, (3 * n_slices) // 4, n_slices - 1})
    picks = [i for i in picks if 0 <= i < n_slices]
    k_abs = np.sqrt(kspace[..., 0] ** 2 + kspace[..., 1] ** 2)
    s_abs = np.sqrt(maps[..., 0] ** 2 + maps[..., 1] ** 2)
    s_phase = np.arctan2(maps[..., 1], maps[..., 0])

    ncols = 8
    fig, axes = plt.subplots(len(picks), ncols, figsize=(22, 2.6 * len(picks)), constrained_layout=True)
    if len(picks) == 1:
        axes = np.expand_dims(axes, 0)
    titles = [
        "Magnitude (DICOM/NIfTI)",
        "Predicted phase",
        "IFFT+RSS",
        "log |k| coil 0",
        "log |k| RSS coils",
        "|S| coil 0",
        "|S| coil 4",
        "∠S coil 0",
    ]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=9)
    for row, sl in enumerate(picks):
        vmax = np.percentile(mag[sl], 99.5)
        axes[row, 0].imshow(mag[sl], cmap="gray", vmin=0, vmax=vmax)
        axes[row, 1].imshow(_phase_rgb(phase[sl], mag[sl]))
        axes[row, 2].imshow(rss[sl], cmap="gray", vmin=0, vmax=vmax)
        axes[row, 3].imshow(np.log1p(k_abs[sl, 0]), cmap="magma")
        axes[row, 4].imshow(np.log1p(np.sqrt((k_abs[sl] ** 2).sum(0))), cmap="magma")
        axes[row, 5].imshow(s_abs[sl, 0], cmap="viridis", vmin=0, vmax=float(s_abs[sl].max()))
        axes[row, 6].imshow(s_abs[sl, 4], cmap="viridis", vmin=0, vmax=float(s_abs[sl].max()))
        axes[row, 7].imshow(s_phase[sl, 0], cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[row, 0].set_ylabel(f"{name}\nslice {sl}/{n_slices - 1}", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{name}  |  {MAP_MODE} {NUM_COILS} coils  |  RSS NMSE {diagnostics['rss_nmse']:.1e}",
        fontsize=11,
    )
    fig.savefig(out_dir / f"{name}_slices.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    mid = n_slices // 2
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    vmax = float(s_abs[mid].max())
    for coil, ax in enumerate(axes.ravel()):
        ax.imshow(s_abs[mid, coil], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"|S_{coil}|", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name} Biot-Savart |S_c|  mid-slice {mid}", fontsize=11)
    fig.savefig(out_dir / f"{name}_sensitivities.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for idx, coil in enumerate(range(0, 16, 2)[:8]):
        ax = axes.ravel()[idx]
        ax.imshow(np.log1p(k_abs[mid, coil]), cmap="magma")
        ax.set_title(f"log |k_{coil}|", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name} k-space magnitude  mid-slice {mid}", fontsize=11)
    fig.savefig(out_dir / f"{name}_kspace.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    volumes = [
        Path("/data/groups/public/archive_moved/MAMA-MIA/images/DUKE_001/DUKE_001_0000.nii.gz"),
        Path("/data/groups/public/archive_moved/MAMA-MIA/images/ISPY1_1088/ISPY1_1088_0000.nii.gz"),
        Path("/data/groups/public/archive_moved/MAMA-MIA/images/ISPY2_706943/ISPY2_706943_0000.nii.gz"),
    ]
    for path in volumes:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    plots = args.output / "plots"
    plots.mkdir(exist_ok=True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} ckpt={args.checkpoint} steps={args.steps}")
    model = load_model(args.checkpoint, device)

    summary = []
    for path in volumes:
        name = path.parent.name
        print(f"=== {name} {path} ===")
        raw, nifti_meta = _load_nifti(path)
        mag_np, prep_meta = preprocess_volume(raw)
        print(f"  raw={tuple(raw.shape)} slices={mag_np.shape} p99={prep_meta['scale_value']:.4g}")
        mag, phase, maps, kspace = synthesize_volume(
            mag_np, model, device, steps=args.steps, seed=args.seed, batch_size=args.batch_size
        )
        from direct.synthesis.physics import SynthesisOutput

        output = SynthesisOutput(
            kspace=kspace,
            sensitivity_map=maps,
            phase=phase,
            magnitude=mag,
        )
        diag = synthesis_diagnostics(output)
        print("  diagnostics", diag)
        coils = T.ifft2(kspace, dim=(2, 3))
        rss = T.root_sum_of_squares(coils, dim=1).numpy()
        h5_path = args.output / f"{name}.h5"
        if h5_path.exists():
            h5_path.unlink()
        export_h5(
            h5_path,
            output,
            {
                "source": str(path),
                "preprocessing": prep_meta,
                "nifti": {k: nifti_meta[k] for k in ("shape", "zooms")},
                "maps": MAP_MODE,
                "num_coils": NUM_COILS,
                "ddim_steps": args.steps,
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
            },
        )
        plot_volume(
            name,
            mag[:, 0].numpy(),
            phase.numpy(),
            maps.numpy(),
            kspace.numpy(),
            rss,
            plots,
            diag,
        )
        summary.append({"volume": name, "file": str(h5_path), "diagnostics": diag, "preprocessing": prep_meta})

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("done", args.output)


if __name__ == "__main__":
    main()
