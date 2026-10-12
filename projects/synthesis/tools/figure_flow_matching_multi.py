#!/usr/bin/env python3
"""Publication-style method figure for multi-anatomy phase flow matching."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path("/home/g.yiasemis/projects/direct/projects/synthesis/figures")
FIG_W, FIG_H = 14.2, 9.4
EXAMPLES = {
    "Brain": (
        Path("/mnt/rhea/data_public/fastmri/brain/multicoil_train") / "file_brain_AXFLAIR_200_6002425.h5",
        "#3D5A80",
    ),
    "Knee": (
        Path("/mnt/rhea/data_public/fastmri/knee/multicoil_train") / "file1000023.h5",
        "#6B5B95",
    ),
    "Prostate": (
        Path("/mnt/rhea/data_public/fastmri/prostate/training/T2") / "file_prostate_AXT2_0022.h5",
        "#2A9D8F",
    ),
    "Breast": (
        Path("/mnt/rhea/data_public/fastmri/breast/derived/cartesian") / "fastMRI_breast_038_1.h5",
        "#C1666B",
    ),
}


def _ifft2c(kspace: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def _as_complex(k: np.ndarray) -> np.ndarray:
    if np.isrealobj(k) and k.shape[-1] == 2:
        return k[..., 0] + 1j * k[..., 1]
    return k


def simulate_mode_maps(
    rss: np.ndarray,
    reference_maps: np.ndarray | None = None,
    num_coils: int = 8,
) -> dict[str, np.ndarray]:
    """Coil-0 magnitudes from DIRECT's existing simulators."""
    import torch

    from direct.data.coil_sensitivity import simulate_coil_sensitivity_maps

    h, w = rss.shape[-2:]
    out: dict[str, np.ndarray] = {}
    for mode in ("birdcage", "surface", "biot_savart"):
        maps = simulate_coil_sensitivity_maps(
            (h, w),
            num_coils,
            mode=mode,
            seed=0,
            normalize=True,
        )
        mag = torch.linalg.vector_norm(maps, dim=-1).numpy()
        out[mode] = mag
    if reference_maps is not None:
        ref = torch.from_numpy(np.stack((reference_maps.real, reference_maps.imag), axis=-1)).float()
        maps = simulate_coil_sensitivity_maps(
            (h, w),
            min(num_coils, ref.shape[0]),
            mode="empirical",
            reference_maps=ref,
            normalize=True,
        )
        out["empirical"] = torch.linalg.vector_norm(maps, dim=-1).numpy()
    return out


def load_example(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        ds = handle["kspace"]
        sl = int(ds.shape[0] // 2)
        k = _as_complex(np.asarray(ds[sl]))
    if k.ndim == 2:
        k = k[np.newaxis, ...]
    coils = _ifft2c(k)
    rss = np.sqrt((np.abs(coils) ** 2).sum(axis=0))
    maps = coils / np.maximum(rss, 1e-8)
    sense = (np.conj(maps) * coils).sum(axis=0)
    mag = np.maximum(rss, 0)
    mag_n = mag / max(float(np.percentile(mag, 99)), 1e-8)
    phase = np.asarray(np.angle(sense), dtype=np.float64)
    phase_show = np.where(mag_n > 0.08, phase, np.nan)
    kshow = np.log1p(np.abs(k).mean(axis=0))
    nshow = min(4, maps.shape[0])
    return {
        "rss": np.asarray(rss, dtype=np.float64),
        "phase": phase_show,
        "kshow": np.asarray(kshow, dtype=np.float64),
        "map": np.abs(maps[0]),
        "maps": np.abs(maps[:nshow]),
        "maps_complex": maps,
    }


def _box(ax, x, y, w, h, facecolor, edgecolor="#222222", lw=1.1, radius=0.012):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            zorder=2,
        )
    )


def _arrow(ax, x1, y1, x2, y2, color="#222222", lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            zorder=4,
        )
    )


def _imshow_box(ax, image, x0, y0, box_w, box_h, cmap, vmin=None, vmax=None, fig=None):
    """Draw ``image`` inside a data-coord box, preserving native pixel aspect."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr[..., 0]
    ih, iw = arr.shape[:2]
    finite = arr[np.isfinite(arr)]
    if vmax is None:
        vmax = np.percentile(finite, 99) if finite.size else 1.0
    if vmin is None:
        vmin = 0.0 if np.nanmin(arr) >= 0 else np.percentile(finite, 1)
    fig_w, fig_h = (fig or ax.figure).get_size_inches()
    box_w_in = box_w * fig_w
    box_h_in = box_h * fig_h
    img_aspect = iw / max(ih, 1)
    box_aspect = box_w_in / max(box_h_in, 1e-8)
    if img_aspect >= box_aspect:
        draw_w_in = box_w_in
        draw_h_in = draw_w_in / img_aspect
    else:
        draw_h_in = box_h_in
        draw_w_in = draw_h_in * img_aspect
    draw_w = draw_w_in / fig_w
    draw_h = draw_h_in / fig_h
    x = x0 + (box_w - draw_w) / 2
    y = y0 + (box_h - draw_h) / 2
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f4f6f8")
    ax.imshow(
        arr,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        extent=(x, x + draw_w, y, y + draw_h),
        origin="upper",
        interpolation="nearest",
        zorder=3,
        aspect="auto",
    )
    ax.add_patch(Rectangle((x, y), draw_w, draw_h, fill=False, ec="#222", lw=0.55, zorder=5))
    return x, y, draw_w, draw_h


def main() -> None:
    examples = {}
    for name, (path, color) in EXAMPLES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        examples[name] = (load_example(path), color)
    query = examples["Brain"][0]
    sim_maps = simulate_mode_maps(query["rss"], query["maps_complex"], num_coils=8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0,
        }
    )
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.988,
        "Conditional phase flow matching for multi-anatomy k-space synthesis",
        ha="center",
        va="top",
        fontsize=13.6,
        fontweight="bold",
        color="#1b1b1b",
    )
    ax.text(
        0.5,
        0.972,
        r"Learn $p(\varphi\mid m,\mathrm{organ})$, then $\tilde k_c=\mathrm{FFT}(S_c\,m\,e^{i\varphi})$  ·  $\sum_c|S_c|^2=1$",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#444444",
    )

    # ---- A ----
    _box(ax, 0.016, 0.528, 0.30, 0.412, "#f4f6f8")
    ax.text(0.030, 0.915, "A   Fully sampled FastMRI  (S_train)", fontsize=9.4, fontweight="bold")
    ax.text(0.030, 0.892, "Native FOV  ·  organ-conditioned volumes", fontsize=7.4, color="#555")

    for i, (name, (data, color)) in enumerate(examples.items()):
        y = 0.800 - i * 0.080
        _box(ax, 0.028, y - 0.004, 0.276, 0.074, color, edgecolor=color, lw=0.0, radius=0.008)
        ax.text(0.038, y + 0.052, name, color="white", fontsize=7.5, fontweight="bold", va="top")
        _imshow_box(ax, data["kshow"], 0.112, y + 0.004, 0.070, 0.058, "gray", fig=fig)
        _imshow_box(ax, data["rss"], 0.186, y + 0.004, 0.070, 0.058, "gray", fig=fig)
        ax.text(0.147, y + 0.002, "k", ha="center", fontsize=6.0, color="white")
        ax.text(0.221, y + 0.002, "RSS", ha="center", fontsize=6.0, color="white")

    ax.text(0.030, 0.540, "Coil counts batched separately  ·  no pad_coils", fontsize=7.1, color="#555")

    # ---- B ----
    _box(ax, 0.332, 0.528, 0.328, 0.412, "#eef6f4")
    ax.text(0.346, 0.915, "B   Train labels  (not ACS / ESPIRiT)", fontsize=9.4, fontweight="bold")
    ax.text(
        0.346, 0.892, r"$S_c=\mathrm{IFFT}(k_c)/\mathrm{RSS}$  (FULL_RSS on S_train $k$)", fontsize=7.3, color="#2A6F63"
    )

    _imshow_box(ax, query["rss"], 0.348, 0.718, 0.092, 0.145, "gray", fig=fig)
    ax.text(0.394, 0.704, r"magnitude $m$", ha="center", fontsize=7.1)
    _imshow_box(ax, query["map"], 0.448, 0.718, 0.092, 0.145, "magma", fig=fig)
    ax.text(0.494, 0.704, r"map $|S_c|$", ha="center", fontsize=7.1)
    _imshow_box(ax, query["phase"], 0.548, 0.718, 0.092, 0.145, "twilight", vmin=-np.pi, vmax=np.pi, fig=fig)
    ax.text(0.594, 0.704, r"SENSE $\varphi$", ha="center", fontsize=7.1)

    _box(ax, 0.346, 0.562, 0.300, 0.128, "white", edgecolor="#2A9D8F", lw=1.15)
    ax.text(0.496, 0.662, "Unit-phase target", ha="center", fontsize=8.0, fontweight="bold", color="#1b4d46")
    ax.text(0.496, 0.630, r"$x_1=(\cos\varphi,\,\sin\varphi)$", ha="center", fontsize=9.0)
    ax.text(0.496, 0.598, r"SENSE combination after FULL_RSS maps", ha="center", fontsize=6.8, color="#444")
    ax.text(
        0.496, 0.572, r"$\sum_c|S_c|^2=1$  $\Rightarrow$  RSS$(\mathrm{IFFT}(\tilde k))=m$", ha="center", fontsize=7.3
    )

    _arrow(ax, 0.316, 0.74, 0.332, 0.74)

    # ---- C ----
    _box(ax, 0.676, 0.528, 0.308, 0.412, "#f3eef8")
    ax.text(0.690, 0.915, "C   OT conditional flow matching", fontsize=9.4, fontweight="bold")
    ax.text(0.690, 0.892, r"128-ch / 4-pool U-Net  ·  AdaIN$(t,\,\mathrm{organ})$", fontsize=7.4, color="#553a7a")

    _box(ax, 0.692, 0.755, 0.276, 0.115, "white", edgecolor="#6B5B95")
    ax.text(
        0.830,
        0.848,
        r"$v_\theta(x_t, t \mid m, \mathrm{organ})$",
        ha="center",
        fontsize=9.8,
        fontweight="bold",
        color="#3d2a5c",
    )
    ax.text(0.830, 0.818, r"input: $x_t$ + $m$ + $t$ channel", ha="center", fontsize=7.3)
    ax.text(0.830, 0.790, r"$x_t=(1-t)\,x_0 + t\,x_1$", ha="center", fontsize=8.0)
    ax.text(0.830, 0.766, "organ: brain, knee, prostate, breast", ha="center", fontsize=7.0, color="#555")

    _box(ax, 0.692, 0.562, 0.276, 0.175, "white")
    ax.text(0.830, 0.710, "Training  (velocity only)", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(0.830, 0.678, r"$x_0\sim\mathcal{N}(0,I)$", ha="center", fontsize=8.0)
    ax.text(0.830, 0.648, r"$v^\star = x_1-x_0$", ha="center", fontsize=8.0)
    ax.text(
        0.830,
        0.610,
        r"$\mathcal{L}=\|v_\theta-v^\star\|^2$",
        ha="center",
        fontsize=10.0,
        fontweight="bold",
        color="#3d2a5c",
    )
    ax.text(0.830, 0.578, "Eval: Euler ODE, 30 steps, project to $S^1$", ha="center", fontsize=7.1, color="#555")

    _arrow(ax, 0.660, 0.74, 0.676, 0.74)

    # ---- D ----
    _box(ax, 0.016, 0.028, 0.968, 0.500, "#f7f4ee")
    ax.text(
        0.030, 0.505, "D   Inference: phase sample + sensitivity maps + recon transfer", fontsize=9.4, fontweight="bold"
    )

    # phase branch
    _box(ax, 0.028, 0.318, 0.455, 0.170, "white", edgecolor="#8a7040")
    ax.text(
        0.255,
        0.468,
        "Phase  (R_train magnitude; $k$ hidden)",
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="#5c4a22",
    )
    phase_steps = [
        (0.040, r"$m$ + organ"),
        (0.155, r"$x_0$ noise"),
        (0.270, "ODE 30 steps"),
        (0.385, r"$\hat\varphi$ on $S^1$"),
    ]
    for x, lab in phase_steps:
        _box(ax, x, 0.348, 0.100, 0.085, "#efe6d4", edgecolor="#8a7040")
        ax.text(x + 0.050, 0.390, lab, ha="center", va="center", fontsize=7.2)
    for x in (0.140, 0.255, 0.370):
        _arrow(ax, x, 0.390, x + 0.015, 0.390)

    # map generation
    _box(ax, 0.500, 0.318, 0.468, 0.170, "white", edgecolor="#2A9D8F")
    ax.text(
        0.734,
        0.468,
        "Sensitivity maps at apply time  (no R_train $k$)",
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="#1b4d46",
    )
    ax.text(
        0.734,
        0.448,
        "DIRECT coil simulator: birdcage  ·  surface  ·  Biot–Savart  ·  empirical",
        ha="center",
        fontsize=7.0,
        color="#1b4d46",
    )
    sim_x = 0.518
    for mode, label in (
        ("birdcage", "birdcage"),
        ("surface", "surface"),
        ("biot_savart", "Biot–Savart"),
    ):
        _imshow_box(ax, sim_maps[mode][0], sim_x, 0.332, 0.078, 0.100, "magma", fig=fig)
        ax.text(sim_x + 0.039, 0.324, label, ha="center", fontsize=6.4)
        sim_x += 0.092
    _imshow_box(ax, sim_maps["empirical"][0], sim_x, 0.332, 0.078, 0.100, "magma", fig=fig)
    ax.text(sim_x + 0.039, 0.324, "empirical", ha="center", fontsize=6.4)

    # forward model merge
    _box(ax, 0.028, 0.175, 0.940, 0.128, "white", edgecolor="#8a7040")
    ax.text(0.498, 0.278, "Forward model", ha="center", fontsize=8.0, fontweight="bold", color="#5c4a22")
    _box(ax, 0.045, 0.192, 0.145, 0.062, "#efe6d4", edgecolor="#8a7040")
    ax.text(0.117, 0.223, r"$m$, $\hat\varphi$", ha="center", va="center", fontsize=8.0)
    _box(ax, 0.220, 0.192, 0.175, 0.062, "#d9efe9", edgecolor="#2A9D8F")
    ax.text(0.307, 0.223, r"$S_c^{\mathrm{sim}}$", ha="center", va="center", fontsize=8.0)
    _arrow(ax, 0.190, 0.223, 0.220, 0.223)
    _arrow(ax, 0.395, 0.223, 0.430, 0.223)
    _box(ax, 0.430, 0.192, 0.380, 0.062, "#efe6d4", edgecolor="#6B5B95")
    ax.text(
        0.620,
        0.223,
        r"$\tilde k_c=\mathrm{FFT}\!\left(S_c^{\mathrm{sim}}\, m\, e^{i\hat\varphi}\right)$",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#3d2a5c",
    )
    _arrow(ax, 0.810, 0.223, 0.830, 0.223)
    _box(ax, 0.830, 0.192, 0.120, 0.062, "#efe6d4", edgecolor="#8a7040")
    ax.text(0.890, 0.223, r"synth $\tilde k$", ha="center", va="center", fontsize=8.0)

    # recon + splits
    ax.text(
        0.030,
        0.155,
        "Matched recon pair  (same R_train patients, same masks; ACS maps at recon time)",
        fontsize=7.8,
        fontweight="bold",
        color="#2c415c",
    )
    _box(ax, 0.028, 0.078, 0.200, 0.068, "#d9e4f0", edgecolor="#3D5A80")
    ax.text(0.128, 0.112, "Real $k$  ·  R_train", ha="center", va="center", fontsize=7.6)
    _box(ax, 0.245, 0.078, 0.200, 0.068, "#efe6d4", edgecolor="#8a7040")
    ax.text(0.345, 0.112, r"Synth $\tilde k$  ·  R_train", ha="center", va="center", fontsize=7.6)
    ax.text(
        0.455,
        0.112,
        r"$\rightarrow$  test both on real $k$ of R_val / frozen R_test",
        va="center",
        fontsize=7.5,
        color="#333",
    )

    split_items = [
        ("S_train 55%", "#2A9D8F", "synthesizer train"),
        ("S_val 5%", "#3D5A80", "synthesizer val"),
        ("R_train 40%", "#C1666B", "recon train"),
        ("R_val / R_test", "#6B5B95", "official val, real $k$"),
    ]
    x = 0.028
    for lab, color, sub in split_items:
        _box(ax, x, 0.038, 0.175, 0.032, color, edgecolor=color, lw=0)
        ax.text(x + 0.087, 0.054, f"{lab}  ·  {sub}", ha="center", color="white", fontsize=6.4)
        x += 0.185
    ax.text(
        0.030,
        0.016,
        "Breast (no official val): 45 / 5 / 40 / 5 / 5.  Apply-time maps: simulated (birdcage / surface / Biot–Savart) or empirical.  Brain stratified by contrast.",
        fontsize=6.6,
        color="#555",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "flow_matching_multi_method.png"
    pdf = OUT_DIR / "flow_matching_multi_method.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
