#!/usr/bin/env python3
"""Generate seeded sampling-mask figures for the docs tutorial.

Run from the repository root:

    uv run --with matplotlib python docs/scripts/generate_sampling_mask_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from direct.common.subsample import build_masking_function

OUT_DIR = Path(__file__).resolve().parents[1] / "_static" / "tutorials"
SHAPE = (256, 256, 2)
SEED = 0
ACCELERATION = 4
CENTER_FRACTION = 0.08


def squeeze_mask(mask) -> np.ndarray:
    """Drop coil / complex singleton axes for display."""
    return np.asarray(mask.squeeze().cpu(), dtype=float)


def imshow_mask(ax, mask: np.ndarray, title: str) -> None:
    ax.imshow(mask, cmap="gray", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def build(name: str, **kwargs):
    defaults = {
        "accelerations": [ACCELERATION],
        "center_fractions": [CENTER_FRACTION],
        "range_mode": "discrete",
    }
    defaults.update(kwargs)
    return build_masking_function(name=name, **defaults)


def generate_catalog() -> None:
    schemes = (
        ("Random", {}),
        ("Equispaced", {}),
        ("Magic", {}),
        ("Gaussian1D", {}),
        ("Gaussian2D", {}),
        ("VariableDensityPoisson", {}),
        ("Radial", {"center_fractions": None}),
        ("Spiral", {"center_fractions": None}),
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.4), constrained_layout=True)
    for ax, (name, extra) in zip(axes.ravel(), schemes, strict=True):
        mask_func = build(name, **extra)
        mask = squeeze_mask(mask_func(SHAPE, seed=SEED))
        sampled = 100.0 * float(mask.mean())
        imshow_mask(ax, mask, f"{name}\n{sampled:.1f}% sampled")
    fig.suptitle(f"Built-in sampling schemes at {ACCELERATION}×  ({SHAPE[0]}×{SHAPE[1]})", fontsize=13)
    fig.savefig(OUT_DIR / "sampling_masks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def generate_dynamic_and_acs() -> None:
    n_frames = 4
    dynamic = build("Random", mode="dynamic")
    frames = squeeze_mask(dynamic((n_frames, *SHAPE), seed=SEED))

    static = build("Random")
    sampling = squeeze_mask(static(SHAPE, seed=SEED))
    acs = squeeze_mask(static(SHAPE, seed=SEED, return_acs=True))

    fig, axes = plt.subplots(1, 6, figsize=(14.4, 2.8), constrained_layout=True)
    for idx in range(n_frames):
        imshow_mask(axes[idx], frames[idx], f"dynamic t={idx}")
    imshow_mask(axes[4], sampling, "sampling mask")
    imshow_mask(axes[5], acs, "ACS (return_acs)")
    fig.suptitle("Dynamic random masks (left) and ACS region (right), 4×", fontsize=13)
    fig.savefig(OUT_DIR / "sampling_masks_dynamic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_catalog()
    generate_dynamic_and_acs()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
