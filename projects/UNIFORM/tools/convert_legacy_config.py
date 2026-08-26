#!/usr/bin/env python3
"""Convert legacy UNIFORM / multiorgan flat transform configs to current DIRECT nested format."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _modernize_transforms(transforms: dict[str, Any]) -> dict[str, Any]:
    """Map flat transform keys to nested TransformsConfig groups."""
    if not transforms:
        return transforms

    # Already modern if nested groups are present.
    if any(k in transforms for k in ("cropping", "sensitivity_map_estimation", "normalization")):
        return transforms

    out: dict[str, Any] = {}
    if "masking" in transforms:
        out["masking"] = transforms["masking"]

    cropping: dict[str, Any] = {}
    for key in ("crop", "crop_type", "image_center_crop"):
        if key in transforms:
            cropping[key] = transforms[key]
    if cropping:
        out["cropping"] = cropping

    random_aug: dict[str, Any] = {}
    for key in (
        "random_rotation_degrees",
        "random_rotation_probability",
        "random_flip_type",
        "random_flip_probability",
        "random_reverse_probability",
    ):
        if key in transforms:
            random_aug[key] = transforms[key]
    if random_aug:
        out["random_augmentations"] = random_aug

    sens: dict[str, Any] = {}
    for key in (
        "estimate_sensitivity_maps",
        "sensitivity_maps_type",
        "sensitivity_maps_espirit_threshold",
        "sensitivity_maps_espirit_kernel_size",
        "sensitivity_maps_espirit_crop",
        "sensitivity_maps_espirit_max_iters",
        "sensitivity_maps_gaussian",
    ):
        if key in transforms:
            sens[key] = transforms[key]
    if sens:
        out["sensitivity_map_estimation"] = sens

    norm: dict[str, Any] = {}
    for key in ("scaling_key", "scale_percentile"):
        if key in transforms:
            norm[key] = transforms[key]
    if norm:
        out["normalization"] = norm

    passthrough = (
        "padding_eps",
        "estimate_body_coil_image",
        "use_acs_as_mask",
        "delete_acs_mask",
        "delete_kspace",
        "image_recon_type",
        "compress_coils",
        "pad_coils",
        "use_seed",
        "transforms_type",
        "mask_split_ratio",
        "mask_split_acs_region",
        "mask_split_keep_acs",
        "mask_split_type",
        "mask_split_gaussian_std",
        "mask_split_half_direction",
        "target_acceleration",
        "dynamic_mask",
    )
    for key in passthrough:
        if key in transforms:
            out[key] = transforms[key]

    return out


def _strip_null_engine(cfg_section: dict[str, Any]) -> None:
    for key in list(cfg_section.keys()):
        if key == "engine_name" and cfg_section[key] is None:
            del cfg_section[key]


def convert_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy config dict in-place copy to modern format."""
    out = copy.deepcopy(cfg)

    if "model" in out:
        _strip_null_engine(out["model"])
    if "additional_models" in out:
        for sub in out["additional_models"].values():
            if isinstance(sub, dict):
                _strip_null_engine(sub)

    if "physics" in out:
        out["physics"].pop("use_noise_matrix", None)
        out["physics"].pop("noise_matrix_scaling", None)

    for section in ("training", "validation"):
        if section not in out:
            continue
        datasets = out[section].get("datasets")
        if not datasets:
            continue
        for ds in datasets:
            if "transforms" in ds:
                ds["transforms"] = _modernize_transforms(ds["transforms"])

    if "inference" in out and "dataset" in out["inference"]:
        ds = out["inference"]["dataset"]
        if "transforms" in ds:
            ds["transforms"] = _modernize_transforms(ds["transforms"])
        if ds.get("name") in (None, "???"):
            ds.pop("name", None)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Legacy config YAML")
    parser.add_argument("output", type=Path, help="Modernized config YAML")
    args = parser.parse_args()

    with args.input.open() as f:
        cfg = yaml.safe_load(f)

    modern = convert_config(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(modern, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote modernized config to {args.output}")


if __name__ == "__main__":
    main()
