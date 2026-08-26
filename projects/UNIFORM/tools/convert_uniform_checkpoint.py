#!/usr/bin/env python3
"""Rewrite pre-ModConv DIRECT checkpoints to the current state_dict layout.

Loads a ``*.pt``, remaps Sequential U-Net keys (``layers.0`` / ``layers.4``,
``up_conv.N.1`` → ``conv_out``, etc.) onto ``layer_*.conv`` / ``conv_out``, then
writes a new checkpoint so current DIRECT can ``load_state_dict`` strictly.

Based on the Hub rewrite used for calgary-campinas / multianatomy releases.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from direct.utils.models import fix_state_dict_module_prefix

logger = logging.getLogger("convert_uniform_checkpoint")

MODEL_SECTIONS = ("model", "sensitivity_model", "sampling_model", "registration_model")


def remap_legacy_unet_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remap pre-ModConv U-Net / initializer keys onto the current layout."""
    remapped: dict[str, Any] = {}
    n_mapped = 0
    for key, value in state_dict.items():
        new_key = key

        # Final head used to be up_conv.{last} = Sequential(ConvBlock, Conv2d 1x1).
        m_head = re.fullmatch(r"(.*?)up_conv\.(\d+)\.1\.(weight|bias)", key)
        if m_head:
            new_key = f"{m_head.group(1)}conv_out.{m_head.group(3)}"
        elif re.search(r"up_conv\.\d+\.0\.layers\.0\.weight$", key):
            new_key = re.sub(r"up_conv\.(\d+)\.0\.layers\.0\.weight$", r"up_conv.\1.layer_1.conv.weight", key)
        elif re.search(r"up_conv\.\d+\.0\.layers\.4\.weight$", key):
            new_key = re.sub(r"up_conv\.(\d+)\.0\.layers\.4\.weight$", r"up_conv.\1.layer_2.conv.weight", key)
        elif "up_transpose_conv." in key and key.endswith(".layers.0.weight"):
            new_key = key[: -len(".layers.0.weight")] + ".conv.weight"
        elif re.search(r"up_transpose_conv\.\d+\.layer_1\.conv\.(weight|bias)$", key):
            # Fix mistaken Sequential→ConvBlock remap on TransposeConvBlock.
            new_key = re.sub(
                r"up_transpose_conv\.(\d+)\.layer_1\.conv\.(weight|bias)$",
                r"up_transpose_conv.\1.conv.\2",
                key,
            )
        elif key.endswith(".layers.0.weight"):
            new_key = key[: -len(".layers.0.weight")] + ".layer_1.conv.weight"
        elif key.endswith(".layers.4.weight"):
            new_key = key[: -len(".layers.4.weight")] + ".layer_2.conv.weight"
        elif key.endswith("out_block.0.weight"):
            new_key = key[: -len("out_block.0.weight")] + "out_block.weight"
        elif key.endswith("out_block.0.bias"):
            # Keep bias when present (UNIFORM initializer uses it; some ModConv
            # configs use bias=NONE and can drop it separately if unused).
            new_key = key[: -len("out_block.0.bias")] + "out_block.bias"

        if new_key != key:
            n_mapped += 1

        # Keep denoiser conv_out.bias: the original Sequential U-Net always had a
        # biased 1x1 head. Load with image_unet_conv_out_bias=true so these match.
        remapped[new_key] = value

    if n_mapped:
        logger.info("Remapped %s legacy Unet keys.", n_mapped)
    return remapped


def needs_remap(state_dict: Mapping[str, Any]) -> bool:
    return any(
        k.endswith((".layers.0.weight", ".layers.4.weight", "out_block.0.weight", "out_block.0.bias"))
        or re.search(r"up_conv\.\d+\.1\.(weight|bias)$", k)
        or re.search(r"up_transpose_conv\.\d+\.layer_1\.conv\.(weight|bias)$", k)
        for k in state_dict
    )


def remap_legacy_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Apply Unet legacy remaps."""
    return remap_legacy_unet_state_dict(fix_state_dict_module_prefix(dict(state_dict)))


def convert_checkpoint(input_path: Path, output_path: Path, source_note: str | None = None) -> None:
    """Convert a full checkpoint or weights-only file."""
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)}")

    if "model" in checkpoint:
        converted = {k: v for k, v in checkpoint.items() if k not in ("optimizer", "lr_scheduler", "scaler")}
        for section in MODEL_SECTIONS:
            if section not in converted or not isinstance(converted[section], dict):
                continue
            sd = converted[section]
            if needs_remap(sd) or any(k.endswith((".layers.0.weight", ".layers.4.weight")) for k in sd):
                converted[section] = remap_legacy_state_dict(sd)
            else:
                converted[section] = fix_state_dict_module_prefix(sd)
        converted["__converted_from__"] = str(input_path)
        converted["__converted_note__"] = (
            source_note or "Pre-ModConv Sequential U-Net keys rewritten to ModConv ConvModule layout."
        )
        converted["__datetime__"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        converted = {
            "model": remap_legacy_state_dict(checkpoint),
            "__converted_from__": str(input_path),
            "__converted_note__": source_note
            or "Pre-ModConv Sequential U-Net keys rewritten to ModConv ConvModule layout.",
            "__datetime__": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    logger.info("Saved converted checkpoint to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Legacy checkpoint (.pt)")
    parser.add_argument("output", type=Path, help="Converted checkpoint (.pt)")
    parser.add_argument("--note", type=str, default=None, help="Optional conversion note")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to overwrite")

    convert_checkpoint(args.input, args.output, args.note)


if __name__ == "__main__":
    main()
