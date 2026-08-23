#!/usr/bin/env python3
"""Rewrite pre-ModConv DIRECT checkpoints to the current state_dict layout.

Loads each ``*.pt``, remaps Sequential U-Net keys (``layers.0`` / ``layers.4``,
``up_conv.N.1`` → ``conv_out``, etc.) onto ``layer_*.conv`` / ``conv_out``, then
overwrites the file so current DIRECT can ``load_state_dict`` strictly without
runtime remapping.

Usage (on a machine with the Hub trees)::

    conda activate direct
    python projects/scripts/rewrite_legacy_checkpoints.py /projects/direct/hub_models/direct-calgary-campinas
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("rewrite_legacy_checkpoints")

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
            # Non-modulated ModConv often uses bias=NONE; drop orphan legacy bias.
            n_mapped += 1
            continue

        if new_key != key:
            n_mapped += 1
        remapped[new_key] = value

    if n_mapped:
        logger.info("Remapped %s legacy Unet keys.", n_mapped)
    return remapped


def remap_legacy_mwcnn_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remap pre-ModConv MWCNN Sequential ``net.N`` keys onto ``conv`` / ``conv1`` / ``conv2``.

    Also renames XPDNet wrapper indices: ``image_model_list.i.0`` → ``…mwcnn``,
    ``image_model_list.i.1`` → ``…out_conv``.
    """
    remapped: dict[str, Any] = {}
    n_mapped = 0
    for key, value in state_dict.items():
        new_key = key
        # down.*.convblock{i}.net.0 → down.*.convblock.conv
        new_key = re.sub(r"\.convblock\d+\.net\.0\.(weight|bias)$", r".convblock.conv.\1", new_key)
        # down.*.dilconvblock{i}.net.0 / net.2 → dilconvblock.conv1 / conv2
        new_key = re.sub(r"\.dilconvblock\d+\.net\.0\.(weight|bias)$", r".dilconvblock.conv1.\1", new_key)
        new_key = re.sub(r"\.dilconvblock\d+\.net\.2\.(weight|bias)$", r".dilconvblock.conv2.\1", new_key)
        # up.*.invconvblock{i}.net.0 → up.*.convblock.conv
        new_key = re.sub(r"\.invconvblock-?\d+\.net\.0\.(weight|bias)$", r".convblock.conv.\1", new_key)
        new_key = re.sub(r"\.invdilconvblock-?\d+\.net\.0\.(weight|bias)$", r".dilconvblock.conv1.\1", new_key)
        new_key = re.sub(r"\.invdilconvblock-?\d+\.net\.2\.(weight|bias)$", r".dilconvblock.conv2.\1", new_key)
        # XPDNet: Sequential(MWCNN, Conv2d) → named modules
        new_key = re.sub(r"(image_model_list\.\d+)\.0\.", r"\1.mwcnn.", new_key)
        new_key = re.sub(r"(image_model_list\.\d+)\.1\.(weight|bias)$", r"\1.out_conv.\2", new_key)

        if new_key != key:
            n_mapped += 1
        remapped[new_key] = value

    if n_mapped:
        logger.info("Remapped %s legacy MWCNN/XPDNet keys.", n_mapped)
    return remapped


def needs_remap(state_dict: Mapping[str, Any]) -> bool:
    return any(
        k.endswith((".layers.0.weight", ".layers.4.weight", "out_block.0.weight", "out_block.0.bias"))
        or re.search(r"up_conv\.\d+\.1\.(weight|bias)$", k)
        or re.search(r"up_transpose_conv\.\d+\.layer_1\.conv\.(weight|bias)$", k)
        or re.search(r"\.(?:inv)?(?:dil)?convblock-?\d+\.net\.\d+\.(weight|bias)$", k)
        or re.search(r"image_model_list\.\d+\.0\.", k)
        or re.search(r"image_model_list\.\d+\.1\.(weight|bias)$", k)
        for k in state_dict
    )


def remap_legacy_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Apply Unet then MWCNN legacy remaps."""
    return remap_legacy_mwcnn_state_dict(remap_legacy_unet_state_dict(state_dict))


def rewrite_checkpoint(path: Path, *, dry_run: bool = False) -> tuple[bool, str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return False, "not a dict checkpoint"

    changed = False
    for section in MODEL_SECTIONS:
        if section not in ckpt or not isinstance(ckpt[section], dict):
            continue
        sd = ckpt[section]
        if not needs_remap(sd):
            continue
        ckpt[section] = remap_legacy_state_dict(sd)
        changed = True

    if not changed:
        return False, "already modern keys"

    if dry_run:
        return True, "would rewrite"

    # Backup once
    bak = path.with_suffix(path.suffix + ".pre_modconv.bak")
    if not bak.exists():
        path.replace(bak)
        torch.save(ckpt, path)
        # restore naming: we moved original to bak, wrote new to path
    else:
        torch.save(ckpt, path)

    return True, f"rewrote (backup={bak.name})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Directories to scan for *.pt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    pts: list[Path] = []
    for root in args.roots:
        pts.extend(sorted(root.rglob("*.pt")))
    # Skip backups
    pts = [p for p in pts if not p.name.endswith(".bak") and ".pre_modconv.bak" not in p.name]

    n_rewrite = 0
    for path in pts:
        try:
            did, msg = rewrite_checkpoint(path, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: ERROR %s", path, exc)
            continue
        if did:
            n_rewrite += 1
            logger.info("%s: %s", path, msg)
        else:
            logger.info("%s: %s", path, msg)

    logger.info("Done. Rewrote %s / %s checkpoints.", n_rewrite, len(pts))


if __name__ == "__main__":
    main()
