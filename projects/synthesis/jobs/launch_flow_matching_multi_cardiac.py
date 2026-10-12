#!/usr/bin/env python3
"""2-GPU entrypoint: phase flow matching + cardiac + unknown-organ dropout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DIRECT_ROOT = Path("/home/g.yiasemis/projects/direct")
BUILD_DIR = DIRECT_ROOT / "build" / "cp312"


def _disable_editable_rebuild() -> None:
    """Avoid meson-python ninja rebuild races on NFS under torch.multiprocessing.spawn.

    Copies built abi3 extensions into the source tree and removes the editable
    meta-path finder so imports resolve from ``DIRECT_ROOT`` without ninja.
    """
    pairs = (
        (BUILD_DIR / "_poisson.abi3.so", DIRECT_ROOT / "direct" / "common" / "_poisson.abi3.so"),
        (BUILD_DIR / "_gaussian.abi3.so", DIRECT_ROOT / "direct" / "common" / "_gaussian.abi3.so"),
        (BUILD_DIR / "_gaussian_fill.abi3.so", DIRECT_ROOT / "direct" / "ssl" / "_gaussian_fill.abi3.so"),
    )
    for src, dst in pairs:
        if not src.is_file():
            raise FileNotFoundError(f"missing extension {src}; run ninja -C {BUILD_DIR} first")
        if not dst.exists() or dst.stat().st_mtime_ns < src.stat().st_mtime_ns:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    sys.meta_path = [f for f in sys.meta_path if type(f).__name__ != "MesonpyMetaFinder"]
    sys.path_hooks = [
        h
        for h in sys.path_hooks
        if getattr(h, "__self__", None) is None or type(getattr(h, "__self__", None)).__name__ != "MesonpyMetaFinder"
    ]
    # Drop cached path finder entries that still point at the editable hook.
    sys.path_importer_cache.clear()
    root = str(DIRECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_disable_editable_rebuild()

import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from direct.cli.cli import main

CFG = f"{DIRECT_ROOT}/projects/synthesis/flow_matching_multi_cardiac.yaml"
EXP_DIR = "/projects/direct/synthesis_diffusion"

if __name__ == "__main__":
    sys.argv = [
        "direct",
        "train",
        EXP_DIR,
        "--cfg",
        CFG,
        "--num-gpus",
        "2",
        "--num-workers",
        "8",
        "--device",
        "cuda",
        "--name",
        "flow_matching_multi_cardiac",
        "--mixed-precision",
    ]
    main()
