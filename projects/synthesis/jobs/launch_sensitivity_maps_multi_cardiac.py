#!/usr/bin/env python3
"""1-GPU: sensitivity maps + cardiac + unknown-organ dropout (nearest upsample)."""

from __future__ import annotations

import sys

import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from direct.cli.cli import main

DIRECT_ROOT = "/home/g.yiasemis/projects/direct"
CFG = f"{DIRECT_ROOT}/projects/synthesis/sensitivity_maps_multi_cardiac.yaml"
EXP_DIR = "/projects/direct/synthesis_diffusion"

if __name__ == "__main__":
    sys.argv = [
        "direct",
        "train",
        EXP_DIR,
        "--cfg",
        CFG,
        "--num-gpus",
        "1",
        "--num-workers",
        "4",
        "--device",
        "cuda",
        "--name",
        "sensitivity_maps_multi_cardiac_500k",
        "--mixed-precision",
        "--resume",
    ]
    main()
