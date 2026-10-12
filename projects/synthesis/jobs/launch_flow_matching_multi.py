#!/usr/bin/env python3
"""2-GPU entrypoint. Spawn cannot re-import a shell heredoc (``<stdin>``)."""

from __future__ import annotations

import sys

import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from direct.cli.cli import main

DIRECT_ROOT = "/home/g.yiasemis/projects/direct"
CFG = f"{DIRECT_ROOT}/projects/synthesis/flow_matching_multi.yaml"
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
        "flow_matching_multi",
        "--mixed-precision",
        "--resume",
        "--force-validation",
    ]
    main()
