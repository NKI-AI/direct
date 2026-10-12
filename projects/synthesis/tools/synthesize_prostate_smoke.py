#!/usr/bin/env python3
"""Smoke: phase + sensitivity synthesis on 1–2 prostate S_val volumes (320×320).

Standalone entrypoint (mama_mia-style). Same pipeline as ``direct synthesize``:

  magnitude m → SensitivityMapSynthesizer → complex S (∑|S|²=1)
             → PhaseFlowMatching.sample → φ
             → k = FFT(S · m · e^{iφ})

Writes H5 + GT-vs-synth plots under ``--out``. Prefer the notebook
``projects/synthesis/synthesize_prostate_smoke.ipynb`` for interactive plots.

Example::

  python projects/synthesis/tools/synthesize_prostate_smoke.py \\
    --device cuda
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch


def main(argv: list[str] | None = None) -> int:
    root = pathlib.Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("/projects/direct/synthesis_diffusion/synthesize_prostate_smoke"),
    )
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        default=root / "projects/synthesis/synthesize_prostate_smoke.yaml",
    )
    parser.add_argument(
        "--phase-model",
        type=pathlib.Path,
        default=pathlib.Path("/projects/direct/synthesis_diffusion/flow_matching_multi/model_290000.pt"),
    )
    parser.add_argument(
        "--sensitivity-model",
        type=pathlib.Path,
        default=pathlib.Path("/projects/direct/synthesis_diffusion/sensitivity_maps_multi_500k/model_110000.pt"),
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/rhea/data_public/fastmri/prostate/training/T2"),
    )
    parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        default=root / "projects/synthesis/lists/prostate/s_val_smoke.lst",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cudnn-off", action="store_true", default=True)
    args = parser.parse_args(argv)

    if args.cudnn_off and args.device.startswith("cuda"):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    from direct.synthesize import run_synthesize

    run_synthesize(
        output_directory=args.out,
        data_root=args.data_root,
        cfg_file=args.cfg,
        phase_checkpoint=args.phase_model,
        sensitivity_checkpoint=args.sensitivity_model,
        experiment_directory=args.out,
        filenames_filter=args.filenames_filter,
        device=args.device,
        num_workers=args.num_workers,
        name="synthesize_prostate_smoke",
        seed=args.seed,
    )
    print(f"Plots → {args.out / 'plots'}")
    print(f"H5    → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
