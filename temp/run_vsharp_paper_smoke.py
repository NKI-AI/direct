#!/usr/bin/env python3
"""Smoke-test all vSHARP paper configs with a short training run."""

from __future__ import annotations

import copy
import pathlib
import subprocess
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG_ROOT = REPO / "projects" / "modulated_convolution" / "configs" / "vsharp"
OUT_DIR = REPO / "temp" / "vsharp_paper_smoke"
DEVICE = "mps"
NUM_ITERS = 3
NUM_WORKERS = 0

DATA = {
    "knee": {
        "training_root": pathlib.Path("/Users/g.yiasemis/Documents/data/public/fastmri/knee/"),
        "validation_root": pathlib.Path("/Users/g.yiasemis/Documents/data/public/fastmri/knee/val/"),
    },
    "prostate": {
        "training_root": pathlib.Path("/Users/g.yiasemis/Documents/data/public/fastmri/prostate/"),
        "validation_root": pathlib.Path("/Users/g.yiasemis/Documents/data/public/fastmri/prostate/"),
    },
}


def discover_configs() -> list[tuple[str, pathlib.Path]]:
    configs: list[tuple[str, pathlib.Path]] = []
    for anatomy in ("knee", "prostate"):
        anatomy_dir = CFG_ROOT / anatomy
        if not anatomy_dir.is_dir():
            continue
        for path in sorted(anatomy_dir.glob("*.yaml")):
            configs.append((f"{anatomy}/{path.stem}", path))
    return configs


def smoke_cfg(src: pathlib.Path) -> pathlib.Path:
    cfg = yaml.safe_load(src.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["training"]["num_iterations"] = NUM_ITERS
    cfg["training"]["validation_steps"] = NUM_ITERS + 1
    cfg["training"]["checkpointer"]["checkpoint_steps"] = NUM_ITERS + 1
    cfg["training"]["lr_warmup_iter"] = 0
    cfg["validation"]["batch_size"] = 1
    cfg["logging"]["tensorboard"]["num_images"] = 1

    smoke_dir = OUT_DIR / "configs"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    out = smoke_dir / src.name
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_one(name: str, cfg_path: pathlib.Path) -> tuple[bool, str]:
    anatomy = name.split("/", maxsplit=1)[0]
    paths = DATA[anatomy]
    if not paths["training_root"].is_dir():
        return False, f"Missing training data: {paths['training_root']}"
    if not paths["validation_root"].is_dir():
        return False, f"Missing validation data: {paths['validation_root']}"

    exp_dir = OUT_DIR / "experiments" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "direct",
        "train",
        str(exp_dir),
        "--cfg",
        str(cfg_path),
        "--training-root",
        str(paths["training_root"]),
        "--validation-root",
        str(paths["validation_root"]),
        "--device",
        DEVICE,
        "--num-workers",
        str(NUM_WORKERS),
    ]
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    output = result.stdout + result.stderr
    tail = "\n".join(output.splitlines()[-50:])
    return result.returncode == 0, tail


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Run only these config keys (e.g. knee/vsharp_triang).")
    args = parser.parse_args()

    configs = discover_configs()
    if not configs:
        print(f"No configs found under {CFG_ROOT}", file=sys.stderr)
        return 1

    if args.only:
        selected = {k: p for k, p in configs if k in args.only or any(k.endswith(x) for x in args.only)}
        if not selected:
            print(f"No matching configs for: {args.only}", file=sys.stderr)
            return 1
        configs = list(selected.items())

    results: dict[str, bool] = {}
    for name, src in configs:
        smoke_path = smoke_cfg(src)
        ok, tail = run_one(name, smoke_path)
        results[name] = ok
        print(f">>> {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(tail)

    print(f"\n{'=' * 70}\nSummary ({NUM_ITERS} iterations each)\n{'=' * 70}")
    failed = []
    for name, ok in results.items():
        print(f"  {name:55} {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}")
        return 1
    print(f"\nAll {len(results)} vSHARP paper configs passed smoke training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
