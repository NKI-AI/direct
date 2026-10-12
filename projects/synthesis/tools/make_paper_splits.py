#!/usr/bin/env python3
"""Patient/volume splits for magnitude→k-space synthesis vs real recon.

Train volumes (official FastMRI train, or the single breast folder):
  S_train / S_val / R_train  = 55 / 5 / 40  (breast: 45 / 5 / 40, remaining 10% → R_val+R_test)

Official val (brain, knee, prostate T2):
  R_val / R_test = 50 / 50, stratified by brain contrast.

S_train and R_train are further split by coil count. Seed 42.
Does not overwrite the older brain ``train_*_coils.lst`` used by running jobs.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py

SEED = 42
LISTS_ROOT = Path("/home/g.yiasemis/projects/direct/projects/synthesis")
SYNTHESIS_DIR = LISTS_ROOT
YAML_PATH = SYNTHESIS_DIR / "flow_matching_multi.yaml"

SOURCES = {
    "brain": {
        "train": Path("/mnt/rhea/data_public/fastmri/brain/multicoil_train"),
        "val": Path("/mnt/rhea/data_public/fastmri/brain/multicoil_val"),
    },
    "knee": {
        "train": Path("/mnt/rhea/data_public/fastmri/knee/multicoil_train"),
        "val": Path("/mnt/rhea/data_public/fastmri/knee/multicoil_val"),
    },
    "prostate": {
        "train": Path("/mnt/rhea/data_public/fastmri/prostate/training/T2"),
        "val": Path("/mnt/rhea/data_public/fastmri/prostate/validation/T2"),
    },
    "breast": {
        "train": Path("/mnt/rhea/data_public/fastmri/breast/derived/cartesian"),
        "val": None,
    },
}

TRAIN_FRACS = {"s_train": 0.55, "s_val": 0.05, "r_train": 0.40}
BREAST_FRACS = {
    "s_train": 0.45,
    "s_val": 0.05,
    "r_train": 0.40,
    "r_val": 0.05,
    "r_test": 0.05,
}

BRAIN_MODALITY_RE = re.compile(r"^file_brain_([A-Za-z0-9]+)_", re.IGNORECASE)
BREAST_PATIENT_RE = re.compile(r"fastMRI_breast_(\d+)_", re.IGNORECASE)
PROSTATE_ID_RE = re.compile(r"file_prostate_[A-Za-z0-9]+_(\d+)", re.IGNORECASE)


def _scan_one(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        shape = tuple(handle["kspace"].shape)
        acquisition = handle.attrs.get("acquisition", "")
        if isinstance(acquisition, bytes):
            acquisition = acquisition.decode("utf-8", errors="replace")
    return {
        "name": path.name,
        "ncoils": int(shape[1]),
        "nslices": int(shape[0]),
        "acquisition": str(acquisition),
    }


def scan_root(root: Path) -> list[dict]:
    files = sorted(root.glob("*.h5"))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_scan_one, path): path for path in files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {path}: {exc}", file=sys.stderr)
    rows.sort(key=lambda row: row["name"])
    return rows


def brain_modality(name: str, acquisition: str) -> str:
    match = BRAIN_MODALITY_RE.match(name)
    if match:
        return match.group(1).upper()
    if acquisition:
        return str(acquisition).upper()
    return "UNKNOWN"


def knee_modality(name: str, acquisition: str) -> str:
    text = f"{name} {acquisition}".upper()
    if "PDFS" in text or "PD_FS" in text:
        return "PDFS"
    if "PD" in text:
        return "PD"
    return "KNEE"


def patient_key(anatomy: str, name: str) -> str:
    if anatomy == "breast":
        match = BREAST_PATIENT_RE.search(name)
        return f"breast_{match.group(1)}" if match else name
    if anatomy == "prostate":
        match = PROSTATE_ID_RE.search(name)
        return f"prostate_{match.group(1)}" if match else name
    if anatomy == "brain":
        return name
    return name


def stratum(anatomy: str, name: str, acquisition: str) -> str:
    if anatomy == "brain":
        return brain_modality(name, acquisition)
    if anatomy == "knee":
        return knee_modality(name, acquisition)
    return anatomy


def split_ids(ids: list[str], rng, fracs: list[tuple[str, float]]) -> dict[str, list[str]]:
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    assigned: dict[str, list[str]] = {name: [] for name, _ in fracs}
    if n == 0:
        return assigned
    acc = 0.0
    cuts: list[tuple[str, int]] = []
    for name, frac in fracs:
        acc += frac
        cuts.append((name, round(acc * n)))
    cuts[-1] = (cuts[-1][0], n)
    start = 0
    for name, end in cuts:
        assigned[name] = shuffled[start:end]
        start = end
    return assigned


def write_lst(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(names))
    path.write_text("\n".join(unique) + ("\n" if unique else ""))


def coil_group(names: list[str], by_name: dict[str, dict]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for name in names:
        groups[int(by_name[name]["ncoils"])].append(name)
    return {coil: sorted(files) for coil, files in sorted(groups.items())}


def assert_disjoint(split_to_files: dict[str, list[str]], anatomy: str) -> None:
    seen: dict[str, str] = {}
    for split, names in split_to_files.items():
        for name in names:
            if name in seen:
                raise RuntimeError(f"{anatomy}: {name} in {seen[name]} and {split}")
            seen[name] = split


TRANSFORM_BLOCK = """\
        transforms_type: SYNTHESIS
        use_seed: {use_seed}
        masking: null
        cropping:
          crop: "[320, 320]"
          image_center_crop: true
        augmentation:
          pad:
            - 320
            - 320
        sensitivity_map_estimation:
          estimate_sensitivity_maps: true
          sensitivity_maps_type: FULL_RSS
          sensitivity_maps_gaussian: null
        normalization:
          scaling_key: kspace
          scale_percentile: 0.99
        synthesis_calibration:
          synthesis_extract_phase_and_maps: true
          coil_map_mode: acs"""


def dataset_yaml(
    data_root: Path,
    rel_lst: str,
    description: str,
    use_seed: bool,
) -> str:
    seed = "false" if not use_seed else "true"
    return (
        "    - name: FastMRI\n"
        f"      data_root: {data_root}\n"
        "      filenames_lists:\n"
        f"        - {rel_lst}\n"
        f"      filenames_lists_root: {SYNTHESIS_DIR}\n"
        "      pass_attrs: true\n"
        "      transforms:\n"
        f"{TRANSFORM_BLOCK.format(use_seed=seed)}\n"
        f"      text_description: {description}\n"
    )


def write_yaml(train_entries: list[tuple[Path, str, str]], val_entries: list[tuple[Path, str, str]]) -> None:
    chunks = [
        "# Multi-anatomy conditional phase flow matching (paper split).\n",
        "# S_train coil groups from brain/knee/prostate/breast. Organ AdaIN + 128-ch U-Net.\n",
        "# Do not pass --training-root (each dataset has data_root).\n",
        "#\n",
        "#   direct train /projects/direct/synthesis_diffusion \\\n",
        "#     --cfg projects/synthesis/flow_matching_multi.yaml \\\n",
        "#     --num-gpus 2 --num-workers 8 --device cuda --name flow_matching_multi\n",
        "\n",
        "physics:\n",
        "  forward_operator: fft2\n",
        "  backward_operator: ifft2\n",
        "model:\n",
        "  model_name: synthesis.synthesis.PhaseFlowMatching\n",
        "  engine_name: PhaseFlowMatchingEngine\n",
        "  num_filters: 128\n",
        "  num_pool_layers: 4\n",
        "  num_integration_steps: 50\n",
        "  eval_integration_steps: 30\n",
        "  num_organs: 4\n",
        "  organ_embed_dim: 16\n",
        "\n",
        "training:\n",
        "  datasets:\n",
    ]
    for root, rel_lst, desc in train_entries:
        chunks.append(dataset_yaml(root, rel_lst, desc, use_seed=False))
    chunks.extend(
        [
            "  batch_size: 4\n",
            "  optimizer: AdamW\n",
            "  lr: 0.0001\n",
            "  weight_decay: 0.01\n",
            "  lr_step_size: 50000\n",
            "  lr_gamma: 0.9\n",
            "  lr_warmup_iter: 1000\n",
            "  num_iterations: 885000\n",
            "  validation_steps: 5000\n",
            "  gradient_clipping: 1.0\n",
            "  checkpointer:\n",
            "    checkpoint_steps: 5000\n",
            "  loss:\n",
            "    losses:\n",
            "      - function: velocity_loss\n",
            "        multiplier: 1.0\n",
            "  metrics:\n",
            "    - phase_mae_deg\n",
            "    - phase_chordal\n",
            "    - phase_gradient\n",
            "\n",
            "validation:\n",
            "  datasets:\n",
        ]
    )
    for root, rel_lst, desc in val_entries:
        chunks.append(dataset_yaml(root, rel_lst, desc, use_seed=True))
    chunks.extend(
        [
            "  batch_size: 1\n",
            "  metrics:\n",
            "    - phase_mae_deg\n",
            "    - phase_chordal\n",
            "    - phase_gradient\n",
            "\n",
            "logging:\n",
            "  log_interval: 20\n",
            "  tensorboard:\n",
            "    num_images: 4\n",
            "\n",
            "inference:\n",
            "  batch_size: 1\n",
            "  crop: null\n",
            "  metrics: []\n",
            "  dataset:\n",
            "    name: FastMRI\n",
            "    pass_attrs: true\n",
            "    data_root: /mnt/rhea/data_public/fastmri/brain/multicoil_train\n",
            "    filenames_lists:\n",
            "      - lists/brain/s_val.lst\n",
            "    filenames_lists_root: /home/g.yiasemis/projects/direct/projects/synthesis\n",
            "    transforms:\n",
            f"{TRANSFORM_BLOCK.format(use_seed='true')}\n",
            "    text_description: brain_s_val\n",
        ]
    )
    YAML_PATH.write_text("".join(chunks))


def main() -> int:
    import random

    rng = random.Random(SEED)
    summary: dict[str, Any] = {"seed": SEED, "anatomies": {}}
    train_yaml: list[tuple[Path, str, str]] = []
    val_yaml: list[tuple[Path, str, str]] = []

    for anatomy, paths in SOURCES.items():
        train_root = paths["train"]
        val_root = paths["val"]
        print(f"Scanning {anatomy} train {train_root} ...")
        train_rows = scan_root(train_root)
        val_rows = scan_root(val_root) if val_root is not None else []
        print(f"  train files={len(train_rows)} val files={len(val_rows)}")

        by_name = {row["name"]: row for row in train_rows + val_rows}
        train_names = [row["name"] for row in train_rows]
        val_names = [row["name"] for row in val_rows]

        units: dict[str, list[str]] = defaultdict(list)
        unit_stratum: dict[str, str] = {}
        for name in train_names:
            key = patient_key(anatomy, name)
            units[key].append(name)
            unit_stratum[key] = stratum(anatomy, name, by_name[name]["acquisition"])

        by_stratum: dict[str, list[str]] = defaultdict(list)
        for key in units:
            by_stratum[unit_stratum[key]].append(key)

        fracs = list(BREAST_FRACS.items()) if anatomy == "breast" else list(TRAIN_FRACS.items())
        split_units: dict[str, list[str]] = defaultdict(list)
        for stratum_name, keys in by_stratum.items():
            part = split_ids(keys, rng, fracs)
            for split_name, assigned in part.items():
                split_units[split_name].extend(assigned)
            print(f"  stratum {stratum_name}: units={len(keys)} " + " ".join(f"{k}={len(v)}" for k, v in part.items()))

        split_files: dict[str, list[str]] = defaultdict(list)
        for split_name, keys in split_units.items():
            for key in keys:
                split_files[split_name].extend(units[key])

        if anatomy != "breast":
            val_units: dict[str, list[str]] = defaultdict(list)
            val_stratum: dict[str, str] = {}
            for name in val_names:
                key = patient_key(anatomy, name)
                val_units[key].append(name)
                val_stratum[key] = stratum(anatomy, name, by_name[name]["acquisition"])
            val_by_stratum: dict[str, list[str]] = defaultdict(list)
            for key in val_units:
                val_by_stratum[val_stratum[key]].append(key)
            for stratum_name, keys in val_by_stratum.items():
                part = split_ids(keys, rng, [("r_val", 0.5), ("r_test", 0.5)])
                for split_name, assigned in part.items():
                    for key in assigned:
                        split_files[split_name].extend(val_units[key])

        assert_disjoint(split_files, anatomy)

        out_dir = SYNTHESIS_DIR / "lists" / anatomy
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, names in split_files.items():
            write_lst(out_dir / f"{split_name}.lst", names)

        for split_name in ("s_train", "r_train"):
            groups = coil_group(split_files.get(split_name, []), by_name)
            for ncoils, names in groups.items():
                rel = f"lists/{anatomy}/{split_name}_{ncoils}_coils.lst"
                write_lst(SYNTHESIS_DIR / rel, names)
                if split_name == "s_train":
                    train_yaml.append((train_root, rel, f"{anatomy}_{split_name}_{ncoils}c"))

        s_val_rel = f"lists/{anatomy}/s_val.lst"
        val_yaml.append((train_root, s_val_rel, f"{anatomy}_s_val"))

        counts = {split: len(names) for split, names in split_files.items()}
        slices = {split: int(sum(by_name[name]["nslices"] for name in names)) for split, names in split_files.items()}
        modality_counts = defaultdict(lambda: defaultdict(int))
        for split, names in split_files.items():
            for name in names:
                modality_counts[split][stratum(anatomy, name, by_name[name]["acquisition"])] += 1

        summary["anatomies"][anatomy] = {
            "train_root": str(train_root),
            "val_root": str(val_root) if val_root else None,
            "files": counts,
            "slices": slices,
            "modalities": {split: dict(vals) for split, vals in modality_counts.items()},
            "s_train_coils": {str(k): len(v) for k, v in coil_group(split_files.get("s_train", []), by_name).items()},
            "r_train_coils": {str(k): len(v) for k, v in coil_group(split_files.get("r_train", []), by_name).items()},
        }

    write_yaml(train_yaml, val_yaml)
    manifest = SYNTHESIS_DIR / "lists" / "paper_split_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {YAML_PATH}")
    print(f"Wrote {manifest}")
    print(json.dumps({a: d["files"] for a, d in summary["anatomies"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
