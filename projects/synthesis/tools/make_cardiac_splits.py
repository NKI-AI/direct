#!/usr/bin/env python3
"""CMRxRecon Cine FullSample splits (2023 + 2024), paper fractions, seed 42.

Creates flat ``*.mat`` symlinks for :class:`~direct.data.datasets.CMRxReconDataset`
(``kspace_key=kspace_full``). Patient ids are year-prefixed so 2023/2024 ``P001``
do not collide.

Train patients: S_train 55% / S_val 5% / R_train 40% (stratified by modality set).
ValidationSet FullSample patients: R_val / R_test 50/50.

Does **not** rewrite brain/knee/prostate/breast lists.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py

SEED = 42
LISTS = Path("/home/g.yiasemis/projects/direct/projects/synthesis/lists/cardiac")
MANIFEST = Path("/home/g.yiasemis/projects/direct/projects/synthesis/lists/paper_split_manifest.json")
SYMLINK_ROOT = Path("/projects/direct/synthesis_diffusion/cmrxrecon_cine")

SOURCES = {
    "2023": {
        "train": Path(
            "/mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2023/ChallegeData/MultiCoil/Cine/TrainingSet/FullSample"
        ),
        "val": Path(
            "/mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2023/ChallegeData/MultiCoil/Cine/ValidationSet/FullSample"
        ),
    },
    "2024": {
        "train": Path(
            "/mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample"
        ),
        "val": Path(
            "/mnt/rhea/data_public/CMRxRecon/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/ValidationSet/FullSample"
        ),
    },
}

TRAIN_FRACS = [("s_train", 0.55), ("s_val", 0.05), ("r_train", 0.40)]
VAL_FRACS = [("r_val", 0.50), ("r_test", 0.50)]


def _scan_mat(path: Path) -> dict | None:
    try:
        with h5py.File(path, "r") as handle:
            if "kspace_full" not in handle:
                return None
            shape = tuple(handle["kspace_full"].shape)
    except OSError as exc:
        print(f"SKIP {path}: {exc}", file=sys.stderr)
        return None
    # (nz, nt, nc, ky, kx)
    if len(shape) < 5:
        print(f"SKIP {path}: unexpected shape {shape}", file=sys.stderr)
        return None
    return {"ncoils": int(shape[2]), "shape": shape}


def _link_year(year: str, src_root: Path, dst_root: Path) -> list[dict]:
    dst_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for patient_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("P")):
        for mat in sorted(patient_dir.glob("cine_*.mat")):
            meta = _scan_mat(mat)
            if meta is None:
                continue
            # Year-prefixed flat name for CMRxReconDataset root glob / lists.
            name = f"y{year}_{patient_dir.name}_{mat.name}"  # y2024_P001_cine_sax.mat
            link = dst_root / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(mat.resolve())
            rows.append(
                {
                    "name": name,
                    "patient": f"{year}_{patient_dir.name}",
                    "year": year,
                    "modality": mat.stem,
                    "ncoils": meta["ncoils"],
                    "src": str(mat),
                }
            )
    return rows


def _split_ids(ids: list[str], rng: random.Random, fracs: list[tuple[str, float]]) -> dict[str, list[str]]:
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    assigned = {name: [] for name, _ in fracs}
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


def _write_lst(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(names))
    path.write_text("\n".join(unique) + ("\n" if unique else ""))


def main() -> int:
    rng = random.Random(SEED)
    # Wipe previous flat links (old unprefixed 2023-only names).
    for sub in ("train", "val"):
        dest = SYMLINK_ROOT / sub
        if dest.exists():
            for path in dest.glob("*.mat"):
                path.unlink()
        dest.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for year, paths in SOURCES.items():
        print(f"Linking {year} train {paths['train']}")
        train_rows.extend(_link_year(year, paths["train"], SYMLINK_ROOT / "train"))
        print(f"Linking {year} val {paths['val']}")
        val_rows.extend(_link_year(year, paths["val"], SYMLINK_ROOT / "val"))

    print(f"train files={len(train_rows)} val files={len(val_rows)}")
    print("train coils", dict(Counter(r["ncoils"] for r in train_rows)))
    print("val coils", dict(Counter(r["ncoils"] for r in val_rows)))

    patient_files: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        patient_files[row["patient"]].append(row)

    # Stratify by year + modality set (sax / lax / lvot / combinations)
    strata: dict[str, list[str]] = defaultdict(list)
    for patient, files in patient_files.items():
        year = files[0]["year"]
        mods = "+".join(sorted({f["modality"] for f in files}))
        strata[f"{year}:{mods}"].append(patient)

    split_patients: dict[str, set[str]] = {name: set() for name, _ in TRAIN_FRACS}
    for key, patients in sorted(strata.items()):
        part = _split_ids(patients, rng, TRAIN_FRACS)
        for split, ids in part.items():
            split_patients[split].update(ids)
        print(f"  stratum {key}: {{{', '.join(f'{s}:{len(ids)}' for s, ids in part.items())}}}")

    split_files: dict[str, list[str]] = {name: [] for name, _ in TRAIN_FRACS}
    for split, patients in split_patients.items():
        for patient in patients:
            split_files[split].extend(f["name"] for f in patient_files[patient])

    val_patients = sorted({r["patient"] for r in val_rows})
    # Stratify val by year
    val_by_year: dict[str, list[str]] = defaultdict(list)
    for patient in val_patients:
        val_by_year[patient.split("_", 1)[0]].append(patient)
    for year, patients in sorted(val_by_year.items()):
        part = _split_ids(patients, rng, VAL_FRACS)
        print(f"  val year {year}: {{{', '.join(f'{s}:{len(ids)}' for s, ids in part.items())}}}")
        for split, ids in part.items():
            split_files.setdefault(split, [])
            for patient in ids:
                split_files[split].extend(r["name"] for r in val_rows if r["patient"] == patient)

    for split, names in split_files.items():
        _write_lst(LISTS / f"{split}.lst", names)
        print(f"  {split}: {len(names)} files")

    # Coil-grouped train lists (CMRxRecon batches must not mix coil counts)
    by_name = {r["name"]: r for r in train_rows + val_rows}
    for split in ("s_train", "r_train"):
        groups: dict[int, list[str]] = defaultdict(list)
        for name in split_files[split]:
            groups[int(by_name[name]["ncoils"])].append(name)
        for ncoil, names in sorted(groups.items()):
            _write_lst(LISTS / f"{split}_{ncoil}_coils.lst", names)
            print(f"  {split}_{ncoil}_coils: {len(names)}")

    # Tiny smoke lists (2 train + 2 val volumes) for dummy CMRxRecon runs
    smoke_train = sorted(split_files["s_train"])[:2]
    smoke_val = sorted(split_files["s_val"])[:2]
    _write_lst(LISTS / "s_train_smoke.lst", smoke_train)
    _write_lst(LISTS / "s_val_smoke.lst", smoke_val)

    summary = {
        "seed": SEED,
        "root_train": str(SYMLINK_ROOT / "train"),
        "root_val": str(SYMLINK_ROOT / "val"),
        "sources": {y: {k: str(v) for k, v in p.items()} for y, p in SOURCES.items()},
        "counts": {k: len(v) for k, v in split_files.items()},
        "n_train_patients": len(patient_files),
        "n_val_patients": len(val_patients),
        "train_coils": dict(Counter(r["ncoils"] for r in train_rows)),
        "smoke_train": smoke_train,
        "smoke_val": smoke_val,
    }
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
    else:
        data = {"seed": SEED, "anatomies": {}}
    data.setdefault("anatomies", {})["cardiac"] = summary
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Updated {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
