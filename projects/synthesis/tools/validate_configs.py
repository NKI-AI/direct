#!/usr/bin/env python3
"""Validate synthesis YAML configs without launching a run.

For each config this script:

1. merges the YAML into DIRECT's typed schema (same path as ``direct train``),
2. instantiates the model(s) and the engine on CPU,
3. builds the transform pipeline for every training / validation dataset,
4. checks that every ``filenames_lists`` file exists and, with ``--check-files``,
   that every listed volume exists under its ``data_root``.

Usage::

    python projects/synthesis/tools/validate_configs.py projects/synthesis/*.yaml --check-files
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import traceback

from omegaconf import DictConfig, OmegaConf

from direct.common.subsample import build_masking_function
from direct.config.defaults import DefaultConfig, InferenceConfig, TrainingConfig, ValidationConfig
from direct.data.mri_transforms import build_mri_transforms
from direct.environment import (
    build_operators,
    extract_names,
    initialize_models_from_config,
    load_dataset_config,
    load_models_into_environment_config,
    setup_engine,
)
from direct.utils import dict_flatten, remove_keys
from direct.utils.io import read_list


def load_config(path: pathlib.Path) -> DictConfig:
    """Merge a YAML file into the structured DIRECT config (mirrors ``setup_common_environment``)."""
    raw = OmegaConf.load(path)
    cfg = OmegaConf.structured(DefaultConfig)
    _, models_config = load_models_into_environment_config(raw)
    cfg.model = models_config.model
    del models_config["model"]
    cfg.additional_models = models_config
    new = raw.copy()
    for key in raw:
        if key in ("models", "additional_models"):
            continue
        if key in ("training", "validation", "inference"):
            if not raw[key]:
                continue
            schema = {"training": TrainingConfig, "validation": ValidationConfig, "inference": InferenceConfig}[key]
            cfg[key] = OmegaConf.structured(schema)
            if key in ("training", "validation"):
                for idx, (name, dataset_cfg) in enumerate(extract_names(raw[key].datasets)):
                    new[key].datasets[idx] = dataset_cfg
                    cfg[key].datasets.append(load_dataset_config(name))
            else:
                name, dataset_cfg = extract_names(raw[key].dataset)
                new[key].dataset = dataset_cfg
                cfg[key].dataset = load_dataset_config(name)
        cfg[key] = OmegaConf.merge(cfg[key], new[key])
    return cfg


def _list_paths(dataset_cfg: DictConfig, config_dir: pathlib.Path) -> list[pathlib.Path]:
    # ``direct train`` resolves relative lists against the config file's folder unless
    # ``filenames_lists_root`` is set.
    root = getattr(dataset_cfg, "filenames_lists_root", None) or config_dir
    lists = getattr(dataset_cfg, "filenames_lists", None) or []
    return [pathlib.Path(root) / name for name in lists]


def validate(path: pathlib.Path, check_files: bool) -> list[str]:
    """Return a list of problems (empty when the config is valid)."""
    problems: list[str] = []
    cfg = load_config(path)
    forward_operator, backward_operator = build_operators(cfg.physics)
    models, _ = load_models_into_environment_config(OmegaConf.load(path))
    model, additional_models = initialize_models_from_config(cfg, models, forward_operator, backward_operator, "cpu")
    engine = setup_engine(cfg, "cpu", model, additional_models, forward_operator, backward_operator)

    for split in ("training", "validation"):
        split_cfg = getattr(cfg, split, None)
        if split_cfg is None:
            continue
        for dataset_cfg in split_cfg.datasets:
            label = f"{split}/{getattr(dataset_cfg, 'text_description', None) or dataset_cfg.name}"
            masking = dataset_cfg.transforms.masking
            mask_func = None if masking is None else build_masking_function(**masking)
            builder = functools.partial(
                build_mri_transforms,
                forward_operator=engine.forward_operator,
                backward_operator=engine.backward_operator,
                mask_func=mask_func,
            )
            try:
                builder(**dict_flatten(dict(remove_keys(dataset_cfg.transforms, "masking"))))
            except Exception as exc:  # noqa: BLE001  # report every config error, keep going
                problems.append(f"{label}: transforms failed: {exc}")
            for list_path in _list_paths(dataset_cfg, path.parent):
                if not list_path.is_file():
                    problems.append(f"{label}: missing list {list_path}")
                    continue
                names = read_list(list_path)
                if not names:
                    problems.append(f"{label}: empty list {list_path}")
                data_root = getattr(dataset_cfg, "data_root", None)  # else given by --training-root
                if check_files and data_root:
                    root = pathlib.Path(data_root)
                    missing = [n for n in names if not (root / n).exists()]
                    if missing:
                        problems.append(f"{label}: {len(missing)} files missing under {root}, e.g. {missing[0]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=pathlib.Path)
    parser.add_argument("--check-files", action="store_true", help="Also stat every listed volume.")
    args = parser.parse_args()
    failed = 0
    for path in args.configs:
        try:
            problems = validate(path, args.check_files)
        except Exception:  # noqa: BLE001  # one broken config must not hide the others
            problems = ["config failed to load:\n" + traceback.format_exc(limit=3)]
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {path}")
        for problem in problems:
            print(f"     - {problem}")
        failed += bool(problems)
    print(f"\n{len(args.configs) - failed}/{len(args.configs)} configs valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
