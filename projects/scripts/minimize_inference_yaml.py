#!/usr/bin/env python3
"""Build minimal inference-only YAML configs (no training/validation).

Keeps ``model``, ``additional_models``, ``physics``, and ``inference``.
Derives ``inference`` from ``validation.datasets`` (pins one R+ACS pair).

Preferred for git / Hub (few files)::

    python projects/scripts/minimize_inference_yaml.py projects/e2e_ads_recon/*.yaml \\
      --out-dir projects/e2e_ads_recon --suffix _inference --commented-rates

That writes ``{stem}_inference.yaml``: active ``val-4x``, other rates commented
under ``masking``. Training YAMLs keep all rates under ``training``/``validation``.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml


def _strip_nulls(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            sv = _strip_nulls(v)
            if sv is None or sv == {} or sv == []:
                continue
            out[k] = sv
        return out
    if isinstance(obj, list):
        return [_strip_nulls(x) for x in obj]
    return obj


def _single_pair(values: Any) -> list | None:
    """Inference masking must be one acceleration (and matching ACS) pair.

    ``MaskFunc.choose_acceleration`` randomly draws from the configured lists
    (DISCRETE). Training configs may list many R / ACS values; released
    inference YAMLs must pin exactly one combination.
    """
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        return [values]
    if len(values) == 0:
        return None
    return [values[0]]


def _pin_masking_to_single_pair(masking: dict) -> dict:
    m = dict(masking or {})
    if "accelerations" in m:
        m["accelerations"] = _single_pair(m.get("accelerations"))
    if "center_fractions" in m:
        m["center_fractions"] = _single_pair(m.get("center_fractions"))
    # Drop train-only sampling knobs that do not belong in inference dumps.
    for k in ("val_accelerations", "val_center_fractions", "uniform_range"):
        m.pop(k, None)
    return m


def _minimal_transforms(transforms: dict) -> dict:
    """Keep only fields that matter for reproducible inference."""
    t = transforms or {}
    masking = _pin_masking_to_single_pair(t.get("masking") or {})
    # Support both nested (modern) and flat (legacy) schemas.
    cropping = t.get("cropping") or {}
    sens = t.get("sensitivity_map_estimation") or {}
    norm = t.get("normalization") or {}

    masking_out: dict[str, Any] = {
        "name": masking.get("name"),
        "accelerations": masking.get("accelerations"),
        "center_fractions": masking.get("center_fractions"),
    }
    # Phase-specific / dynamic LOUPE policies need per-frame ACS masks.
    if masking.get("mode") is not None:
        masking_out["mode"] = masking["mode"]

    out: dict[str, Any] = {
        "use_seed": bool(t.get("use_seed", True)),
        "delete_kspace": bool(t.get("delete_kspace", False)),
        "masking": masking_out,
        "cropping": {
            "crop": cropping.get("crop", t.get("crop")),
            "image_center_crop": bool(cropping.get("image_center_crop", t.get("image_center_crop", False))),
        },
        "sensitivity_map_estimation": {
            "estimate_sensitivity_maps": bool(
                sens.get("estimate_sensitivity_maps", t.get("estimate_sensitivity_maps", True))
            ),
        },
        "normalization": {
            "scaling_key": norm.get("scaling_key", t.get("scaling_key", "masked_kspace")),
            "scale_percentile": norm.get("scale_percentile", t.get("scale_percentile", 0.99)),
        },
    }
    # Preserve ADS / CMRxRecon flags when present on the source transforms.
    for k in ("use_acs_as_mask", "delete_acs_mask", "dynamic_mask", "padding_eps"):
        if k in t:
            out[k] = t[k]
    # init2 policies: equispaced init mask at one R, policy samples toward another.
    if t.get("target_acceleration") is not None:
        out["target_acceleration"] = t["target_acceleration"]
    # Joint recon+registration models need a reference frame at inference.
    reg = t.get("registration") or {}
    if reg.get("registration"):
        out["registration"] = {
            "registration": True,
            "registration_simulate_reference": reg.get("registration_simulate_reference", "FROM_KEY"),
            "registration_simulate_reference_from_key_index": int(
                reg.get("registration_simulate_reference_from_key_index", 0)
            ),
            "registration_estimate_displacement": bool(reg.get("registration_estimate_displacement", False)),
        }
    aug = t.get("augmentation") or {}
    if aug.get("pad") is not None:
        out["augmentation"] = {"pad": aug["pad"]}
    return _strip_nulls(out)


def _minimal_model(model: dict) -> dict:
    """Drop engine_name nulls and obvious unused sibling architectures when UNET."""
    m = copy.deepcopy(model)
    m.pop("engine_name", None)
    arch = str(m.get("image_model_architecture", m.get("model_name", ""))).upper()
    if "UNET" in arch or m.get("image_model_architecture") == "UNET":
        for prefix in ("image_resnet_", "image_didn_", "image_conv_"):
            for k in list(m):
                if k.startswith(prefix):
                    m.pop(k)
    # Drop null-ish optional modulation defaults when NONE / unused
    if m.get("conv_modulation") in (None, "NONE"):
        for k in (
            "conv_modulation",
            "aux_in_features",
            "auxiliary_features",
            "log_aux",
            "fc_hidden_features",
            "fc_groups",
            "fc_activation",
            "num_weights",
            "modulation_at_input",
            "image_unet_norm_type",
            "image_unet_adain_hidden_features",
        ):
            # Keep aux_in_features if explicitly set non-null in source and needed — VSharp default is 2
            if k == "aux_in_features" and m.get(k) is not None:
                continue
            if k in m and (m[k] is None or m[k] is False or m[k] == "NONE" or m[k] == "INSTANCE"):
                m.pop(k, None)
    # Always keep conv_out bias flags when True (pre-ModConv VSharp checkpoints expect them).
    if m.get("image_unet_conv_out_bias") is True:
        m["image_unet_conv_out_bias"] = True
    return _strip_nulls(m)


def _minimal_additional_models(additional: dict | None) -> dict | None:
    if not additional:
        return None
    out = {}
    for name, cfg in additional.items():
        c = copy.deepcopy(cfg)
        c.pop("engine_name", None)
        # Drop NONE modulation defaults on UNet sensitivity
        if c.get("modulation") in (None, "NONE"):
            for k in (
                "modulation",
                "aux_in_features",
                "fc_hidden_features",
                "fc_groups",
                "fc_activation",
                "num_weights",
            ):
                if k in c and (c[k] is None or c[k] == "NONE" or c[k] == 1 or c[k] == "SIGMOID"):
                    c.pop(k, None)
        out[name] = _strip_nulls(c)
    return out or None


def _rate_tag(ds: dict) -> str:
    """Map a validation dataset to a short rate tag (``4x``, ``6x``, …)."""
    text = str(ds.get("text_description") or "").lower()
    m = re.search(r"(\d+x)\b", text)
    if m:
        return m.group(1)
    acc = ((ds.get("transforms") or {}).get("masking") or {}).get("accelerations") or []
    if acc:
        a = float(acc[0])
        if abs(a - round(a)) < 1e-6:
            return f"{round(a)}x"
        # e.g. 4.0327 → keep one decimal if .0 else compact
        return f"{a:g}x"
    return "default"


def _pick_validation_dataset(val_datasets: list, hint: str | None = None) -> dict:
    """Pick a validation dataset that already has a single R / ACS pair."""
    hint_l = (hint or "").lower()
    ds0 = copy.deepcopy(val_datasets[0])
    # Prefer a validation entry matching filename hints like ``_5x`` / ``_10x``.
    for ds in val_datasets:
        text = str(ds.get("text_description") or "").lower()
        acc = ((ds.get("transforms") or {}).get("masking") or {}).get("accelerations") or []
        acc_strs = {str(int(a)) if float(a).is_integer() else str(a) for a in acc}
        if "_10x" in hint_l and ("10x" in text or "10" in acc_strs):
            return copy.deepcopy(ds)
        if "_5x" in hint_l and ("5x" in text or "5" in acc_strs) and "10" not in acc_strs:
            return copy.deepcopy(ds)
        if "_8x" in hint_l and ("8x" in text or "8" in acc_strs):
            return copy.deepcopy(ds)
        if "_6x" in hint_l and ("6x" in text or "6" in acc_strs):
            return copy.deepcopy(ds)
        if "_4x" in hint_l and ("4x" in text or "4" in acc_strs):
            return copy.deepcopy(ds)
    # Default: first *4x* validation entry when present (common eval default).
    if not any(tag in hint_l for tag in ("_10x", "_8x", "_6x", "_5x", "_4x")):
        for ds in val_datasets:
            text = str(ds.get("text_description") or "").lower()
            if "4x" in text or text.endswith("4") or text == "4x":
                return copy.deepcopy(ds)
    return ds0


def _inference_from_dataset(cfg: dict, ds0: dict) -> dict:
    """Build an inference block from one validation (or inference) dataset entry."""
    name = ds0.get("name")
    transforms = _minimal_transforms(ds0.get("transforms") or {})
    transforms["use_seed"] = True
    out_ds: dict[str, Any] = {"name": name, "transforms": transforms}
    # Preserve Calgary-Campinas outer-slice crop used in zoo validation metrics.
    if "crop_outer_slices" in ds0:
        out_ds["crop_outer_slices"] = ds0["crop_outer_slices"]
    for k in ("kspace_key", "kspace_context", "pass_attrs", "compute_mask", "extra_keys"):
        if ds0.get(k) is not None:
            out_ds[k] = ds0[k]
    return _strip_nulls(
        {
            "batch_size": 1,
            "crop": (cfg.get("validation") or {}).get("crop") or (cfg.get("inference") or {}).get("crop"),
            "dataset": out_ds,
        }
    )


def _inference_from_cfg(cfg: dict, hint: str | None = None) -> dict:
    """Build inference block with a single acceleration / ACS combination.

    Prefer ``validation.datasets`` (each entry is already one R+ACS pair) over
    ``inference`` placeholders or multi-accel training masks. ``MaskFunc``
    randomly samples from lists, so released inference YAMLs must pin one pair.
    """
    val_datasets = (cfg.get("validation") or {}).get("datasets") or []
    inf_block = cfg.get("inference") or {}
    inf_ds = (inf_block.get("dataset") or {}) if isinstance(inf_block, dict) else {}
    inf_mask = (inf_ds.get("transforms") or {}).get("masking") or {}
    inf_mask_name = inf_mask.get("name")
    inf_acc = inf_mask.get("accelerations") or []
    # Ignore placeholder / incomplete / multi-accel inference dumps.
    use_existing_inference = (
        bool(inf_ds)
        and inf_mask_name not in (None, "???", "???")
        and len(inf_acc) == 1
        and not val_datasets  # prefer validation when available
    )

    if use_existing_inference:
        inf = copy.deepcopy(inf_block)
        ds = inf["dataset"]
        for k in (
            "filenames_lists",
            "filenames_lists_root",
            "data_root",
            "filenames_filter",
            "regex_filter",
            "text_description",
        ):
            ds.pop(k, None)
        ds["transforms"] = _minimal_transforms(ds.get("transforms") or {})
        if "crop_outer_slices" not in ds and ds.get("name") == "CalgaryCampinas":
            ds["crop_outer_slices"] = True
        return _strip_nulls(
            {
                "batch_size": inf.get("batch_size", 1),
                "crop": inf.get("crop"),
                "dataset": ds,
            }
        )

    if not val_datasets:
        if not inf_ds:
            raise ValueError("Config has neither inference.dataset nor validation.datasets")
        # Last resort: inference block, but pin to a single R+ACS pair.
        inf = copy.deepcopy(inf_block)
        ds = inf["dataset"]
        for k in (
            "filenames_lists",
            "filenames_lists_root",
            "data_root",
            "filenames_filter",
            "regex_filter",
            "text_description",
        ):
            ds.pop(k, None)
        ds["transforms"] = _minimal_transforms(ds.get("transforms") or {})
        if "crop_outer_slices" not in ds and ds.get("name") == "CalgaryCampinas":
            ds["crop_outer_slices"] = True
        return _strip_nulls(
            {
                "batch_size": inf.get("batch_size", 1),
                "crop": inf.get("crop"),
                "dataset": ds,
            }
        )

    ds0 = _pick_validation_dataset(val_datasets, hint=hint)
    return _inference_from_dataset(cfg, ds0)


def minimize_config(cfg: dict, hint: str | None = None, *, dataset: dict | None = None) -> dict:
    physics = cfg.get("physics") or {"forward_operator": "fft2", "backward_operator": "ifft2"}
    physics = {k: physics[k] for k in ("forward_operator", "backward_operator") if k in physics}
    inference = _inference_from_dataset(cfg, dataset) if dataset is not None else _inference_from_cfg(cfg, hint=hint)
    out = {
        "model": _minimal_model(cfg["model"]),
        "additional_models": _minimal_additional_models(cfg.get("additional_models")),
        "physics": physics,
        "inference": inference,
    }
    return _strip_nulls(out)


def dump_pretty(cfg: dict, path: Path) -> None:
    """Write YAML with 4-space indentation (prettier-style)."""

    class IndentDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            cfg,
            Dumper=IndentDumper,
            default_flow_style=False,
            sort_keys=False,
            width=100,
            indent=4,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _dump_block(obj: dict, base_indent: int = 0) -> str:
    class IndentDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    text = yaml.dump(
        obj,
        Dumper=IndentDumper,
        default_flow_style=False,
        sort_keys=False,
        width=100,
        indent=4,
        allow_unicode=True,
    )
    if base_indent <= 0:
        return text
    pad = " " * base_indent
    return "".join((pad + line if line.strip() else line) for line in text.splitlines(True))


def dump_inference_with_commented_rates(cfg: dict, path: Path, *, hint: str | None = None) -> None:
    """Write one inference YAML: active val-4x, other validation rates commented under masking."""
    val_datasets = (cfg.get("validation") or {}).get("datasets") or []
    if not val_datasets:
        minimal = minimize_config(cfg, hint=hint)
        dump_pretty(minimal, path)
        return

    ordered = sorted(val_datasets, key=lambda d: (0 if _rate_tag(d) == "4x" else 1, _rate_tag(d)))
    active = next((d for d in ordered if _rate_tag(d) == "4x"), ordered[0])
    active_tag = _rate_tag(active)
    others = [d for d in ordered if _rate_tag(d) != active_tag]

    minimal = minimize_config(cfg, hint=hint or path.stem, dataset=active)
    body = _dump_block(minimal, 0)

    alt: list[str] = []
    for d in others:
        tag = _rate_tag(d)
        t = _minimal_transforms(d.get("transforms") or {})
        m = t.get("masking") or {}
        alt.append(f"                # --- {tag}: uncomment these; comment out the active lists above ---")
        alt.append(f"                # accelerations: {m.get('accelerations')}")
        alt.append(f"                # center_fractions: {m.get('center_fractions')}")
        if t.get("target_acceleration") is not None:
            alt.append(f"                # and set target_acceleration: {t.get('target_acceleration')}")

    if alt:
        m = re.search(r"(^                center_fractions:\n(?:                    - .+\n)+)", body, re.MULTILINE)
        if not m:
            raise RuntimeError(f"{path}: could not locate masking.center_fractions to inject comments")
        body = body[: m.end()] + "\n".join(alt) + "\n" + body[m.end() :]

    header = (
        f"# Inference-only config (active: {active_tag}).\n"
        f"# Other validation rates are commented under inference.dataset.transforms.masking.\n"
        f"# Pair with checkpoint {{stem}}.pt (same stem as the training YAML).\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")
    # Ensure still valid YAML after comment injection
    yaml.safe_load(path.read_text(encoding="utf-8"))


def checkpoint_stem_for_inference_yaml(yaml_stem: str) -> str:
    """Map ``vsharp_ads_1d_inference`` / ``vsharp_ads_1d_4x`` → ``vsharp_ads_1d``."""
    stem = re.sub(r"_inference$", "", yaml_stem)
    return re.sub(r"_(?:4|5|6|8|10)x$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--in-place", action="store_true", help="Overwrite inputs instead of writing to out-dir")
    parser.add_argument(
        "--per-validation",
        action="store_true",
        help="(Discouraged for git) Emit one YAML per validation rate ({stem}_{4x,6x,…}.yaml).",
    )
    parser.add_argument(
        "--also-default",
        action="store_true",
        help="With --per-validation, also write {stem}.yaml as a copy of the 4x (or first) rate.",
    )
    parser.add_argument(
        "--commented-rates",
        action="store_true",
        help="Emit one inference YAML with other validation rates commented under masking.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Output filename suffix before .yaml (e.g. '_inference' → {stem}_inference.yaml).",
    )
    args = parser.parse_args()
    for src in args.inputs:
        cfg = yaml.safe_load(src.read_text())
        if args.per_validation:
            val_datasets = (cfg.get("validation") or {}).get("datasets") or []
            if not val_datasets:
                raise ValueError(f"{src}: --per-validation requires validation.datasets")
            written: list[Path] = []
            default_dest: Path | None = None
            for ds in val_datasets:
                tag = _rate_tag(ds)
                minimal = minimize_config(cfg, hint=f"{src.stem}_{tag}", dataset=ds)
                dest = src if args.in_place else args.out_dir / f"{src.stem}_{tag}.yaml"
                dump_pretty(minimal, dest)
                written.append(dest)
                print(f"{src} -> {dest} ({sum(1 for _ in dest.open())} lines) [{tag}]")
                if default_dest is None and (tag == "4x" or len(written) == 1):
                    default_dest = dest
            if args.also_default and default_dest is not None and not args.in_place:
                alias = args.out_dir / f"{src.stem}.yaml"
                alias.write_text(default_dest.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"{src} -> {alias} (alias of {default_dest.name})")
        else:
            out_name = f"{src.stem}{args.suffix}.yaml"
            dest = src if args.in_place else args.out_dir / out_name
            if args.commented_rates:
                dump_inference_with_commented_rates(cfg, dest, hint=src.stem)
            else:
                minimal = minimize_config(cfg, hint=src.stem)
                dump_pretty(minimal, dest)
            print(f"{src} -> {dest} ({sum(1 for _ in dest.open())} lines)")


if __name__ == "__main__":
    main()
