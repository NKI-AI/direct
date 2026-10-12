# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthesize multicoil k-space from magnitude with phase + sensitivity models.

Pipeline:
  magnitude ``m`` (+ organ, coil count ``N``)
    → ``SensitivityMapSynthesizer`` → complex maps ``S`` with ``∑_c |S_c|² = 1``
    → phase model ``.sample`` → unit phase ``φ``
    → ``k_c = FFT(S_c m e^{iφ})``

Input may be FastMRI k-space (transforms extract RSS magnitude + FULL_RSS labels)
or magnitude-only data once the dataset/transform path supports it.
"""

from __future__ import annotations

import argparse
import logging
import math
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

import direct.data.transforms as T
from direct.checkpointer import Checkpointer, load_ema_weights
from direct.common.subsample import build_masking_function
from direct.data.datasets import build_dataset_from_input
from direct.environment import load_model_from_name, setup_inference_environment
from direct.nn.synthesis.synthesis import organ_id_from_filename
from direct.synthesis.data import export_h5
from direct.synthesis.physics import SynthesisOutput
from direct.train import build_transforms_from_environment
from direct.types import FileOrUrl, PathOrString
from direct.utils import dict_to_device, filter_arguments_by_signature, set_all_seeds
from direct.utils.io import read_list

logger = logging.getLogger(__name__)


def _as_path(value: PathOrString | FileOrUrl) -> pathlib.Path:
    return pathlib.Path(str(value))


def _build_model(cfg_block: DictConfig | dict[str, Any], device: str) -> nn.Module:
    cfg = OmegaConf.create(cfg_block)
    cls = load_model_from_name(cfg.model_name)
    kwargs = {k: v for k, v in dict(cfg).items() if k not in {"model_name", "engine_name"}}
    kwargs["forward_operator"] = T.fft2
    kwargs["backward_operator"] = T.ifft2
    model = cls(**filter_arguments_by_signature(cls, kwargs, warn_dropped=True))
    return model.to(device)


def _load_weights(model: nn.Module, checkpoint: PathOrString | FileOrUrl, device: str) -> None:
    path = _as_path(checkpoint)
    checkpointer = Checkpointer(save_directory=path.parent, save_to_disk=False, model=model)
    checkpointer.load_models_from_file(path if path.is_file() else checkpoint)
    load_ema_weights(model, path)  # prefer EMA weights when the run kept them
    model.to(device)
    model.eval()


def _coil_images(magnitude: torch.Tensor, phase: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    image = magnitude[:, 0, ..., None] * phase
    return T.expand_operator(image, maps, dim=1)


def _log_kspace(kspace: torch.Tensor) -> np.ndarray:
    return torch.log1p(torch.linalg.vector_norm(kspace, dim=-1).mean(0)).detach().cpu().numpy()


def _mag_weighted_phase(phase: torch.Tensor, magnitude: torch.Tensor) -> np.ndarray:
    angle = torch.atan2(phase[..., 1], phase[..., 0]).detach().cpu().numpy()
    mag = magnitude.detach().cpu().numpy()
    mag = mag / max(float(np.percentile(mag, 99)), 1e-8)
    return np.where(mag > 0.05, angle, np.nan)


def make_comparison_figure(
    *,
    filename: str,
    slice_no: int,
    magnitude: torch.Tensor,
    gt_phase: torch.Tensor | None,
    pred_phase: torch.Tensor,
    gt_maps: torch.Tensor | None,
    pred_maps: torch.Tensor,
    pred_kspace: torch.Tensor,
    mask: torch.Tensor | None,
):
    """Build GT-vs-synth comparison figure (magnitude, phase, coils, maps, ZF).

    Returns:
        ``(main_fig, undersampled_fig_or_None)``. Caller closes figures.
    """
    import matplotlib.pyplot as plt

    mag = magnitude.detach().cpu().float()
    pred_coils = _coil_images(magnitude[None, None], pred_phase[None], pred_maps[None])[0]
    pred_rss = T.root_sum_of_squares(pred_coils, dim=0)
    n_show = min(4, pred_maps.shape[0])
    has_gt = gt_maps is not None

    rows = 3 + int(has_gt)
    cols = 2 + n_show
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.4 * rows), squeeze=False)

    def _im(ax, arr, title, cmap="gray", vmin=None, vmax=None):
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().float().numpy()
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    row = 0
    _im(axes[row, 0], mag, "magnitude m")
    _im(axes[row, 1], _mag_weighted_phase(pred_phase, mag), "synth φ", cmap="twilight", vmin=-math.pi, vmax=math.pi)
    if gt_phase is not None:
        _im(axes[row, 2], _mag_weighted_phase(gt_phase, mag), "GT φ", cmap="twilight", vmin=-math.pi, vmax=math.pi)
    else:
        axes[row, 2].axis("off")
    _im(axes[row, min(3, cols - 1)], _log_kspace(pred_kspace), "synth |k| log")
    for c in range(cols):
        if not axes[row, c].images:
            axes[row, c].axis("off")

    row = 1
    _im(axes[row, 0], torch.linalg.vector_norm(pred_coils[0], dim=-1), "synth |coil0|")
    _im(axes[row, 1], pred_rss, "synth RSS")
    for i in range(n_show):
        _im(axes[row, 2 + i], torch.linalg.vector_norm(pred_maps[i], dim=-1), f"synth |S{i}|", cmap="magma")

    if has_gt:
        row = 2
        gt_phase_use = gt_phase if gt_phase is not None else pred_phase
        gt_coils = _coil_images(magnitude[None, None], gt_phase_use[None], gt_maps[None])[0]
        gt_rss = T.root_sum_of_squares(gt_coils, dim=0)
        _im(axes[row, 0], torch.linalg.vector_norm(gt_coils[0], dim=-1), "GT |coil0|")
        _im(axes[row, 1], gt_rss, "GT RSS")
        for i in range(n_show):
            _im(axes[row, 2 + i], torch.linalg.vector_norm(gt_maps[i], dim=-1), f"GT |S{i}|", cmap="magma")

    row = rows - 1
    for i in range(n_show):
        _im(axes[row, 2 + i], torch.linalg.vector_norm(pred_coils[i], dim=-1), f"synth |coil{i}|")
    _im(axes[row, 0], pred_rss, "synth RSS")
    if mask is not None:
        masked = pred_kspace * mask.to(pred_kspace.device).to(pred_kspace.dtype)
        if masked.ndim == 4:  # (C,H,W,2)
            zf = T.root_sum_of_squares(T.ifft2(masked[None], dim=(2, 3))[0], dim=0)
        else:
            zf = pred_rss
        _im(axes[row, 1], zf, "synth ZF × mask")
    else:
        axes[row, 1].axis("off")

    fig.suptitle(
        f"{filename}  slice={slice_no}  N={pred_maps.shape[0]}  maps complex (H,W,2)",
        fontsize=11,
    )
    fig.tight_layout()

    fig2 = None
    if mask is not None:
        fig2, ax2 = plt.subplots(1, 3, figsize=(9.5, 3.1))
        mshow = mask[..., 0] if mask.ndim == 3 else mask
        ax2[0].imshow(mshow.detach().cpu().numpy(), cmap="gray")
        ax2[0].set_title("sampling mask")
        ax2[0].axis("off")
        ax2[1].imshow(pred_rss.detach().cpu().numpy(), cmap="gray")
        ax2[1].set_title("synth RSS (full)")
        ax2[1].axis("off")
        masked = pred_kspace * mask.to(pred_kspace.dtype)
        zf = T.root_sum_of_squares(T.ifft2(masked[None], dim=(2, 3))[0], dim=0)
        ax2[2].imshow(zf.detach().cpu().numpy(), cmap="gray")
        ax2[2].set_title("synth zero-filled")
        ax2[2].axis("off")
        fig2.suptitle(f"{filename} slice {slice_no}", fontsize=10)
        fig2.tight_layout()

    return fig, fig2


def _save_comparison_figure(
    path: pathlib.Path,
    *,
    filename: str,
    slice_no: int,
    magnitude: torch.Tensor,
    gt_phase: torch.Tensor | None,
    pred_phase: torch.Tensor,
    gt_maps: torch.Tensor | None,
    pred_maps: torch.Tensor,
    pred_kspace: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, fig2 = make_comparison_figure(
        filename=filename,
        slice_no=slice_no,
        magnitude=magnitude,
        gt_phase=gt_phase,
        pred_phase=pred_phase,
        gt_maps=gt_maps,
        pred_maps=pred_maps,
        pred_kspace=pred_kspace,
        mask=mask,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if fig2 is not None:
        fig2.savefig(path.with_name(path.stem + "_undersampled.png"), dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
    logger.info("Wrote comparison figure %s", path)


def _resolve_num_coils(cfg: DictConfig, sample: dict[str, Any]) -> int:
    synth = cfg.synthesize
    configured = getattr(synth, "num_coils", None)
    if configured is not None:
        return int(configured)
    maps = sample.get("sensitivity_map")
    if maps is None:
        raise ValueError("Set synthesize.num_coils or provide sensitivity_map labels in the batch.")
    return int(maps.shape[0] if maps.ndim == 4 else maps.shape[1])


@torch.no_grad()
def synthesize_volume(
    phase_model: nn.Module,
    sens_model: nn.Module,
    batch: dict[str, Any],
    *,
    num_coils: int,
    num_steps: int | None,
    seed: int,
    device: str,
) -> SynthesisOutput:
    magnitude = batch["magnitude"]
    if magnitude.ndim == 3:
        magnitude = magnitude.unsqueeze(1)
    filenames = batch.get("filename", ["unknown"] * magnitude.shape[0])
    if isinstance(filenames, (str, bytes, pathlib.Path)):
        filenames = [filenames]
    organ_ids = torch.tensor(
        [organ_id_from_filename(name) for name in filenames],
        device=device,
        dtype=torch.long,
    )
    maps = sens_model(magnitude, num_coils, organ_id=organ_ids)
    if maps.shape[-1] != 2:
        raise RuntimeError(f"Expected complex maps (..., 2), got shape {tuple(maps.shape)}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    sample_fn = phase_model.sample
    kwargs: dict[str, Any] = {
        "magnitude": magnitude,
        "maps": maps,
        "generator": generator,
    }
    if num_steps is not None:
        kwargs["num_steps"] = int(num_steps)
    try:
        return sample_fn(**kwargs, organ_id=organ_ids)
    except TypeError:
        return sample_fn(**kwargs)


def run_synthesize(
    *,
    output_directory: pathlib.Path,
    data_root: pathlib.Path,
    cfg_file: PathOrString | FileOrUrl,
    phase_checkpoint: PathOrString | FileOrUrl,
    sensitivity_checkpoint: PathOrString | FileOrUrl,
    experiment_directory: pathlib.Path | None,
    filenames_filter: pathlib.Path | None,
    device: str,
    num_workers: int,
    name: str,
    seed: int,
    mixed_precision: bool = False,
    debug: bool = False,
) -> None:
    set_all_seeds(seed)
    torch.set_num_threads(1)
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    experiment_directory = pathlib.Path(experiment_directory or output_directory)
    experiment_directory.mkdir(parents=True, exist_ok=True)

    env = setup_inference_environment(
        name,
        experiment_directory,
        device,
        machine_rank=0,
        mixed_precision=mixed_precision,
        cfg_file=cfg_file,
        debug=debug,
    )
    cfg = env.cfg
    if not hasattr(cfg, "synthesize") or cfg.synthesize is None:
        raise ValueError("Config must define a `synthesize:` block.")
    sens_cfg = None
    if hasattr(cfg, "additional_models") and cfg.additional_models is not None:
        sens_cfg = cfg.additional_models.get("sensitivity_model")
    if sens_cfg is None and hasattr(cfg, "sensitivity_model"):
        sens_cfg = cfg.sensitivity_model
    if sens_cfg is None:
        raise ValueError("Config must define `additional_models.sensitivity_model`.")

    phase_model = _build_model(cfg.model, device)
    sens_model = _build_model(sens_cfg, device)
    _load_weights(phase_model, phase_checkpoint, device)
    _load_weights(sens_model, sensitivity_checkpoint, device)
    logger.info("phase=%s  sensitivity=%s  device=%s", type(phase_model).__name__, type(sens_model).__name__, device)

    if hasattr(cfg, "inference") and getattr(cfg.inference, "dataset", None) is not None:
        dataset_cfg = cfg.inference.dataset
        batch_size = int(getattr(cfg.inference, "batch_size", 1) or 1)
    else:
        raise ValueError("Config needs inference.dataset.")

    transforms = build_transforms_from_environment(env, dataset_cfg)

    names = None
    if filenames_filter is not None:
        names = [pathlib.Path(x) for x in read_list(filenames_filter)]
        # Prefer absolute under data_root when list entries are bare filenames.
        names = [(_as_path(data_root) / n) if not n.is_absolute() else n for n in names]

    dataset = build_dataset_from_input(
        transforms=transforms,
        dataset_config=dataset_cfg,
        data_root=_as_path(data_root),
        filenames_filter=names,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    synth_cfg = cfg.synthesize
    max_volumes = int(getattr(synth_cfg, "max_volumes", 0) or 0)
    slices_per_volume = int(getattr(synth_cfg, "slices_per_volume", 3) or 3)
    num_steps = getattr(synth_cfg, "num_integration_steps", None)
    plot_accel = float(getattr(synth_cfg, "plot_acceleration", 4))
    plot_cf = float(getattr(synth_cfg, "plot_center_fraction", 0.08))
    mask_func = build_masking_function(
        name="FastMRIRandom",
        accelerations=[plot_accel],
        center_fractions=[plot_cf],
    )
    plot_dir = output_directory / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[tuple[int, SynthesisOutput, dict[str, Any]]]] = defaultdict(list)
    plotted: dict[str, int] = defaultdict(int)
    seen_order: list[str] = []

    for batch in loader:
        batch = dict_to_device(batch, device)
        if batch["magnitude"].ndim == 3:
            batch["magnitude"] = batch["magnitude"].unsqueeze(1)
        filenames = batch.get("filename")
        if isinstance(filenames, (str, bytes, pathlib.Path)):
            filenames = [filenames]
        stem = pathlib.Path(str(filenames[0])).name
        if stem not in buffers:
            if max_volumes and len(seen_order) >= max_volumes:
                break
            seen_order.append(stem)

        num_coils = _resolve_num_coils(cfg, {"sensitivity_map": batch.get("sensitivity_map")})
        # When batched, sensitivity_map is (B,C,H,W,2)
        if "sensitivity_map" in batch and batch["sensitivity_map"].ndim == 5:
            num_coils = (
                int(batch["sensitivity_map"].shape[1]) if getattr(synth_cfg, "num_coils", None) is None else num_coils
            )

        output = synthesize_volume(
            phase_model,
            sens_model,
            batch,
            num_coils=num_coils,
            num_steps=num_steps,
            seed=seed,
            device=device,
        )
        slice_nos = batch.get("slice_no", list(range(output.kspace.shape[0])))
        if torch.is_tensor(slice_nos):
            slice_nos = slice_nos.detach().cpu().tolist()

        for i in range(output.kspace.shape[0]):
            piece = SynthesisOutput(
                kspace=output.kspace[i : i + 1].detach().cpu(),
                sensitivity_map=output.sensitivity_map[i : i + 1].detach().cpu(),
                phase=output.phase[i : i + 1].detach().cpu(),
                magnitude=output.magnitude[i : i + 1].detach().cpu(),
            )
            meta = {
                "gt_phase": batch["phase"][i].detach().cpu() if "phase" in batch else None,
                "gt_maps": batch["sensitivity_map"][i].detach().cpu() if "sensitivity_map" in batch else None,
            }
            buffers[stem].append((int(slice_nos[i]), piece, meta))

            if plotted[stem] < slices_per_volume:
                h, w = piece.kspace.shape[-3:-1]
                # MaskFunc needs >=3 dims; trailing complex dim matches k-space layout.
                sampling_mask = mask_func((h, w, 2), seed=seed + plotted[stem], return_acs=False)
                if isinstance(sampling_mask, tuple):
                    sampling_mask = sampling_mask[0]
                # (1, H, W, 1) → (H, W, 1)
                if sampling_mask.ndim == 4 and sampling_mask.shape[0] == 1:
                    sampling_mask = sampling_mask[0]
                coil_mask = sampling_mask.cpu().float()
                _save_comparison_figure(
                    plot_dir / f"{pathlib.Path(stem).stem}_slice{int(slice_nos[i]):03d}.png",
                    filename=stem,
                    slice_no=int(slice_nos[i]),
                    magnitude=piece.magnitude[0, 0],
                    gt_phase=meta["gt_phase"],
                    pred_phase=piece.phase[0],
                    gt_maps=meta["gt_maps"],
                    pred_maps=piece.sensitivity_map[0],
                    pred_kspace=piece.kspace[0],
                    mask=coil_mask,
                )
                plotted[stem] += 1

    written = 0
    for stem in seen_order:
        items = buffers[stem]
        items.sort(key=lambda x: x[0])
        volume = SynthesisOutput(
            kspace=torch.cat([x[1].kspace for x in items], dim=0),
            sensitivity_map=torch.cat([x[1].sensitivity_map for x in items], dim=0),
            phase=torch.cat([x[1].phase for x in items], dim=0),
            magnitude=torch.cat([x[1].magnitude for x in items], dim=0),
        )
        if volume.sensitivity_map.shape[-1] != 2:
            raise RuntimeError("Sensitivity maps must be complex with last dim 2.")
        out_path = output_directory / stem
        if out_path.exists():
            out_path.unlink()
        export_h5(
            out_path,
            volume,
            {
                "engine": "direct.synthesize",
                "phase_checkpoint": str(phase_checkpoint),
                "sensitivity_checkpoint": str(sensitivity_checkpoint),
                "num_coils": int(volume.sensitivity_map.shape[1]),
                "complex_maps": True,
            },
        )
        logger.info(
            "Wrote %s  k=%s  maps=%s (complex)",
            out_path.name,
            tuple(volume.kspace.shape),
            tuple(volume.sensitivity_map.shape),
        )
        written += 1

    logger.info("Synthesize finished: %d volumes → %s", written, output_directory)


def synthesize_from_argparse(args: argparse.Namespace) -> None:
    run_synthesize(
        output_directory=args.output_directory,
        data_root=args.data_root,
        cfg_file=args.cfg_file,
        phase_checkpoint=args.phase_model,
        sensitivity_checkpoint=args.sensitivity_model,
        experiment_directory=args.experiment_directory,
        filenames_filter=args.filenames_filter,
        device=args.device,
        num_workers=args.num_workers,
        name=args.name or "synthesize",
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        debug=args.debug,
    )
