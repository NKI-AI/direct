"""Engines for magnitude-conditioned synthesis within DIRECT.

Primary supervised engines:

1. **PhaseFromMagnitudeEngine** — ``magnitude → phase``, loss vs SENSE phase label.
2. **ComplexFromMagnitudeEngine** — ``magnitude → complex``, loss vs SENSE complex label.

Additional:

3. FactorizedSynthesisEngine / PhaseDiffusionSynthesisEngine / CycleConsistencySynthesisEngine
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput, Engine
from direct.exceptions import NonFiniteLossError
from direct.nn.synthesis.losses import (
    chordal_phase_loss,
    complex_supervision_losses,
    cosine_phase_loss,
    mean_angular_error_deg,
    phase_gradient_loss,
    phase_supervision_losses,
    snr_phase_weight,
)
from direct.nn.synthesis.synthesis import PhaseDiffusion, _weighted_mean
from direct.synthesis.physics import SynthesisOutput, synthesis_diagnostics, synthesize, unit_complex
from direct.types import FFTOperator
from direct.utils import (
    communication,
    detach_dict,
    dict_to_device,
    merge_list_of_dicts,
    normalize_image,
    reduce_list_of_dicts,
)
from direct.utils.communication import reduce_tensor_dict
from direct.utils.events import get_event_storage

# YAML ``validation.metrics`` / ``training.metrics`` names for synthesis engines.
_SYNTHESIS_METRICS: dict[str, Callable] = {
    "phase_mae_deg": mean_angular_error_deg,
    "phase_chordal": chordal_phase_loss,
    "phase_cosine": cosine_phase_loss,
    "phase_gradient": phase_gradient_loss,
}
_DEFAULT_SYNTHESIS_METRICS = ["phase_mae_deg", "phase_chordal", "phase_gradient"]


def _build_synthesis_metrics(metrics_list: list[str] | None) -> dict[str, Callable]:
    """Resolve YAML metric names to callables ``(pred, target, weight) -> scalar``."""
    names = list(metrics_list) if metrics_list else list(_DEFAULT_SYNTHESIS_METRICS)
    out: dict[str, Callable] = {}
    for name in names:
        key = name[:-7] if name.endswith("_metric") else name
        if key not in _SYNTHESIS_METRICS:
            allowed = ", ".join(sorted(_SYNTHESIS_METRICS))
            raise ValueError(f"Unknown synthesis metric '{name}'. Choose from: {allowed}.")
        out[f"{key}_metric"] = _SYNTHESIS_METRICS[key]
    return out


def _case_keys(data: dict[str, Any], batch_size: int) -> list[str]:
    filenames = data.get("filename", ["unknown"] * batch_size)
    if isinstance(filenames, str):
        filenames = [filenames] * batch_size
    slice_nos = data.get("slice_no", list(range(batch_size)))
    if torch.is_tensor(slice_nos):
        slice_nos = slice_nos.detach().cpu().tolist()
    keys: list[str] = []
    for idx in range(batch_size):
        filename = filenames[idx] if idx < len(filenames) else filenames[0]
        slice_no = slice_nos[idx] if idx < len(slice_nos) else idx
        keys.append(f"{pathlib.Path(str(filename)).name}_s{int(slice_no)}")
    return keys


def _per_case_phase_metrics(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    magnitude: torch.Tensor,
    support: torch.Tensor | None,
    data: dict[str, Any],
    metric_fns: dict[str, Callable],
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute YAML metrics per slice for the validation JSON / TensorBoard."""
    weight = snr_phase_weight(magnitude, support, power=2.0)
    cases: dict[str, dict[str, torch.Tensor]] = {}
    for idx, key in enumerate(_case_keys(data, pred_phase.shape[0])):
        cases[key] = {
            name: fn(pred_phase[idx : idx + 1], target_phase[idx : idx + 1], weight[idx : idx + 1]).detach()
            for name, fn in metric_fns.items()
        }
    return cases


def _mag_weighted_phase(phase: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Grayscale ``(1, H, W)``: wrapped phase × peak-normalized magnitude.

    Background air phase is random even for GT; masking by magnitude is what
    makes tissue structure visible in TensorBoard.
    """
    angle = torch.atan2(phase[..., 1], phase[..., 0])
    mag = magnitude[0] if magnitude.ndim == 3 else magnitude
    mag = mag / mag.amax().clamp_min(1e-8)
    return (((angle + torch.pi) / (2 * torch.pi)) * mag).unsqueeze(0)


def _phase_hsv_rgb(phase: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """RGB ``(3, H, W)``: hue = phase, value = magnitude (standard MRI phase view)."""
    hue = (torch.atan2(phase[..., 1], phase[..., 0]) + torch.pi) / (2 * torch.pi)
    mag = magnitude[0] if magnitude.ndim == 3 else magnitude
    value = mag / mag.amax().clamp_min(1e-8)
    sat = torch.full_like(value, 0.9)
    chroma = value * sat
    x = chroma * (1.0 - (hue * 6.0 % 2.0 - 1.0).abs())
    m = value - chroma
    h6 = hue * 6.0
    zeros = torch.zeros_like(hue)
    rgb = torch.zeros(3, *hue.shape, device=hue.device, dtype=hue.dtype)
    sectors = (
        ((h6 < 1), (chroma, x, zeros)),
        ((h6 >= 1) & (h6 < 2), (x, chroma, zeros)),
        ((h6 >= 2) & (h6 < 3), (zeros, chroma, x)),
        ((h6 >= 3) & (h6 < 4), (zeros, x, chroma)),
        ((h6 >= 4) & (h6 < 5), (x, zeros, chroma)),
        ((h6 >= 5), (chroma, zeros, x)),
    )
    for mask, channels in sectors:
        for c_idx, channel in enumerate(channels):
            rgb[c_idx] = torch.where(mask, channel, rgb[c_idx])
    return (rgb + m).clamp(0, 1)


def _support_weight(data: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
    """Return per-pixel weight ``(B, H, W)``, defaulting to ones."""
    if "weight" in data:
        weight = data["weight"]
        return weight[:, 0] if weight.ndim == 4 else weight
    return torch.ones(reference.shape[:3], device=reference.device, dtype=reference.dtype)


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return default
    return float(getattr(model_cfg, name, default))


def _tb_num_images(cfg: Any) -> int:
    logging_cfg = getattr(cfg, "logging", None)
    tb = getattr(logging_cfg, "tensorboard", None)
    return int(getattr(tb, "num_images", 4) or 4)


def _yaml_loss_weights(cfg: Any) -> dict[str, float]:
    """Parse ``training.loss.losses`` into ``{function_name: multiplier}``."""
    losses_cfg = getattr(getattr(cfg, "training", None), "loss", None)
    entries = getattr(losses_cfg, "losses", None) if losses_cfg is not None else None
    if not entries:
        return {}
    weights: dict[str, float] = {}
    for curr in entries:
        fn = curr.function
        name = fn.value if hasattr(fn, "value") else str(fn)
        weights[name] = float(getattr(curr, "multiplier", 1.0))
    return weights


def _batch_filename(data: dict[str, Any]) -> str:
    filenames = data.get("filename", ["unknown"])
    if isinstance(filenames, (list, tuple)):
        filenames = filenames[0]
    return pathlib.Path(str(filenames)).name


class SynthesisEngine(Engine):
    """Shared train / val / predict path for magnitude-conditioned synthesis.

    Validation matches the recon contract in :class:`~direct.engine.Engine`:
    a 6-tuple from :meth:`evaluate`, per-case keys with a ``_metric`` suffix,
    TensorBoard tags under ``val/{dataset}/prediction`` and ``train/target``,
    and a DDP gather of the metric dict. ``direct predict`` uses the same
    checkpoint loading and volume loop as recon, writing k-space H5 files.
    """

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        """Map one collated batch to :class:`SynthesisOutput`."""
        raise NotImplementedError

    def _coil_sensitivity_cfg(self):
        inference = getattr(self.cfg, "inference", None)
        return getattr(inference, "coil_sensitivity", None)

    def _require_maps(self, data: dict[str, Any]) -> torch.Tensor:
        maps = data.get("sensitivity_map")
        if maps is None:
            raise KeyError(
                "Generation needs `sensitivity_map` from the same ACS estimator as training. "
                "Use a fully sampled FastMRI volume with synthesis transforms, not a maps-free DICOM."
            )
        return maps

    def _maps_for_generation(self, data: dict[str, Any]) -> torch.Tensor:
        """Simulated coil maps, or ACS maps when ``inference.coil_sensitivity.mode`` is ``acs``."""
        from direct.synthesis.physics import simulated_maps

        cfg = self._coil_sensitivity_cfg()
        mode = str(getattr(cfg, "mode", "acs") or "acs").lower()
        if mode in {"acs", "none"}:
            return self._require_maps(data)

        magnitude = data["magnitude"]
        num_coils = getattr(cfg, "num_coils", None)
        if num_coils is None:
            acs = data.get("sensitivity_map")
            num_coils = int(acs.shape[1]) if acs is not None else 15
        seed = getattr(cfg, "seed", 0)
        if seed is None:
            seed = 0

        cache = getattr(self, "_simulated_maps_cache", None)
        if cache is None:
            self._simulated_maps_cache = {}
            cache = self._simulated_maps_cache
        cache_key = (
            mode,
            int(num_coils),
            tuple(int(v) for v in magnitude.shape[-2:]),
            int(seed),
            float(getattr(cfg, "coil_radius", 1.5)),
            float(getattr(cfg, "coil_size", 0.55)),
            float(getattr(cfg, "falloff_power", 1.5)),
            float(getattr(cfg, "phase_strength", 0.7)),
            float(getattr(cfg, "angular_jitter", 0.08)),
            int(getattr(cfg, "biot_savart_segments", 96)),
            bool(getattr(cfg, "normalize", True)),
        )
        kwargs = {
            "mode": mode,
            "coil_radius": float(getattr(cfg, "coil_radius", 1.5)),
            "coil_size": float(getattr(cfg, "coil_size", 0.55)),
            "falloff_power": float(getattr(cfg, "falloff_power", 1.5)),
            "phase_strength": float(getattr(cfg, "phase_strength", 0.7)),
            "angular_jitter": float(getattr(cfg, "angular_jitter", 0.08)),
            "biot_savart_segments": int(getattr(cfg, "biot_savart_segments", 96)),
            "normalize": bool(getattr(cfg, "normalize", True)),
        }
        if mode == "empirical":
            maps = simulated_maps(
                magnitude,
                int(num_coils),
                int(seed),
                reference_maps=self._require_maps(data)[0].detach().cpu(),
                **kwargs,
            )
            return maps

        if cache_key not in cache:
            unit = torch.zeros(1, 1, *magnitude.shape[-2:], dtype=torch.float32)
            cache[cache_key] = simulated_maps(unit, int(num_coils), int(seed), **kwargs)[0].cpu()
        maps = cache[cache_key].to(device=magnitude.device, dtype=torch.float32)
        return maps.expand(magnitude.shape[0], -1, -1, -1, -1).contiguous()

    def build_metrics(self, metrics_list) -> dict:
        """YAML ``metrics:`` names, with the recon ``_metric`` postfix."""
        return _build_synthesis_metrics(metrics_list)

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        """Predicted unit phase ``(B, H, W, 2)`` for validation metrics. Override per engine."""
        return None

    def _validation_visualization(
        self,
        data: dict[str, Any],
        pred_phase: torch.Tensor | None,
        iteration_output: DoIterationOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grayscale ``(1, H, W)`` pair for ``val/{dataset}/prediction`` vs ``target``."""
        mag = data["magnitude"][0, 0] if data["magnitude"].ndim == 4 else data["magnitude"][0]
        if pred_phase is not None and "phase" in data:
            return (
                _mag_weighted_phase(pred_phase[0], mag).cpu(),
                _mag_weighted_phase(data["phase"][0], mag).cpu(),
            )
        pred = iteration_output.output_image
        if pred is None:
            pred = mag
        pred_vis = pred[0].detach().cpu()
        if pred_vis.ndim == 2:
            pred_vis = pred_vis.unsqueeze(0)
        return pred_vis, mag.detach().cpu().unsqueeze(0)

    def _finalize_evaluation(
        self,
        val_losses: list[dict],
        val_metrics: dict,
        viz_slices: list[torch.Tensor],
        viz_target: list[torch.Tensor],
    ):
        loss_dict = reduce_list_of_dicts(val_losses) if val_losses else {}
        if loss_dict:
            reduce_tensor_dict(loss_dict)
        communication.synchronize()
        gathered = merge_list_of_dicts(communication.all_gather(dict(val_metrics)))
        return loss_dict, gathered, viz_slices, None, viz_target, None

    def log_first_training_example_and_model(self, data):
        """Log magnitude (as ``train/target``) and GT phase, matching recon TB names."""
        storage = get_event_storage()
        self.logger.info(
            "First case: slice_no: %s, filename: %s.",
            data.get("slice_no", ["?"])[0],
            data.get("filename", ["?"])[0],
        )
        if "magnitude" in data:
            mag = data["magnitude"][0]
            mag = mag[0] if mag.ndim == 3 else mag
            storage.add_image("train/target", normalize_image(mag.unsqueeze(0)))
        if "phase" in data:
            mag = data["magnitude"][0, 0] if data["magnitude"].ndim == 4 else data["magnitude"][0]
            storage.add_image("train/phase", _mag_weighted_phase(data["phase"][0], mag))
        self.write_to_logs()

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, loss_fns: dict[str, Callable] | None):
        """Validate like :class:`~direct.nn.mri_models.MRIModelEngine`."""
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()
        metric_fns = self.build_metrics(getattr(getattr(self.cfg, "validation", None), "metrics", None))
        num_images = _tb_num_images(self.cfg)
        val_losses: list[dict] = []
        val_metrics: dict[str, dict] = {}
        viz_slices: list[torch.Tensor] = []
        viz_target: list[torch.Tensor] = []
        for idx, data in enumerate(data_loader):
            data = dict_to_device(data, self.device)
            iteration_output = self._do_iteration(data, loss_fns)
            val_losses.append(detach_dict(iteration_output.data_dict))
            pred_phase = self._phase_from_batch(data)
            if pred_phase is not None and "phase" in data:
                cases = _per_case_phase_metrics(
                    pred_phase,
                    data["phase"],
                    data["magnitude"],
                    data.get("weight"),
                    data,
                    metric_fns,
                )
                val_metrics.update(cases)
                for filename, curr_metrics in cases.items():
                    curr_metrics_string = ", ".join(f"{x}: {float(y)}" for x, y in curr_metrics.items())
                    self.logger.info("Metrics for %s: %s", filename, curr_metrics_string)
            if len(viz_slices) < num_images:
                pred_vis, tgt_vis = self._validation_visualization(data, pred_phase, iteration_output)
                viz_slices.append(pred_vis)
                viz_target.append(tgt_vis)
            total = max(len(data_loader), 1)
            if total >= 10:
                self.log_process(idx, total)
        result = self._finalize_evaluation(val_losses, val_metrics, viz_slices, viz_target)
        torch.cuda.empty_cache()
        return result

    def _load_predict_checkpoint(self, experiment_directory: pathlib.Path, checkpoint: Any) -> None:
        from direct.checkpointer import Checkpointer

        self.checkpointer = Checkpointer(
            save_directory=experiment_directory,
            save_to_disk=False,
            model=self.model,  # type: ignore[attr-defined]
            **self.models,  # type: ignore[attr-defined]
        )
        if isinstance(checkpoint, int) or checkpoint in {"latest", None}:
            if self.checkpointer.checkpoint_loaded is not checkpoint:
                self.checkpointer.load(iteration=checkpoint, checkpointable_objects=None)
        else:
            self.checkpointer.load_models_from_file(checkpoint)

    @torch.no_grad()
    def reconstruct_volumes(self, data_loader, add_target: bool = False, **kwargs):
        """Assemble per-volume synthetic k-space for ``direct predict``."""
        self.models_to_device()  # type: ignore[attr-defined]
        self.models_validation_mode()  # type: ignore[attr-defined]
        torch.cuda.empty_cache()

        last_filename: str | None = None
        buffer: list[tuple[int, SynthesisOutput, torch.Tensor | None]] = []
        filenames_seen = 0
        time_start = time.time()

        def flush():
            nonlocal filenames_seen
            buffer.sort(key=lambda item: item[0])
            output = SynthesisOutput(
                kspace=torch.cat([item[1].kspace.cpu() for item in buffer], dim=0),
                sensitivity_map=torch.cat([item[1].sensitivity_map.cpu() for item in buffer], dim=0),
                phase=torch.cat([item[1].phase.cpu() for item in buffer], dim=0),
                magnitude=torch.cat([item[1].magnitude.cpu() for item in buffer], dim=0),
            )
            diagnostics = synthesis_diagnostics(output)
            metrics = {f"{key}_metric": value for key, value in diagnostics.items()}
            if buffer[0][2] is not None:
                gt_phase = torch.cat([item[2] for item in buffer if item[2] is not None], dim=0)
                weight = snr_phase_weight(output.magnitude, None, power=2.0)
                metrics["phase_mae_deg_metric"] = float(mean_angular_error_deg(output.phase, gt_phase, weight))
                metrics["phase_chordal_metric"] = float(chordal_phase_loss(output.phase, gt_phase, weight))
            payload = {
                "kspace": output.kspace,
                "sensitivity_map": output.sensitivity_map,
                "phase": output.phase,
                "magnitude": output.magnitude,
                "metadata": {
                    "engine": type(self).__name__,
                    "coil_sensitivity": {
                        "mode": str(getattr(self._coil_sensitivity_cfg(), "mode", "acs")),
                        "num_coils": int(output.sensitivity_map.shape[1]),
                    },
                },
            }
            filenames_seen += 1
            self.logger.info(
                "%i volumes reconstructed: %s (shape = %s) in %.3fs.",
                filenames_seen,
                last_filename,
                list(output.kspace.shape),
                time.time() - time_start,
            )
            if add_target:
                yield payload, output.magnitude, None, metrics, last_filename
            else:
                yield payload, None, metrics, last_filename

        for data in data_loader:
            data = dict_to_device(data, self.device)  # type: ignore[attr-defined]
            filename = _batch_filename(data)
            if last_filename is not None and filename != last_filename:
                yield from flush()
                buffer = []
            last_filename = filename
            output = self.synthesize_from_batch(data)
            slice_nos = data.get("slice_no", list(range(output.kspace.shape[0])))
            if torch.is_tensor(slice_nos):
                slice_nos = slice_nos.detach().cpu().tolist()
            gt = data["phase"].detach().cpu() if "phase" in data else None
            for idx in range(output.kspace.shape[0]):
                slice_no = int(slice_nos[idx]) if idx < len(slice_nos) else idx
                piece = SynthesisOutput(
                    kspace=output.kspace[idx : idx + 1].detach().cpu(),
                    sensitivity_map=output.sensitivity_map[idx : idx + 1].detach().cpu(),
                    phase=output.phase[idx : idx + 1].detach().cpu(),
                    magnitude=output.magnitude[idx : idx + 1].detach().cpu(),
                )
                buffer.append((slice_no, piece, None if gt is None else gt[idx : idx + 1]))
        if buffer:
            yield from flush()

    def predict(
        self,
        dataset,
        experiment_directory: pathlib.Path,
        checkpoint: int | str | pathlib.Path | None = -1,
        num_workers: int = 6,
        batch_size: int = 1,
        crop: str | None = None,
    ):
        """Load a checkpoint and generate synthetic k-space volumes."""
        self.logger.info("Predicting...")
        torch.cuda.empty_cache()
        self.ndim = getattr(dataset, "ndim", 2)
        self.logger.info("Data dimensionality: %s.", self.ndim)
        self.logger.info(
            "Generating synthetic multicoil k-space (coil maps: %s).",
            getattr(self._coil_sensitivity_cfg(), "mode", "acs"),
        )
        self._load_predict_checkpoint(experiment_directory, checkpoint)
        batch_sampler = self.build_batch_sampler(  # type: ignore[attr-defined]
            dataset, batch_size=batch_size, sampler_type="sequential", limit_number_of_volumes=None
        )
        data_loader = self.build_loader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)  # type: ignore[attr-defined]
        volumes = []
        metrics: dict[str, dict] = {}
        for payload, _mask, volume_metrics, filename in self.reconstruct_volumes(
            data_loader, add_target=False, crop=crop
        ):
            volumes.append((payload, None, filename))
            metrics[str(filename)] = {key: float(value) for key, value in volume_metrics.items()}
            self.logger.info(  # type: ignore[attr-defined]
                "Generated %s: rss_nmse=%.2e  phase_mae_deg=%s",
                filename,
                volume_metrics.get("rss_nmse_metric", float("nan")),
                f"{volume_metrics['phase_mae_deg_metric']:.1f}" if "phase_mae_deg_metric" in volume_metrics else "n/a",
            )
        return volumes, metrics


class PhaseFromMagnitudeEngine(SynthesisEngine):
    """Supervised phase prediction: magnitude → unit phase (chordal + gradient)."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator | None = None,
        backward_operator: FFTOperator | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        super().__init__(
            cfg, model, device,
            forward_operator=forward_operator or T.fft2,
            backward_operator=backward_operator or T.ifft2,
            mixed_precision=mixed_precision,
            **models,
        )
        self.ndim = 2

    def build_loss(self) -> dict:
        weights = _yaml_loss_weights(self.cfg)
        if "phase_chordal_loss" not in weights:
            weights["phase_chordal_loss"] = 1.0
        if "phase_gradient_loss" not in weights:
            weights["phase_gradient_loss"] = _cfg_float(self.cfg, "phase_gradient_weight", 0.1)
        return weights

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: dict[str, Callable] | None = None,
        regularizer_fns: dict[str, Callable] | None = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = self.build_loss()
        data = dict_to_device(data, self.device)
        grad_w = float(loss_fns.get("phase_gradient_loss", 0.0))
        chordal_w = float(loss_fns.get("phase_chordal_loss", 1.0))
        with autocast("cuda", enabled=self.mixed_precision):
            pred_phase = self.model(data["magnitude"])
            losses = phase_supervision_losses(
                pred_phase,
                data["phase"],
                data["magnitude"],
                data.get("weight"),
                gradient_weight=grad_w,
            )
            loss = chordal_w * losses["phase_loss"]
            if "phase_grad_loss" in losses:
                loss = loss + grad_w * losses["phase_grad_loss"]

        if self.model.training:
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"Non-finite phase loss: {loss}")
            self._scaler.scale(loss).backward()

        data_dict = {
            "phase_chordal_loss": losses["phase_loss"].detach(),
            "phase_mae_deg_metric": losses["phase_mae_deg"].detach(),
        }
        if "phase_grad_loss" in losses:
            data_dict["phase_gradient_loss"] = (grad_w * losses["phase_grad_loss"]).detach()

        return DoIterationOutput(
            output_image=data["magnitude"][:, 0],
            sensitivity_map=data.get("sensitivity_map"),
            data_dict=data_dict,
        )

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        return self.model(data["magnitude"])

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        phase = self.model(data["magnitude"])
        return synthesize(data["magnitude"], phase, self._maps_for_generation(data))


class ComplexFromMagnitudeEngine(SynthesisEngine):
    """Supervised complex prediction: magnitude → SENSE complex (ℂ L2 + mag + phase)."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator | None = None,
        backward_operator: FFTOperator | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        super().__init__(
            cfg, model, device,
            forward_operator=forward_operator or T.fft2,
            backward_operator=backward_operator or T.ifft2,
            mixed_precision=mixed_precision,
            **models,
        )
        self.ndim = 2

    def build_loss(self) -> dict:
        weights = _yaml_loss_weights(self.cfg)
        defaults = {
            "complex_l2_loss": _cfg_float(self.cfg, "complex_weight", 1.0),
            "mag_l2_loss": _cfg_float(self.cfg, "magnitude_weight", 1.0),
            "phase_chordal_loss": _cfg_float(self.cfg, "phase_weight", 1.0),
            "phase_gradient_loss": _cfg_float(self.cfg, "phase_gradient_weight", 0.1),
        }
        for key, value in defaults.items():
            weights.setdefault(key, value)
        return weights

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: dict[str, Callable] | None = None,
        regularizer_fns: dict[str, Callable] | None = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = self.build_loss()
        data = dict_to_device(data, self.device)
        cpx_w = float(loss_fns.get("complex_l2_loss", _cfg_float(self.cfg, "complex_weight", 1.0)))
        mag_w = float(loss_fns.get("mag_l2_loss", _cfg_float(self.cfg, "magnitude_weight", 1.0)))
        phase_w = float(loss_fns.get("phase_chordal_loss", _cfg_float(self.cfg, "phase_weight", 1.0)))
        grad_w = float(loss_fns.get("phase_gradient_loss", _cfg_float(self.cfg, "phase_gradient_weight", 0.1)))
        with autocast("cuda", enabled=self.mixed_precision):
            pred = self.model(data["magnitude"])
            losses = complex_supervision_losses(
                pred,
                data["complex_image"],
                data["magnitude"],
                data.get("weight"),
                phase_weight=phase_w,
                magnitude_weight=mag_w,
                complex_weight=cpx_w,
                gradient_weight=grad_w,
            )
            loss = losses["loss"]

        if self.model.training:
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"Non-finite complex loss: {loss}")
            self._scaler.scale(loss).backward()

        data_dict = {
            "complex_l2_loss": (cpx_w * losses["complex_loss"]).detach(),
            "mag_l2_loss": (mag_w * losses["mag_loss"]).detach(),
            "phase_chordal_loss": (phase_w * losses["phase_loss"]).detach(),
            "phase_mae_deg_metric": losses["phase_mae_deg"].detach(),
        }
        if "phase_grad_loss" in losses:
            data_dict["phase_gradient_loss"] = (grad_w * losses["phase_grad_loss"]).detach()

        pred_mag = torch.sqrt((pred**2).sum(-1))
        return DoIterationOutput(
            output_image=pred_mag.detach(),
            sensitivity_map=data.get("sensitivity_map"),
            data_dict=data_dict,
        )

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        return unit_complex(self.model(data["magnitude"]))

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        pred = self.model(data["magnitude"])
        return synthesize(data["magnitude"], unit_complex(pred), self._maps_for_generation(data))


class FactorizedSynthesisEngine(SynthesisEngine):
    """Engine for supervised factorized synthesis.

    Trains on calibrated pseudo-labels (magnitude, phase, sensitivity_map, weight)
    extracted from real fully sampled multicoil data. Loss = coil-image L2 +
    weighted phase L2 + weighted map L2.

    Compatible with DIRECT's ``training_loop``, ``validation_loop``, checkpointing,
    and TensorBoard logging.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator | None = None,
        backward_operator: FFTOperator | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Initialize the engine.

        Args:
            cfg: DIRECT configuration.
            model: :class:`FactorizedSynthesizer` instance.
            device: ``"cuda"`` or ``"cpu"``.
            forward_operator: FFT operator (default ``fft2``).
            backward_operator: iFFT operator (default ``ifft2``).
            mixed_precision: Enable AMP.
            **models: Additional models (unused).
        """
        super().__init__(
            cfg, model, device,
            forward_operator=forward_operator or T.fft2,
            backward_operator=backward_operator or T.ifft2,
            mixed_precision=mixed_precision,
            **models,
        )
        self.ndim = 2

    def build_loss(self) -> dict:
        return {"coil_loss": 1.0, "phase_loss": 0.25, "map_loss": 0.25}

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: dict[str, Callable] | None = None,
        regularizer_fns: dict[str, Callable] | None = None,
    ) -> DoIterationOutput:
        """Perform one training iteration.

        Args:
            data: Batch with ``magnitude``, ``phase``, ``sensitivity_map``, ``weight``.
            loss_fns: Loss functions from ``build_loss``.
            regularizer_fns: Regularizers (unused).

        Returns:
            :class:`DoIterationOutput` with synthesis diagnostics.
        """
        if loss_fns is None:
            loss_fns = {}
        data = dict_to_device(data, self.device)

        with autocast("cuda", enabled=self.mixed_precision):
            output = self.model(data["magnitude"])
            weight = snr_phase_weight(data["magnitude"], data.get("weight"), power=2.0)
            phase_loss = chordal_phase_loss(output.phase, data["phase"], weight)
            map_loss = _weighted_mean(
                (output.sensitivity_map - data["sensitivity_map"]).square().sum(-1).sum(1),
                weight,
            )
            predicted = T.complex_multiplication(output.sensitivity_map, output.phase[:, None])
            target = T.complex_multiplication(data["sensitivity_map"], data["phase"][:, None])
            # Coil-image error ↔ k-space MSE by Parseval; weight ∝ support (not m²)
            # so bright coils don't dominate exclusively.
            coil_weight = snr_phase_weight(data["magnitude"], data.get("weight"), power=1.0)
            coil_loss = _weighted_mean(
                (predicted - target).square().sum(-1).sum(1),
                coil_weight,
            )
            loss = coil_loss + 0.25 * (phase_loss + map_loss)

        if self.model.training:
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"Non-finite synthesis loss: {loss}")
            self._scaler.scale(loss).backward()

        loss_dict = {
            "synthesis_loss": loss.detach(),
            "phase_loss": phase_loss.detach(),
            "map_loss": map_loss.detach(),
            "coil_loss": coil_loss.detach(),
        }

        # RSS of synthesized data for visualization
        coils = T.ifft2(output.kspace, dim=(2, 3))
        rss = T.root_sum_of_squares(coils, dim=1)

        return DoIterationOutput(
            output_image=rss,
            sensitivity_map=output.sensitivity_map,
            data_dict=loss_dict,
        )

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        return self.model(data["magnitude"]).phase

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        return self.model(data["magnitude"])


class PhaseDiffusionSynthesisEngine(SynthesisEngine):
    """Engine for conditional phase diffusion synthesis.

    Trains a DDPM on unit-phase coordinates conditioned on magnitude images.
    Uses empirical sensitivity maps from calibration data.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator | None = None,
        backward_operator: FFTOperator | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Initialize the engine.

        Args:
            cfg: DIRECT configuration.
            model: :class:`PhaseDiffusion` instance.
            device: ``"cuda"`` or ``"cpu"``.
            forward_operator: FFT operator.
            backward_operator: iFFT operator.
            mixed_precision: Enable AMP.
            **models: Additional models.
        """
        super().__init__(
            cfg, model, device,
            forward_operator=forward_operator or T.fft2,
            backward_operator=backward_operator or T.ifft2,
            mixed_precision=mixed_precision,
            **models,
        )
        self.ndim = 2
        self._generator = torch.Generator(device=device).manual_seed(42)

    def build_loss(self) -> dict:
        """YAML ``training.loss`` multipliers; ``noise_loss`` defaults to 1.0."""
        weights = _yaml_loss_weights(self.cfg)
        if "noise_loss" not in weights:
            weights["noise_loss"] = 1.0
        return weights

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: dict[str, Callable] | None = None,
        regularizer_fns: dict[str, Callable] | None = None,
    ) -> DoIterationOutput:
        """Perform one training iteration with diffusion noise prediction.

        Args:
            data: Batch with ``magnitude``, ``phase``, ``weight``.
            loss_fns: Loss functions from ``build_loss``.
            regularizer_fns: Regularizers (unused).

        Returns:
            :class:`DoIterationOutput` with diffusion loss.
        """
        if loss_fns is None:
            loss_fns = self.build_loss()
        data = dict_to_device(data, self.device)
        model: PhaseDiffusion = self.model  # type: ignore

        # Get loss weights from build_loss
        noise_w = loss_fns.get("noise_loss", 1.0)
        chordal_w = loss_fns.get("phase_chordal_loss", 0.0)
        grad_w = loss_fns.get("phase_gradient_loss", 0.0)
        mae_w = loss_fns.get("phase_mae_loss", 0.0)

        with autocast("cuda", enabled=self.mixed_precision):
            clean = data["phase"].permute(0, 3, 1, 2)  # (B, 2, H, W)
            times = torch.randint(
                model.timesteps, (clean.shape[0],),
                device=clean.device, generator=self._generator,
            )
            noise = torch.randn(clean.shape, device=clean.device, generator=self._generator)
            alpha = model.alpha_bar[times, None, None, None]
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            prediction = model(noisy, data["magnitude"], times)

            # Noise prediction loss (core diffusion objective)
            weight = snr_phase_weight(data["magnitude"], data.get("weight"), power=2.0)
            noise_loss = _weighted_mean((prediction - noise).square().sum(1), 0.05 + weight)

            out_dict = {"noise_loss": noise_loss.detach()}
            total_loss = noise_w * noise_loss

            # Optional phase losses (require denoising estimate)
            if chordal_w > 0 or grad_w > 0 or mae_w > 0:
                x0_est = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                pred_phase = unit_complex(x0_est.permute(0, 2, 3, 1))
                target_phase = data["phase"]

                if chordal_w > 0:
                    chordal = chordal_phase_loss(pred_phase, target_phase, weight)
                    total_loss = total_loss + chordal_w * chordal
                    out_dict["phase_chordal_loss"] = chordal.detach()

                if grad_w > 0:
                    grad_loss = phase_gradient_loss(pred_phase, target_phase, weight)
                    total_loss = total_loss + grad_w * grad_loss
                    out_dict["phase_gradient_loss"] = grad_loss.detach()

                if mae_w > 0:
                    mae = mean_angular_error_deg(pred_phase, target_phase, weight) * (torch.pi / 180.0)
                    total_loss = total_loss + mae_w * mae
                    out_dict["phase_mae_loss"] = mae.detach()

        if self.model.training:
            if not torch.isfinite(total_loss):
                raise NonFiniteLossError(f"Non-finite diffusion loss: {total_loss}")
            self._scaler.scale(total_loss).backward()

        return DoIterationOutput(
            output_image=data["magnitude"][:, 0],
            sensitivity_map=None,
            data_dict=out_dict,
        )

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        model: PhaseDiffusion = self.model  # type: ignore
        gen = getattr(self, "_generator", None) or torch.Generator(device=self.device).manual_seed(42)
        eval_steps = int(getattr(model, "eval_timesteps", 25))
        return model.sample_phase(data["magnitude"], gen, num_steps=eval_steps)

    @torch.no_grad()
    def evaluate(self, data_loader, loss_fns=None):
        """DDIM-sample phase vs GT using the shared recon validation path."""
        eval_steps = int(getattr(self.model, "eval_timesteps", 25))
        metric_fns = self.build_metrics(getattr(getattr(self.cfg, "validation", None), "metrics", None))
        self.logger.info(
            "Validation: DDIM sample on full set (%d steps). Metrics: %s.",
            eval_steps,
            ", ".join(metric_fns) or "none",
        )
        self._generator = torch.Generator(device=self.device).manual_seed(42)
        return super().evaluate(data_loader, loss_fns)

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        model: PhaseDiffusion = self.model  # type: ignore
        gen = getattr(self, "_generator", None) or torch.Generator(device=self.device).manual_seed(42)
        eval_steps = int(getattr(model, "eval_timesteps", 25))
        return model.sample(data["magnitude"], self._maps_for_generation(data), gen, num_steps=eval_steps)


class PhaseFlowMatchingEngine(SynthesisEngine):
    """Engine for conditional flow matching on unit-phase coordinates.

    Learns a velocity field v(x_t, t | magnitude) via optimal transport CFM:
        - x_0 ~ N(0, I), x_1 = target phase
        - x_t = (1-t)*x_0 + t*x_1
        - v_target = x_1 - x_0
        - Loss = ||v_pred - v_target||^2

    Simpler than diffusion: no noise schedule, direct velocity regression.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator | None = None,
        backward_operator: FFTOperator | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        super().__init__(
            cfg, model, device,
            forward_operator=forward_operator or T.fft2,
            backward_operator=backward_operator or T.ifft2,
            mixed_precision=mixed_precision,
            **models,
        )
        self.ndim = 2
        self._generator = torch.Generator(device=device).manual_seed(42)

    def build_loss(self) -> dict:
        """YAML ``training.loss`` multipliers; ``velocity_loss`` defaults to 1.0."""
        weights = _yaml_loss_weights(self.cfg)
        if "velocity_loss" not in weights:
            weights["velocity_loss"] = 1.0
        return weights

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: dict[str, Callable] | None = None,
        regularizer_fns: dict[str, Callable] | None = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = self.build_loss()
        data = dict_to_device(data, self.device)

        velocity_w = loss_fns.get("velocity_loss", 1.0)
        chordal_w = loss_fns.get("phase_chordal_loss", 0.0)
        grad_w = loss_fns.get("phase_gradient_loss", 0.0)
        mae_w = loss_fns.get("phase_mae_loss", 0.0)

        with autocast("cuda", enabled=self.mixed_precision):
            x_1 = data["phase"].permute(0, 3, 1, 2)
            batch_size = x_1.shape[0]

            t = torch.rand(batch_size, device=x_1.device, generator=self._generator)
            x_0 = torch.randn(x_1.shape, device=x_1.device, generator=self._generator)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * x_1
            v_target = x_1 - x_0

            v_pred = self.model(x_t, data["magnitude"], t)

            weight = snr_phase_weight(data["magnitude"], data.get("weight"), power=2.0)
            velocity_loss = _weighted_mean((v_pred - v_target).square().sum(1), 0.05 + weight)

            out_dict = {"velocity_loss": velocity_loss.detach()}
            total_loss = velocity_w * velocity_loss

            # Optional phase losses (use x_t + v_pred as estimate at t=1)
            if chordal_w > 0 or grad_w > 0 or mae_w > 0:
                # Estimate x_1 from current prediction: x_1_est = x_t + (1-t)*v_pred
                x1_est = x_t + (1 - t_expand) * v_pred
                pred_phase = unit_complex(x1_est.permute(0, 2, 3, 1))
                target_phase = data["phase"]

                if chordal_w > 0:
                    chordal = chordal_phase_loss(pred_phase, target_phase, weight)
                    total_loss = total_loss + chordal_w * chordal
                    out_dict["phase_chordal_loss"] = chordal.detach()

                if grad_w > 0:
                    grad_loss = phase_gradient_loss(pred_phase, target_phase, weight)
                    total_loss = total_loss + grad_w * grad_loss
                    out_dict["phase_gradient_loss"] = grad_loss.detach()

                if mae_w > 0:
                    mae = mean_angular_error_deg(pred_phase, target_phase, weight) * (torch.pi / 180.0)
                    total_loss = total_loss + mae_w * mae
                    out_dict["phase_mae_loss"] = mae.detach()

        if self.model.training:
            if not torch.isfinite(total_loss):
                raise NonFiniteLossError(f"Non-finite flow loss: {total_loss}")
            self._scaler.scale(total_loss).backward()

        return DoIterationOutput(
            output_image=data["magnitude"][:, 0],
            sensitivity_map=None,
            data_dict=out_dict,
        )

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        from direct.nn.synthesis.synthesis import PhaseFlowMatching

        model: PhaseFlowMatching = self.model  # type: ignore
        gen = getattr(self, "_generator", None) or torch.Generator(device=self.device).manual_seed(42)
        eval_steps = int(getattr(model, "eval_steps", 10))
        maps = data.get("sensitivity_map")
        if maps is None:
            maps = self._maps_for_generation(data)
        return model.sample(data["magnitude"], maps, gen, num_steps=eval_steps).phase

    @torch.no_grad()
    def evaluate(self, data_loader, loss_fns=None):
        """ODE-sample phase vs GT using the shared recon validation path."""
        eval_steps = int(getattr(self.model, "eval_steps", 10))
        metric_fns = self.build_metrics(getattr(getattr(self.cfg, "validation", None), "metrics", None))
        self.logger.info(
            "Validation: ODE sample on full set (%d steps). Metrics: %s.",
            eval_steps,
            ", ".join(metric_fns) or "none",
        )
        self._generator = torch.Generator(device=self.device).manual_seed(42)
        return super().evaluate(data_loader, loss_fns)

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        from direct.nn.synthesis.synthesis import PhaseFlowMatching

        model: PhaseFlowMatching = self.model  # type: ignore
        gen = getattr(self, "_generator", None) or torch.Generator(device=self.device).manual_seed(42)
        eval_steps = int(getattr(model, "eval_steps", 10))
        return model.sample(data["magnitude"], self._maps_for_generation(data), gen, num_steps=eval_steps)


class CycleConsistencySynthesisEngine(SynthesisEngine):
    """Engine for cycle-consistency synthesis training.

    No multicoil calibration targets needed.  Training only requires magnitude
    images and a frozen pretrained reconstruction model.

    Training loop:
    1. Synthesizer predicts phase + maps from magnitude
    2. Forward model creates multicoil k-space (exact RSS preservation)
    3. Random undersampling mask applied
    4. Frozen reconstruction model reconstructs from undersampled k-space
    5. Loss = ||reconstructed - magnitude||² + phase smoothness regularization

    This allows training synthesis from any magnitude-only DICOM dataset,
    using the physical consistency constraint to learn realistic phase and
    sensitivity patterns.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator | None = None,
        backward_operator: FFTOperator | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Initialize the cycle-consistency engine.

        Args:
            cfg: DIRECT configuration with ``model.recon_weight`` and ``model.phase_weight``.
            model: :class:`CycleConsistencySynthesizer` instance.
            device: Device string.
            forward_operator: FFT operator.
            backward_operator: iFFT operator.
            mixed_precision: Enable AMP.
            **models: Additional models; expects ``reconstruction_model`` if configured.
        """
        super().__init__(
            cfg, model, device,
            forward_operator=forward_operator or T.fft2,
            backward_operator=backward_operator or T.ifft2,
            mixed_precision=mixed_precision,
            **models,
        )
        self.ndim = 2
        self._recon_weight = getattr(cfg.model, "recon_weight", 1.0)
        self._phase_weight = getattr(cfg.model, "phase_weight", 0.1)

        # Freeze reconstruction model if provided
        if "reconstruction_model" in self.models:
            for p in self.models["reconstruction_model"].parameters():
                p.requires_grad = False

    def build_loss(self) -> dict:
        return {"cycle_loss": 1.0, "phase_smooth_loss": 1.0}

    def _phase_smoothness(self, phase: torch.Tensor) -> torch.Tensor:
        """Total-variation smoothness penalty on phase coordinates.

        Args:
            phase: Unit complex phase ``(B, H, W, 2)``.

        Returns:
            Scalar smoothness loss.
        """
        dx = (phase[:, 1:, :, :] - phase[:, :-1, :, :]).square().sum(-1).mean()
        dy = (phase[:, :, 1:, :] - phase[:, :, :-1, :]).square().sum(-1).mean()
        return dx + dy

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: dict[str, Callable] | None = None,
        regularizer_fns: dict[str, Callable] | None = None,
    ) -> DoIterationOutput:
        """Cycle-consistency training iteration.

        Args:
            data: Batch with ``magnitude`` (and optionally ``sampling_mask``,
                  ``sensitivity_map`` for the reconstruction model).
            loss_fns: Loss functions.
            regularizer_fns: Regularizers.

        Returns:
            :class:`DoIterationOutput` with cycle and smoothness losses.
        """
        if loss_fns is None:
            loss_fns = {}
        data = dict_to_device(data, self.device)
        magnitude = data["magnitude"]

        with autocast("cuda", enabled=self.mixed_precision):
            # Step 1: Synthesize multicoil k-space
            synth_output = self.model(magnitude)

            # Step 2: Create undersampling mask
            if "sampling_mask" in data:
                mask = data["sampling_mask"]
            else:
                from direct.common.subsample import FastMRIRandomMaskFunc
                mask_func = FastMRIRandomMaskFunc(
                    accelerations=getattr(self.cfg.model, "accelerations", [4, 8]),
                    center_fractions=getattr(self.cfg.model, "center_fractions", [0.08, 0.04]),
                )
                kspace_shape = synth_output.kspace.shape[2:]  # (H, W)
                mask, _ = mask_func((*kspace_shape, 2))
                mask = mask[None, None].to(self.device)  # (1, 1, H, W, 1)

            # Step 3: Undersample
            masked_kspace = T.apply_mask(synth_output.kspace, mask, return_mask=False)

            # Step 4: Estimate sensitivity maps from ACS
            from direct.data.mri_transforms import EstimateSensitivityMapModule
            acs_mask = mask.clone()  # Use the center as ACS
            acs_kspace = T.apply_mask(synth_output.kspace, acs_mask, return_mask=False)
            estimated_maps = EstimateSensitivityMapModule()({"acs_kspace": acs_kspace})["sensitivity_map"]

            # Step 5: Reconstruct (frozen)
            if "reconstruction_model" in self.models:
                recon_model = self.models["reconstruction_model"]
                with torch.no_grad():
                    recon_kspace = recon_model(
                        masked_kspace=masked_kspace,
                        sampling_mask=mask,
                        sensitivity_map=estimated_maps,
                    )
                recon_image = T.root_sum_of_squares(
                    T.ifft2(recon_kspace, dim=(2, 3)), dim=1,
                )
            else:
                # Fallback: simple zero-filled reconstruction
                recon_image = T.root_sum_of_squares(
                    T.ifft2(masked_kspace, dim=(2, 3)), dim=1,
                )

            # Step 6: Cycle-consistency loss
            target_mag = magnitude[:, 0]  # (B, H, W)
            cycle_loss = self._recon_weight * F.l1_loss(recon_image, target_mag)

            # Phase smoothness regularization
            smooth_loss = self._phase_weight * self._phase_smoothness(synth_output.phase)

            loss = cycle_loss + smooth_loss

        if self.model.training:
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"Non-finite cycle loss: {loss}")
            self._scaler.scale(loss).backward()

        loss_dict = {
            "cycle_loss": cycle_loss.detach(),
            "phase_smooth_loss": smooth_loss.detach(),
            "total_loss": loss.detach(),
        }

        return DoIterationOutput(
            output_image=recon_image.detach(),
            sensitivity_map=synth_output.sensitivity_map,
            data_dict=loss_dict,
        )

    def _phase_from_batch(self, data: dict[str, Any]) -> torch.Tensor | None:
        return self.model(data["magnitude"]).phase

    def synthesize_from_batch(self, data: dict[str, Any]) -> SynthesisOutput:
        return self.model(data["magnitude"])
