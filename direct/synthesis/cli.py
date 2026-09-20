"""Prepare, train and export magnitude-conditioned MRI synthesis.

Run ``python -m direct.synthesis.cli --help`` from an installed DIRECT checkout.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from direct import __version__
from direct.data import transforms as T
from direct.synthesis.data import PreparedDataset, export_h5, new_h5_file, read_dicom, validate_splits
from direct.synthesis.models import FactorizedSynthesizer, PhaseDiffusion
from direct.synthesis.physics import (
    calibration_targets,
    simulated_maps,
    smooth_phase,
    synthesis_diagnostics,
    synthesize,
)

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _prepare(args: argparse.Namespace) -> None:
    if not args.fully_sampled:
        raise ValueError(
            "Preparation requires --fully-sampled to declare that the input contains complete acquisitions."
        )
    records = json.loads(args.manifest.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty list of path, subject_id, split, domain records.")
    subjects: dict[str, str] = {}
    sources: set[Path] = set()
    for record in records:
        if set(record) != {"path", "subject_id", "split", "domain"}:
            raise ValueError("Each record must contain exactly path, subject_id, split, and domain.")
        if not all(isinstance(value, str) and value.strip() for value in record.values()):
            raise ValueError("All manifest values must be nonempty strings.")
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError("Manifest splits must be train, validation or test.")
        subject, split = record["subject_id"], record["split"]
        if subject in subjects and subjects[subject] != split:
            raise ValueError("A subject occurs in multiple splits.")
        subjects[subject] = split
        path = Path(record["path"])
        path = path if path.is_absolute() else args.manifest.parent / path
        path = path.resolve(strict=True)
        if path in sources:
            raise ValueError("The same source file occurs more than once in the manifest.")
        sources.add(path)
        record["path"] = str(path)
    if not 0 < args.max_projection_nmse <= 1:
        raise ValueError("max_projection_nmse must be in (0, 1].")
    args.output.mkdir(parents=True, exist_ok=False)
    for index, record in enumerate(records):
        source = Path(record["path"])
        destination = args.output / record["split"] / f"volume_{index:06d}.h5"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source, "r") as raw:
            if "kspace" not in raw or not np.issubdtype(raw["kspace"].dtype, np.complexfloating):
                raise ValueError(f"Expected native complex kspace (slices, coils, H, W): {source}")
            if raw["kspace"].ndim != 4 or raw["kspace"].shape[0] < 1:
                raise ValueError(f"Expected nonempty 2D multicoil volumes: {source}")
            if raw.attrs.get("synthetic", False) or ("mask" in raw and not np.asarray(raw["mask"]).all()):
                raise ValueError("Synthetic or explicitly undersampled data cannot serve as full calibration targets.")
            kept = 0
            with new_h5_file(destination) as prepared:
                prepared.attrs.update(
                    {
                        "prepared_synthesis_version": 1,
                        "subject_id": record["subject_id"],
                        "domain": record["domain"],
                        "acs_size": args.acs_size,
                        "source_sha256": _sha256(source),
                        "direct_version": __version__,
                    }
                )
                for slice_index in range(raw["kspace"].shape[0]):
                    kspace = T.to_tensor(np.asarray(raw["kspace"][slice_index], dtype=np.complex64))[None]
                    target = calibration_targets(
                        kspace, args.acs_size,
                        crop_readout=not args.no_crop_readout,
                    )
                    error = float(target["projection_nmse"][0])
                    if target["magnitude"].max() <= 0 or error > args.max_projection_nmse:
                        logger.warning("Rejected volume %d slice %d; projection NMSE %.6f", index, slice_index, error)
                        continue
                    for name, value in target.items():
                        array = value.numpy()
                        if name not in prepared:
                            prepared.create_dataset(
                                name,
                                shape=(0, *array.shape[1:]),
                                maxshape=(None, *array.shape[1:]),
                                dtype=np.float32,
                                chunks=True,
                                compression="gzip",
                            )
                        prepared[name].resize(kept + 1, axis=0)
                        prepared[name][kept] = array[0]
                    if "source_slice" not in prepared:
                        prepared.create_dataset("source_slice", shape=(0,), maxshape=(None,), dtype=np.int64)
                    prepared["source_slice"].resize(kept + 1, axis=0)
                    prepared["source_slice"][kept] = slice_index
                    kept += 1
            if not kept:
                destination.unlink()
                raise ValueError(
                    f"No usable slices in {source}; inspect calibration error before relaxing the threshold."
                )
        logger.info("Prepared %d slices in %s", kept, destination)


def _make_model(method: str, config: dict) -> FactorizedSynthesizer | PhaseDiffusion:
    if method == "factorized":
        return FactorizedSynthesizer(**config)
    if method == "diffusion":
        return PhaseDiffusion(**config)
    raise ValueError(f"Unknown learned method: {method}")


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch_size and learning rate must be positive.")
    train, validation = PreparedDataset(args.train), PreparedDataset(args.validation)
    validate_splits(train, validation, args.batch_size)
    config = {"num_filters": args.num_filters, "num_pool_layers": args.num_pool_layers}
    if args.method == "factorized":
        if len(train.coil_counts | validation.coil_counts) != 1:
            raise ValueError("Factorized synthesis requires one consistent coil count and ordering.")
        config.update(num_coils=next(iter(train.coil_counts)), map_size=args.map_size)
    else:
        config["timesteps"] = args.timesteps
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = _make_model(args.method, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    first_epoch, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["method"] != args.method or checkpoint["config"] != config:
            raise ValueError("Resume architecture differs; use the same model arguments.")
        if checkpoint["domain"] != next(iter(train.domains)):
            raise ValueError("Resume domain differs from the training domain.")
        if checkpoint["training_sources"] != sorted(train.source_hashes) or checkpoint["validation_sources"] != sorted(
            validation.source_hashes
        ):
            raise ValueError("Resume data differs from the checkpoint's training or validation sources.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch, best = checkpoint["epoch"] + 1, checkpoint["best_validation"]
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["sampling_rng"])
        loader_generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if first_epoch >= args.epochs:
        raise ValueError("epochs must exceed the completed epoch count in the resume checkpoint.")
    args.output.mkdir(parents=True, exist_ok=False)
    training_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=loader_generator)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    run_info = {
        "method": args.method,
        "config": config,
        "domain": next(iter(train.domains)),
        "seed": args.seed,
        "direct_version": __version__,
        "torch_version": str(torch.__version__),
        "training_subjects": sorted(train.subjects),
        "validation_subjects": sorted(validation.subjects),
        "training_sources": sorted(train.source_hashes),
        "validation_sources": sorted(validation.source_hashes),
    }
    (args.output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    for epoch in range(first_epoch, args.epochs):
        scores = {}
        for split, loader in (("train", training_loader), ("validation", validation_loader)):
            model.train(split == "train")
            running, count = 0.0, 0
            validation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
            with torch.set_grad_enabled(split == "train"):
                for batch in loader:
                    batch = {name: tensor.to(device) for name, tensor in batch.items()}
                    if isinstance(model, PhaseDiffusion):
                        losses = model.loss(batch, generator if split == "train" else validation_generator)
                    else:
                        losses = model.loss(batch)
                    loss = losses["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {split} loss at epoch {epoch}.")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                        optimizer.step()
                    size = batch["magnitude"].shape[0]
                    running += float(loss.detach()) * size
                    count += size
            scores[split] = running / count
        improved = scores["validation"] < best
        best = min(best, scores["validation"])
        checkpoint = {
            **run_info,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation": best,
            "torch_rng": torch.get_rng_state(),
            "sampling_rng": generator.get_state(),
            "loader_rng": loader_generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        _save_checkpoint(args.output / "last.pt", checkpoint)
        if improved:
            _save_checkpoint(args.output / "best.pt", checkpoint)
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch, **scores}) + "\n")
        logger.info("Epoch %d: train %.6f, validation %.6f", epoch + 1, scores["train"], scores["validation"])


def _generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    magnitude, provenance = read_dicom(args.dicom, args.assume_magnitude)
    magnitude = magnitude.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    provenance.update(method=args.method, seed=args.seed, domain=args.domain, direct_version=__version__)
    model = None
    if args.method != "baseline":
        if args.checkpoint is None:
            raise ValueError("Learned synthesis requires a trained --checkpoint.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["method"] != args.method or checkpoint["domain"] != args.domain:
            raise ValueError("Checkpoint method/domain must match the requested DICOM domain.")
        model = _make_model(args.method, checkpoint["config"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        provenance["checkpoint_sha256"] = _sha256(args.checkpoint)
    elif args.checkpoint:
        raise ValueError("The baseline does not use a learned checkpoint.")
    if args.method == "factorized":
        if args.maps:
            raise ValueError("Factorized synthesis predicts its own maps; omit --maps.")
        if not isinstance(model, FactorizedSynthesizer):
            raise ValueError("A factorized checkpoint is required.")
        with torch.no_grad():
            output = model(magnitude)
    else:
        if args.maps:
            with h5py.File(args.maps, "r") as handle:
                if handle.attrs.get("prepared_synthesis_version") != 1 or handle.attrs.get("domain") != args.domain:
                    raise ValueError("Empirical maps must be a prepared file in the requested domain.")
                if not 0 <= args.map_slice < handle["sensitivity_map"].shape[0]:
                    raise ValueError("map_slice is out of range.")
                maps = torch.from_numpy(handle["sensitivity_map"][args.map_slice : args.map_slice + 1]).to(device)
            if maps.shape[2:4] != magnitude.shape[-2:]:
                raise ValueError("Map and DICOM grids differ; match physical FOV/orientation and resample explicitly.")
            provenance.update(map_source_sha256=_sha256(args.maps), map_slice=args.map_slice, maps="empirical")
        else:
            if args.method == "diffusion":
                raise ValueError("Diffusion requires empirical --maps from the same calibration gauge/domain.")
            maps = simulated_maps(magnitude, args.num_coils, args.seed)
            provenance["maps"] = "simulated"
        if isinstance(model, PhaseDiffusion):
            output = model.sample(magnitude, maps, generator)
        else:
            output = synthesize(magnitude, smooth_phase(magnitude, generator, args.phase_scale), maps)
    export_h5(args.output, output, provenance)
    logger.info("Exported %s: %s", args.output, synthesis_diagnostics(output))


def main(argv: list[str] | None = None) -> None:
    """Run the synthesis command-line interface.

    Args:
        argv: Optional arguments, excluding the executable name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Calibrate factors from fully sampled multicoil HDF5 data")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--fully-sampled", action="store_true")
    prepare.add_argument("--acs-size", type=int, default=24)
    prepare.add_argument("--max-projection-nmse", type=float, default=0.35)
    prepare.add_argument("--no-crop-readout", action="store_true", help="Disable readout oversampling removal")
    prepare.set_defaults(function=_prepare)
    train = subparsers.add_parser("train", help="Train factorized or diffusion synthesis with isolated validation")
    train.add_argument("--method", choices=("factorized", "diffusion"), required=True)
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--num-filters", type=int, default=32)
    train.add_argument("--num-pool-layers", type=int, default=3)
    train.add_argument("--map-size", type=int, default=32)
    train.add_argument("--timesteps", type=int, default=1000)
    train.add_argument("--resume", type=Path)
    train.set_defaults(function=_train)
    generate = subparsers.add_parser(
        "generate", help="Convert one reviewed classic magnitude DICOM to synthetic k-space"
    )
    generate.add_argument("--method", choices=("baseline", "factorized", "diffusion"), required=True)
    generate.add_argument("--dicom", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--domain", required=True, help="Explicit cohort label, e.g. prostate_t2_3t_axial")
    generate.add_argument("--checkpoint", type=Path)
    generate.add_argument("--maps", type=Path)
    generate.add_argument("--map-slice", type=int, default=0)
    generate.add_argument("--num-coils", type=int, default=8)
    generate.add_argument("--phase-scale", type=float, default=2.0)
    generate.add_argument("--assume-magnitude", action="store_true")
    generate.set_defaults(function=_generate)
    for command in (train, generate):
        command.add_argument("--device", default="cpu")
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.function(args)


if __name__ == "__main__":
    main()
