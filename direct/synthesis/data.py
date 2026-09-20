"""Validated magnitude DICOM input, prepared factors, and DIRECT-compatible export."""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from direct.synthesis.physics import SynthesisOutput, synthesis_diagnostics, validate_magnitude


@contextmanager
def new_h5_file(path: Path) -> Iterator[h5py.File]:
    """Publish a new HDF5 file only after the writer completes successfully.

    Args:
        path: Destination on the same filesystem as its temporary file.

    Yields:
        Writable HDF5 handle; final publication refuses to overwrite a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".h5.tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            yield handle
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_dicom(path: Path, assume_magnitude: bool = False) -> tuple[torch.Tensor, dict]:
    """Read one classic, single-frame magnitude MR DICOM without display windowing.

    Args:
        path: DICOM file path.
        assume_magnitude: Accept absent magnitude-component metadata only after
            the caller has verified the source; explicit non-magnitude is rejected.

    Returns:
        Max-normalized float32 image (1, 1, H, W) and a limited provenance record.
        The scale permits recovering the rescaled stored-pixel units.

    Raises:
        ImportError: If optional pydicom is unavailable.
        ValueError: If unsupported, ambiguous, non-finite, negative, or empty data
            would otherwise be interpreted as a magnitude MR image.

    Notes:
        Enhanced/multiframe MR, Siemens mosaics, color, MONOCHROME1, secondary
        captures, explicit phase/real/imaginary/ADC maps and burned-in annotations
        are rejected. Stored pixels must already have been reviewed for suitability.
    """
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut
    except ImportError as exc:
        raise ImportError("DICOM input requires the optional dependency: pip install pydicom>=3") from exc

    dataset = pydicom.dcmread(path)
    if str(dataset.get("SOPClassUID", "")) != "1.2.840.10008.5.1.4.1.1.4" or dataset.get("Modality") != "MR":
        raise ValueError("Only classic MR Image Storage is supported; convert enhanced/mosaic data explicitly.")
    if int(dataset.get("NumberOfFrames", 1)) != 1 or int(dataset.get("SamplesPerPixel", 1)) != 1:
        raise ValueError("Only single-frame monochrome images are supported.")
    if dataset.get("PhotometricInterpretation") != "MONOCHROME2":
        raise ValueError("Only MONOCHROME2 stored magnitude pixels are supported.")
    image_type = [str(value).upper() for value in dataset.get("ImageType", [])]
    if "MOSAIC" in image_type or str(dataset.get("BurnedInAnnotation", "NO")).upper() == "YES":
        raise ValueError("Mosaics and images flagged with burned-in annotations are unsupported.")
    component = str(dataset.get("ComplexImageComponent", "")).upper()
    declared = image_type[2] if len(image_type) > 2 else ""
    if component and component != "MAGNITUDE":
        raise ValueError("ComplexImageComponent must be MAGNITUDE.")
    forbidden = {"P", "PHASE", "R", "REAL", "I", "IMAGINARY", "ADC", "DIFFUSION_MAP", "SWI"}
    if forbidden.intersection(image_type):
        raise ValueError("Phase, real/imaginary, and derived quantitative images are not magnitude input.")
    if component != "MAGNITUDE" and declared not in {"M", "MAGNITUDE"} and not assume_magnitude:
        raise ValueError("Magnitude component is unspecified; verify it before using assume_magnitude.")
    stored = dataset.pixel_array
    if stored.ndim != 2:
        raise ValueError("Decoded pixels must be a single 2D image.")
    padding = np.zeros(stored.shape, dtype=bool)
    if "PixelPaddingValue" in dataset:
        first = float(dataset.PixelPaddingValue)
        last = float(dataset.get("PixelPaddingRangeLimit", first))
        padding = (stored >= min(first, last)) & (stored <= max(first, last))
    array = np.asarray(apply_modality_lut(stored, dataset), dtype=np.float32).copy()
    array[padding] = 0
    if not np.isfinite(array).all() or np.any(array < 0) or array.max() <= 0:
        raise ValueError("Rescaled magnitude pixels must be finite, nonnegative, and nonempty.")
    scale = float(array.max())
    magnitude = torch.from_numpy(array / scale)[None, None]
    with path.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "source_sha256": source_hash,
        "intensity_scale": scale,
        "image_type": image_type,
        "assumed_magnitude": component != "MAGNITUDE" and declared not in {"M", "MAGNITUDE"},
        "grid": "stored_pixel_grid",
    }
    for name in ("PixelSpacing", "ImageOrientationPatient"):
        if name in dataset:
            metadata[name] = [float(value) for value in dataset[name].value]
    return magnitude, metadata


def export_h5(path: Path, output: SynthesisOutput, metadata: dict) -> None:
    """Write clean synthetic data consumable by DIRECT's existing FastMRIDataset.

    Args:
        path: New .h5 file; existing files are never overwritten.
        output: Batch interpreted as slices sharing one grid and coil count.
        metadata: JSON-serializable provenance; avoid patient identifiers.

    Raises:
        ValueError: If output is non-finite or inconsistent with its magnitude.
        FileExistsError: If the destination exists.

    Notes:
        Synthesis factors are stored under a separate group. FastMRIDataset does
        not read them, so reconstruction estimates maps from sampled ACS normally.
    """
    validate_magnitude(output.magnitude)
    diagnostics = synthesis_diagnostics(output)
    if not all(np.isfinite(value) for value in diagnostics.values()) or diagnostics["rss_nmse"] > 1e-8:
        raise ValueError("Synthesis failed finite-value or RSS consistency checks.")
    provenance = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with new_h5_file(path) as handle:
        kspace = torch.view_as_complex(output.kspace.detach().cpu().contiguous()).numpy()
        magnitude = output.magnitude[:, 0].detach().cpu().numpy()
        handle.create_dataset("kspace", data=kspace, compression="gzip")
        handle.create_dataset("reconstruction_rss", data=magnitude, compression="gzip")
        handle.attrs["max"] = float(magnitude.max())
        handle.attrs["synthetic"] = True
        handle.attrs["synthesis_version"] = 1
        handle.attrs["provenance"] = provenance
        handle.attrs["diagnostics"] = json.dumps(diagnostics)
        group = handle.create_group("synthesis")
        for name, tensor in (("sensitivity_map", output.sensitivity_map), ("phase", output.phase)):
            group.create_dataset(name, data=tensor.detach().cpu().numpy(), compression="gzip")


class PreparedDataset(Dataset):
    """Read calibrated pseudo-label slices without keeping HDF5 handles open.

    Args:
        root: Directory containing files produced by the prepare command.

    Raises:
        ValueError: If the directory is empty or schemas are incompatible.
    """

    def __init__(self, root: Path):
        self.files = sorted(root.glob("*.h5"))
        self.index: list[tuple[Path, int]] = []
        self.subjects: set[str] = set()
        self.source_hashes: set[str] = set()
        self.domains: set[str] = set()
        self.coil_counts: set[int] = set()
        self.shapes: set[tuple[int, int]] = set()
        for path in self.files:
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("prepared_synthesis_version") != 1:
                    raise ValueError(f"Not a prepared synthesis file: {path}")
                self.subjects.add(str(handle.attrs["subject_id"]))
                self.source_hashes.add(str(handle.attrs["source_sha256"]))
                self.domains.add(str(handle.attrs["domain"]))
                self.coil_counts.add(handle["sensitivity_map"].shape[1])
                self.shapes.add(tuple(handle["magnitude"].shape[-2:]))
                self.index.extend((path, index) for index in range(handle["magnitude"].shape[0]))
        if not self.index:
            raise ValueError(f"No prepared slices found in {root}.")

    def __len__(self) -> int:
        """Return the number of prepared slices."""
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Read a slice and normalize magnitude without altering phase or maps.

        Args:
            index: Slice index.

        Returns:
            Float32 magnitude, phase, maps, and phase-support weights.
        """
        path, slice_index = self.index[index]
        with h5py.File(path, "r") as handle:
            batch = {
                name: torch.from_numpy(np.asarray(handle[name][slice_index], dtype=np.float32))
                for name in ("magnitude", "phase", "sensitivity_map", "weight")
            }
        batch["magnitude"] /= batch["magnitude"].max().clamp_min(1e-8)
        return batch


def validate_splits(train: PreparedDataset, validation: PreparedDataset, batch_size: int) -> None:
    """Check patient isolation, domain agreement and collatable shapes.

    Args:
        train: Prepared training dataset.
        validation: Prepared validation dataset.
        batch_size: Requested loader batch size.

    Raises:
        ValueError: If patients overlap, domains differ, or batches cannot collate.
    """
    if train.subjects & validation.subjects:
        raise ValueError("Training and validation subject IDs overlap.")
    if train.source_hashes & validation.source_hashes:
        raise ValueError("Identical source files occur in training and validation.")
    if train.domains != validation.domains or len(train.domains) != 1:
        raise ValueError("Use one explicitly matched domain for training and validation.")
    if batch_size > 1 and (
        len(train.shapes | validation.shapes) != 1 or len(train.coil_counts | validation.coil_counts) != 1
    ):
        raise ValueError("Batches larger than one require matching shapes and coil counts.")
