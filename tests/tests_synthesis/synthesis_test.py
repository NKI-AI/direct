"""Physical, stochastic, I/O and downstream integration tests for MRI synthesis."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from direct.common.subsample import FastMRIRandomMaskFunc
from direct.data import transforms as T
from direct.data.datasets import FastMRIDataset
from direct.data.mri_transforms import EstimateSensitivityMapModule
from direct.nn.recurrentvarnet.recurrentvarnet import RecurrentVarNet
from direct.synthesis.cli import main
from direct.synthesis.data import export_h5, new_h5_file, read_dicom
from direct.synthesis.evaluation import kspace_features, reconstruction_metrics
from direct.synthesis.models import FactorizedSynthesizer, PhaseDiffusion
from direct.synthesis.physics import (
    calibration_targets,
    normalize_sensitivities,
    simulated_maps,
    smooth_phase,
    synthesis_diagnostics,
    synthesize,
)


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def phantom(batch=1, height=32, width=32):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    image = torch.exp(-5 * (xx.square() + yy.square()))
    return image[None, None].repeat(batch, 1, 1, 1)


def generated(batch=1, coils=4, height=32, width=32):
    magnitude = phantom(batch, height, width)
    maps = simulated_maps(magnitude, coils, 0)
    phase = smooth_phase(magnitude, torch.Generator().manual_seed(3), scale=0.3)
    return synthesize(magnitude, phase, maps)


@pytest.mark.parametrize("shape", [(1, 3, 31, 33), (2, 5, 32, 40), (1, 1, 16, 16)])
def test_exact_magnitude_and_independent_fft(shape):
    output = generated(*shape)
    diagnostics = synthesis_diagnostics(output)
    assert diagnostics["rss_nmse"] < 1e-12
    assert diagnostics["fft_nmse"] < 1e-12
    assert diagnostics["map_power_max_error"] < 1e-6
    kspace = torch.view_as_complex(output.kspace.contiguous())
    independent = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    assert torch.allclose(independent.abs().square().sum(1).sqrt()[:, None], output.magnitude, atol=1e-6)


def test_gauge_invariance():
    output = generated()
    gauge = smooth_phase(output.magnitude, torch.Generator().manual_seed(8))
    phase = T.complex_multiplication(output.phase, gauge)
    maps = T.complex_multiplication(output.sensitivity_map, T.conjugate(gauge[:, None]))
    transformed = synthesize(output.magnitude, phase, maps)
    assert torch.allclose(output.kspace, transformed.kspace, atol=1e-6)


def test_seed_zero_reproducible_and_numpy_state_preserved():
    state = np.random.get_state()
    first = simulated_maps(phantom(), 4, 0)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert torch.equal(first, simulated_maps(phantom(), 4, 0))
    assert not torch.equal(first, simulated_maps(phantom(), 4, 1))
    birdcage = simulated_maps(phantom(), 4, 0, mode="birdcage")
    assert birdcage.shape == first.shape
    power = birdcage.square().sum(-1).sum(1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-5)


def test_zero_signal_and_zero_maps_are_finite():
    magnitude = torch.zeros(1, 1, 16, 16)
    maps = normalize_sensitivities(torch.zeros(1, 4, 16, 16, 2))
    output = synthesize(magnitude, torch.zeros(1, 16, 16, 2), maps)
    assert torch.count_nonzero(output.kspace) == 0
    assert torch.allclose(maps.square().sum(-1).sum(1), torch.ones(1, 16, 16))


@pytest.mark.parametrize("invalid", ["negative", "nan", "dtype", "shape"])
def test_invalid_magnitude_rejected(invalid):
    output = generated()
    magnitude = output.magnitude.clone()
    if invalid == "negative":
        magnitude[0, 0, 0, 0] = -1
    elif invalid == "nan":
        magnitude[0, 0, 0, 0] = float("nan")
    elif invalid == "dtype":
        magnitude = magnitude.double()
    else:
        magnitude = magnitude[:, 0]
    with pytest.raises(ValueError):
        synthesize(magnitude, output.phase, output.sensitivity_map)


def test_calibration_and_projection_error():
    output = generated()
    target = calibration_targets(output.kspace, 16)
    assert torch.allclose(target["magnitude"], output.magnitude, atol=1e-6)
    assert torch.isfinite(target["phase"]).all()
    assert target["projection_nmse"].max() < 0.02
    assert target["weight"].min() >= 0 and target["weight"].max() <= 1
    with pytest.raises(ValueError):
        calibration_targets(output.kspace, 100)


def test_calibration_removes_global_receiver_phase():
    output = generated()
    angle = torch.tensor([0.3, 0.9539392])
    rotated = T.complex_multiplication(output.kspace, angle)
    original = calibration_targets(output.kspace, 16)
    changed = calibration_targets(rotated, 16)
    support = original["weight"][:, 0] > 0.1
    assert torch.allclose(
        original["sensitivity_map"].permute(0, 2, 3, 1, 4)[support],
        changed["sensitivity_map"].permute(0, 2, 3, 1, 4)[support],
        atol=2e-5,
    )
    assert torch.allclose(original["phase"][support], changed["phase"][support], atol=2e-5)


def test_h5_failure_does_not_publish_partial_file(tmp_path):
    path = tmp_path / "incomplete.h5"
    with pytest.raises(ValueError, match="interrupted"), new_h5_file(path) as handle:
        handle["example"] = np.zeros(2)
        raise ValueError("interrupted")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_reconstruction_config_and_transform_pipeline(tmp_path):
    from omegaconf import OmegaConf

    from direct.config.defaults import TrainingConfig, ValidationConfig
    from direct.data.datasets_config import FastMRIConfig
    from direct.data.mri_transforms import build_mri_transforms
    from direct.nn.recurrentvarnet.config import RecurrentVarNetConfig

    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.load(root / "projects/synthesis/reconstruction.yaml")
    OmegaConf.merge(OmegaConf.structured(RecurrentVarNetConfig), config.model)
    OmegaConf.merge(OmegaConf.structured(TrainingConfig), config.training)
    OmegaConf.merge(OmegaConf.structured(ValidationConfig), config.validation)
    for section in (config.training, config.validation):
        for dataset in section.datasets:
            OmegaConf.merge(OmegaConf.structured(FastMRIConfig), dataset)
    export_h5(tmp_path / "image.h5", generated(), {})
    transform = build_mri_transforms(T.fft2, T.ifft2, FastMRIRandomMaskFunc([4], [0.25]), crop=None)
    dataset = FastMRIDataset(tmp_path, transform=transform)
    sample = dataset[0]
    assert sample["masked_kspace"].shape == (4, 32, 32, 2)
    assert sample["sensitivity_map"].shape == (4, 32, 32, 2)
    assert torch.isfinite(sample["target"]).all()


@pytest.mark.parametrize("method", ["factorized", "diffusion"])
def test_learned_model_updates_and_state_roundtrip(method, tmp_path):
    torch.manual_seed(4)
    output = generated(batch=2)
    batch = calibration_targets(output.kspace, 16)
    model = FactorizedSynthesizer(4, 4, 2, 8) if method == "factorized" else PhaseDiffusion(4, 2, 4)
    generator = torch.Generator().manual_seed(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    losses = model.loss(batch) if method == "factorized" else model.loss(batch, generator)
    losses["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    state_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), state_path)
    model.load_state_dict(torch.load(state_path, weights_only=True))
    model.eval()
    if method == "factorized":
        result = model(output.magnitude)
    else:
        result = model.sample(output.magnitude, output.sensitivity_map, torch.Generator().manual_seed(17))
        same = model.sample(output.magnitude, output.sensitivity_map, torch.Generator().manual_seed(17))
        different = model.sample(output.magnitude, output.sensitivity_map, torch.Generator().manual_seed(18))
        assert torch.equal(result.kspace, same.kspace)
        assert not torch.allclose(result.kspace, different.kspace)
    assert synthesis_diagnostics(result)["rss_nmse"] < 1e-12


def test_h5_loader_acs_and_recurrentvarnet_training_step(tmp_path):
    output = generated()
    path = tmp_path / "synthetic.h5"
    export_h5(path, output, {"method": "baseline"})
    dataset = FastMRIDataset(data_root=tmp_path)
    sample = dataset[0]
    assert sample["kspace"].shape == (4, 32, 32)
    assert "sensitivity_map" not in sample
    assert sample["reconstruction_size"] == (32, 32, 1)
    kspace = T.to_tensor(sample["kspace"])[None]
    mask_function = FastMRIRandomMaskFunc([4], [0.25])
    mask = mask_function((32, 32, 2), seed=0)[None]
    acs = mask_function((32, 32, 2), seed=0, return_acs=True)[None]
    estimated = EstimateSensitivityMapModule()({"acs_kspace": kspace * acs})["sensitivity_map"]
    assert not torch.allclose(estimated, output.sensitivity_map)
    model = RecurrentVarNet(T.fft2, T.ifft2, num_steps=1, recurrent_hidden_channels=4, recurrent_num_layers=1)
    prediction = model(kspace * mask, mask, estimated)
    image = T.root_sum_of_squares(T.ifft2(prediction, dim=(2, 3)), dim=1)[:, None]
    loss = (image - output.magnitude).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    with pytest.raises(FileExistsError):
        export_h5(path, output, {})


def make_dicom(path: Path, component="M", **attributes):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MR"
    dataset.ImageType = ["ORIGINAL", "PRIMARY", component]
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = 32, 32
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 16, 16, 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope, dataset.RescaleIntercept = 2, 0
    dataset.WindowCenter, dataset.WindowWidth = 10, 20
    dataset.PixelSpacing = [0.5, 0.5]
    array = (phantom()[0, 0].numpy() * 1000).astype(np.uint16)
    dataset.PixelData = array.tobytes()
    for name, value in attributes.items():
        setattr(dataset, name, value)
    dataset.save_as(path, enforce_file_format=True)
    return array


def test_dicom_rescale_without_windowing_and_padding(tmp_path):
    path = tmp_path / "image.dcm"
    array = make_dicom(path, PixelPaddingValue=0)
    magnitude, provenance = read_dicom(path)
    assert np.allclose(magnitude[0, 0], array / array.max())
    assert provenance["intensity_scale"] == 2 * array.max()
    assert provenance["PixelSpacing"] == [0.5, 0.5]


@pytest.mark.parametrize(
    "attributes",
    [
        {"component": "P"},
        {"component": "ADC"},
        {"PhotometricInterpretation": "MONOCHROME1"},
        {"NumberOfFrames": 2},
        {"BurnedInAnnotation": "YES"},
        {"Modality": "CT"},
        {"ImageType": ["ORIGINAL", "PRIMARY", "M", "MOSAIC"]},
        {"RescaleIntercept": -1000},
    ],
)
def test_dicom_rejects_unsupported_inputs(tmp_path, attributes):
    path = tmp_path / "image.dcm"
    make_dicom(path, **attributes)
    with pytest.raises(ValueError):
        read_dicom(path, assume_magnitude=True)


def test_unknown_component_requires_explicit_assumption(tmp_path):
    path = tmp_path / "image.dcm"
    make_dicom(path, component="OTHER")
    with pytest.raises(ValueError, match="unspecified"):
        read_dicom(path)
    _, metadata = read_dicom(path, assume_magnitude=True)
    assert metadata["assumed_magnitude"]


def test_distribution_features_coil_permutation_and_scale():
    output = generated()
    original = kspace_features(output.kspace)
    permuted = kspace_features(output.kspace[:, [2, 0, 3, 1]] * 7)
    for key in original:
        assert torch.allclose(original[key], permuted[key], atol=1e-5)
    assert torch.allclose(original["radial_power"].sum(1), torch.ones(1))
    scores = reconstruction_metrics(output.magnitude * 0.9, output.magnitude)
    assert scores["nmse"] == pytest.approx(0.01, abs=1e-6)
    assert 0 < scores["ssim"] < 1


@pytest.mark.parametrize("method", ["factorized", "diffusion"])
def test_cli_prepare_train_resume_and_generate(tmp_path, method):
    output = generated()
    raw_path = tmp_path / "raw.h5"
    with h5py.File(raw_path, "w") as handle:
        handle["kspace"] = torch.view_as_complex(output.kspace.contiguous()).numpy()
    validation_path = tmp_path / "validation.h5"
    with h5py.File(validation_path, "w") as handle:
        handle["kspace"] = torch.view_as_complex((output.kspace * 1.1).contiguous()).numpy()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"path": str(raw_path), "subject_id": "001", "split": "train", "domain": "test"},
                {"path": str(validation_path), "subject_id": "002", "split": "validation", "domain": "test"},
            ]
        )
    )
    prepared = tmp_path / "prepared"
    main(["prepare", "--manifest", str(manifest), "--output", str(prepared), "--fully-sampled", "--acs-size", "16"])
    run = tmp_path / "run"
    common = [
        "train",
        "--method",
        method,
        "--train",
        str(prepared / "train"),
        "--validation",
        str(prepared / "validation"),
        "--num-filters",
        "4",
        "--num-pool-layers",
        "2",
        "--timesteps",
        "4",
        "--map-size",
        "8",
    ]
    main([*common, "--output", str(run), "--epochs", "1"])
    resumed = tmp_path / "resumed"
    main([*common, "--output", str(resumed), "--epochs", "2", "--resume", str(run / "last.pt")])
    assert torch.load(resumed / "last.pt", weights_only=True)["epoch"] == 1
    dicom = tmp_path / "image.dcm"
    make_dicom(dicom)
    export = tmp_path / "generated.h5"
    command = [
        "generate",
        "--method",
        method,
        "--checkpoint",
        str(resumed / "last.pt"),
        "--dicom",
        str(dicom),
        "--output",
        str(export),
        "--domain",
        "test",
    ]
    if method == "diffusion":
        command += ["--maps", str(prepared / "train" / "volume_000000.h5")]
    main(command)
    with h5py.File(export, "r") as handle:
        assert handle["kspace"].shape == (1, 4, 32, 32)
        assert handle.attrs["synthetic"]
    baseline = tmp_path / "baseline.h5"
    main(["generate", "--method", "baseline", "--dicom", str(dicom), "--output", str(baseline), "--domain", "test"])


def test_prepare_detects_patient_leakage_before_writing(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"path": str(manifest), "subject_id": "001", "split": "train", "domain": "test"},
                {"path": str(manifest), "subject_id": "001", "split": "test", "domain": "test"},
            ]
        )
    )
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="multiple splits"):
        main(["prepare", "--manifest", str(manifest), "--output", str(destination), "--fully-sampled"])
    assert not destination.exists()
