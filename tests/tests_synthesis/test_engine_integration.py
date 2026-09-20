"""Integration tests for synthesis engine with DIRECT's training infrastructure."""

import numpy as np
import pytest
import torch

from direct.common.subsample import FastMRIRandomMaskFunc
from direct.data import transforms as T
from direct.data.mri_transforms import (
    EstimateSensitivityMapModule,
    ExtractCalibrationTargetsModule,
    TransformsType,
    build_synthesis_mri_transforms,
)
from direct.nn.recurrentvarnet.recurrentvarnet import RecurrentVarNet
from direct.nn.synthesis.config import (
    ComplexFromMagnitudeConfig,
    CycleConsistencySynthesizerConfig,
    FactorizedSynthesizerConfig,
    PhaseDiffusionConfig,
    PhaseFlowMatchingConfig,
    PhaseFromMagnitudeConfig,
)
from direct.nn.synthesis.synthesis import (
    ComplexFromMagnitude,
    CycleConsistencySynthesizer,
    FactorizedSynthesizer,
    PhaseDiffusion,
    PhaseFlowMatching,
    PhaseFromMagnitude,
)
from direct.synthesis.physics import (
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
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij",
    )
    image = torch.exp(-5 * (xx.square() + yy.square()))
    return image[None, None].repeat(batch, 1, 1, 1)


def generated(batch=1, coils=4, height=32, width=32):
    magnitude = phantom(batch, height, width)
    maps = simulated_maps(magnitude, coils, 0)
    phase = smooth_phase(magnitude, torch.Generator().manual_seed(3), scale=0.3)
    return synthesize(magnitude, phase, maps)


class TestPhaseFromMagnitude:
    """Pipeline A: magnitude → unit phase."""

    def test_forward_unit_phase(self):
        model = PhaseFromMagnitude(num_filters=4, num_pool_layers=2)
        phase = model(phantom())
        assert phase.shape == (1, 32, 32, 2)
        norms = phase.square().sum(-1).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradients_flow(self):
        model = PhaseFromMagnitude(num_filters=4, num_pool_layers=2)
        phase = model(phantom())
        phase.square().sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_config_defaults(self):
        cfg = PhaseFromMagnitudeConfig()
        assert cfg.engine_name == "PhaseFromMagnitudeEngine"
        assert cfg.model_name == "synthesis.synthesis.PhaseFromMagnitude"


class TestComplexFromMagnitude:
    """Pipeline B: magnitude → SENSE complex image."""

    def test_forward_shape(self):
        model = ComplexFromMagnitude(num_filters=4, num_pool_layers=2)
        pred = model(phantom())
        assert pred.shape == (1, 32, 32, 2)

    def test_gradients_flow(self):
        model = ComplexFromMagnitude(num_filters=4, num_pool_layers=2)
        pred = model(phantom())
        pred.square().sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_config_defaults(self):
        cfg = ComplexFromMagnitudeConfig()
        assert cfg.engine_name == "ComplexFromMagnitudeEngine"


class TestFactorizedSynthesizer:
    """Tests for the DIRECT-integrated FactorizedSynthesizer."""

    def test_forward_preserves_rss(self):
        model = FactorizedSynthesizer(num_coils=4, num_filters=4, num_pool_layers=2, map_size=8)
        magnitude = phantom()
        output = model(magnitude)
        diag = synthesis_diagnostics(output)
        assert diag["rss_nmse"] < 1e-10
        assert output.kspace.shape == (1, 4, 32, 32, 2)
        assert output.sensitivity_map.shape == (1, 4, 32, 32, 2)
        assert output.phase.shape == (1, 32, 32, 2)

    def test_gradients_flow(self):
        model = FactorizedSynthesizer(num_coils=4, num_filters=4, num_pool_layers=2, map_size=8)
        magnitude = phantom()
        output = model(magnitude)
        loss = output.kspace.square().sum()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_config_defaults(self):
        cfg = FactorizedSynthesizerConfig()
        assert cfg.num_coils == 15
        assert cfg.num_filters == 32
        assert cfg.model_name == "synthesis.synthesis.FactorizedSynthesizer"
        assert cfg.engine_name == "FactorizedSynthesisEngine"

    def test_direct_registry_resolves(self):
        from direct.utils import str_to_class
        cls = str_to_class("direct.nn.synthesis.synthesis", "FactorizedSynthesizer")
        assert cls is FactorizedSynthesizer

    def test_forward_operator_kwarg_ignored(self):
        model = FactorizedSynthesizer(
            num_coils=4, num_filters=4, num_pool_layers=2, map_size=8,
            forward_operator=T.fft2, backward_operator=T.ifft2,
        )
        output = model(phantom())
        assert synthesis_diagnostics(output)["rss_nmse"] < 1e-10


class TestPhaseDiffusion:
    """Tests for the DIRECT-integrated PhaseDiffusion."""

    def test_noise_prediction(self):
        model = PhaseDiffusion(num_filters=4, num_pool_layers=2, timesteps=4)
        magnitude = phantom()
        noisy = torch.randn(1, 2, 32, 32)
        times = torch.tensor([1], dtype=torch.long)
        pred = model(noisy, magnitude, times)
        assert pred.shape == (1, 2, 32, 32)

    def test_loss_and_gradient(self):
        model = PhaseDiffusion(num_filters=4, num_pool_layers=2, timesteps=4)
        magnitude = phantom()
        noisy = torch.randn(1, 2, 32, 32)
        times = torch.tensor([1], dtype=torch.long)
        pred = model(noisy, magnitude, times)
        loss = pred.square().mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_sample_preserves_rss(self):
        model = PhaseDiffusion(num_filters=4, num_pool_layers=2, timesteps=4)
        model.eval()
        magnitude = phantom()
        maps = simulated_maps(magnitude, 4, 0)
        gen = torch.Generator().manual_seed(42)
        output = model.sample(magnitude, maps, gen)
        diag = synthesis_diagnostics(output)
        assert diag["rss_nmse"] < 1e-10

    def test_ddim_sample_preserves_rss(self):
        model = PhaseDiffusion(num_filters=4, num_pool_layers=2, timesteps=8, eval_timesteps=3)
        model.eval()
        magnitude = phantom()
        maps = simulated_maps(magnitude, 4, 0)
        gen = torch.Generator().manual_seed(42)
        output = model.sample(magnitude, maps, gen, num_steps=3)
        diag = synthesis_diagnostics(output)
        assert diag["rss_nmse"] < 1e-10
        phase = model.sample_phase(magnitude, torch.Generator().manual_seed(0), num_steps=3)
        assert phase.shape == (1, 32, 32, 2)

    def test_config_defaults(self):
        cfg = PhaseDiffusionConfig()
        assert cfg.timesteps == 1000
        assert cfg.eval_timesteps == 25
        assert cfg.engine_name == "PhaseDiffusionSynthesisEngine"


class TestPhaseFlowMatching:
    """Tests for conditional flow matching on unit-phase coordinates."""

    def test_velocity_prediction(self):
        model = PhaseFlowMatching(num_filters=4, num_pool_layers=2, num_integration_steps=5)
        magnitude = phantom()
        x_t = torch.randn(1, 2, 32, 32)
        t = torch.tensor([0.5])
        v = model(x_t, magnitude, t)
        assert v.shape == (1, 2, 32, 32)

    def test_loss_and_gradient(self):
        model = PhaseFlowMatching(num_filters=4, num_pool_layers=2, num_integration_steps=5)
        magnitude = phantom()
        x_t = torch.randn(1, 2, 32, 32)
        t = torch.tensor([0.5])
        v = model(x_t, magnitude, t)
        loss = v.square().mean()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_sample_preserves_rss(self):
        model = PhaseFlowMatching(num_filters=4, num_pool_layers=2, num_integration_steps=5)
        model.eval()
        magnitude = phantom()
        maps = simulated_maps(magnitude, 4, 0)
        gen = torch.Generator().manual_seed(42)
        output = model.sample(magnitude, maps, gen)
        diag = synthesis_diagnostics(output)
        assert diag["rss_nmse"] < 1e-10

    def test_config_defaults(self):
        cfg = PhaseFlowMatchingConfig()
        assert cfg.num_integration_steps == 50
        assert cfg.eval_integration_steps == 10
        assert cfg.engine_name == "PhaseFlowMatchingEngine"


class TestCycleConsistencySynthesizer:
    """Tests for the cycle-consistency synthesis model."""

    def test_forward_preserves_rss(self):
        model = CycleConsistencySynthesizer(
            num_coils=4, num_filters=4, num_pool_layers=2, map_size=8,
        )
        output = model(phantom())
        assert synthesis_diagnostics(output)["rss_nmse"] < 1e-10

    def test_cycle_consistency_loss(self):
        """Test the full cycle: synthesize → undersample → reconstruct → loss."""
        model = CycleConsistencySynthesizer(
            num_coils=4, num_filters=4, num_pool_layers=2, map_size=8,
            forward_operator=T.fft2, backward_operator=T.ifft2,
        )
        magnitude = phantom()
        output = model(magnitude)

        # Create mask
        mask_func = FastMRIRandomMaskFunc([4], [0.25])
        mask = mask_func(output.kspace.shape[2:], seed=0)[None]
        masked = output.kspace * mask

        # Zero-filled recon
        recon = T.root_sum_of_squares(T.ifft2(masked, dim=(2, 3)), dim=1)
        loss = (recon - magnitude[:, 0]).square().mean()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_config_defaults(self):
        cfg = CycleConsistencySynthesizerConfig()
        assert cfg.recon_weight == 1.0
        assert cfg.phase_weight == 0.1
        assert cfg.accelerations == [4, 8]


class TestEngineRegistry:
    """Test that engines resolve through DIRECT's registry."""

    @pytest.mark.parametrize("engine_name,module", [
        ("PhaseFromMagnitudeEngine", "direct.nn.synthesis.synthesis_engine"),
        ("ComplexFromMagnitudeEngine", "direct.nn.synthesis.synthesis_engine"),
        ("FactorizedSynthesisEngine", "direct.nn.synthesis.synthesis_engine"),
        ("PhaseDiffusionSynthesisEngine", "direct.nn.synthesis.synthesis_engine"),
        ("PhaseFlowMatchingEngine", "direct.nn.synthesis.synthesis_engine"),
        ("CycleConsistencySynthesisEngine", "direct.nn.synthesis.synthesis_engine"),
    ])
    def test_engine_resolves(self, engine_name, module):
        from direct.utils import str_to_class
        cls = str_to_class(module, engine_name)
        assert cls is not None


class TestFullPipeline:
    """End-to-end synthesis → reconstruction pipeline test."""

    def test_synthesize_undersample_reconstruct_backprop(self):
        """Verify the complete pipeline works with RecurrentVarNet."""
        # Synthesize
        output = generated(coils=4)
        kspace = output.kspace

        # Undersample
        mask_func = FastMRIRandomMaskFunc([4], [0.25])
        mask = mask_func(kspace.shape[2:], seed=0)[None]
        masked_kspace = kspace * mask

        # Estimate sensitivity maps
        acs_mask = mask_func(kspace.shape[2:], seed=0, return_acs=True)[None]
        acs_kspace = kspace * acs_mask
        sens = EstimateSensitivityMapModule()({"acs_kspace": acs_kspace})["sensitivity_map"]

        # Reconstruct
        recon = RecurrentVarNet(
            T.fft2, T.ifft2,
            num_steps=1, recurrent_hidden_channels=4, recurrent_num_layers=1,
        )
        recon_kspace = recon(masked_kspace, mask, sens)
        image = T.root_sum_of_squares(T.ifft2(recon_kspace, dim=(2, 3)), dim=1)

        # Loss and backprop
        loss = (image - output.magnitude[:, 0]).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in recon.parameters())

    def test_export_and_load_with_fastmri_dataset(self, tmp_path):
        """Exported synthetic data loads correctly in FastMRIDataset."""
        from direct.data.datasets import FastMRIDataset
        from direct.synthesis.data import export_h5

        output = generated(coils=4)
        export_h5(tmp_path / "synthetic.h5", output, {"method": "test"})

        dataset = FastMRIDataset(data_root=tmp_path)
        sample = dataset[0]
        assert sample["kspace"].shape == (4, 32, 32)
        assert "reconstruction_size" in sample


class TestSynthesisTransforms:
    """Test synthesis transforms built on the supervised ACS / map path."""

    def _mask_func(self):
        return FastMRIRandomMaskFunc(accelerations=[4], center_fractions=[0.25])

    def _make_h5_with_kspace(self, tmp_path, coils=4, height=32, width=32, slices=2):
        """Create an HDF5 file with fully sampled multicoil k-space."""
        import h5py
        output = generated(batch=slices, coils=coils, height=height, width=width)
        kspace_complex = torch.view_as_complex(output.kspace.contiguous()).numpy()
        magnitude = output.magnitude[:, 0].numpy()
        path = tmp_path / "volume.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=kspace_complex)
            f.create_dataset("reconstruction_rss", data=magnitude)
            f.attrs["max"] = float(magnitude.max())
        return path

    def _synth_transforms(self, **kwargs):
        defaults = dict(
            forward_operator=T.fft2,
            backward_operator=T.ifft2,
            mask_func=self._mask_func(),
            extract_phase_and_maps=True,
        )
        defaults.update(kwargs)
        return build_synthesis_mri_transforms(**defaults)

    def test_extract_calibration_targets_module(self):
        """ExtractCalibrationTargets uses pre-estimated sensitivity_map."""
        from direct.types import TransformKey

        output = generated(coils=4)
        sample = {
            "kspace": output.kspace,
            TransformKey.SENSITIVITY_MAP: output.sensitivity_map,
        }
        module = ExtractCalibrationTargetsModule(extract_phase_and_maps=True)
        result = module(sample)
        assert "magnitude" in result
        assert "phase" in result
        assert "complex_image" in result
        assert "weight" in result
        assert result["magnitude"].shape == (1, 1, 32, 32)
        assert result["phase"].shape == (1, 32, 32, 2)
        assert result["complex_image"].shape == (1, 32, 32, 2)
        assert result["weight"].shape == (1, 1, 32, 32)
        assert torch.isfinite(result["magnitude"]).all()
        # phase is unit-complex of SENSE image
        phase_norms = result["phase"].square().sum(-1).sqrt()
        assert torch.allclose(phase_norms, torch.ones_like(phase_norms), atol=1e-5)

    def test_extract_calibration_magnitude_only(self):
        """extract_phase_and_maps=False writes only magnitude."""
        output = generated(coils=4)
        sample = {"kspace": output.kspace}
        module = ExtractCalibrationTargetsModule(extract_phase_and_maps=False)
        result = module(sample)
        assert "magnitude" in result
        assert result["magnitude"].shape == (1, 1, 32, 32)
        assert "phase" not in result
        assert "weight" not in result

    def test_build_synthesis_transforms(self):
        """Full chain: masker ACS → EstimateSensitivityMap → labels."""
        transforms = self._synth_transforms()
        output = generated(coils=4)
        kspace_complex = torch.view_as_complex(output.kspace[0].contiguous()).numpy()
        sample = {"kspace": kspace_complex, "filename": "phantom.h5"}
        result = transforms(sample)
        assert "magnitude" in result
        assert "phase" in result
        assert "complex_image" in result
        assert "sensitivity_map" in result
        assert "weight" in result
        assert result["magnitude"].ndim == 3
        assert result["phase"].ndim == 3
        assert result["complex_image"].ndim == 3
        assert "kspace" not in result
        assert "sampling_mask" not in result
        assert "acs_mask" not in result

    def test_requires_mask_func_for_phase_maps(self):
        with pytest.raises(ValueError, match="mask_func"):
            build_synthesis_mri_transforms(
                forward_operator=T.fft2,
                backward_operator=T.ifft2,
                mask_func=None,
                extract_phase_and_maps=True,
            )

    def test_synthesis_transforms_with_fastmri_dataset(self, tmp_path):
        from direct.data.datasets import FastMRIDataset
        self._make_h5_with_kspace(tmp_path)
        dataset = FastMRIDataset(data_root=tmp_path, transform=self._synth_transforms())
        sample = dataset[0]
        assert "magnitude" in sample
        assert "phase" in sample
        assert "sensitivity_map" in sample
        assert "weight" in sample
        assert sample["magnitude"].ndim == 3
        assert sample["phase"].ndim == 3

    def test_synthesis_transforms_collate_to_engine_shapes(self, tmp_path):
        from torch.utils.data import DataLoader

        from direct.data.datasets import FastMRIDataset
        from direct.engine import mri_batch_collate

        self._make_h5_with_kspace(tmp_path)
        dataset = FastMRIDataset(data_root=tmp_path, transform=self._synth_transforms())
        loader = DataLoader(dataset, batch_size=2, collate_fn=mri_batch_collate)
        batch = next(iter(loader))
        assert batch["magnitude"].shape == (2, 1, 32, 32)
        assert batch["phase"].shape == (2, 32, 32, 2)
        assert batch["complex_image"].shape == (2, 32, 32, 2)
        assert batch["sensitivity_map"].shape[0] == 2
        assert batch["sensitivity_map"].shape[-1] == 2
        assert batch["weight"].shape == (2, 1, 32, 32)

    def test_phase_model_with_transform_output(self, tmp_path):
        from direct.data.datasets import FastMRIDataset
        self._make_h5_with_kspace(tmp_path)
        dataset = FastMRIDataset(data_root=tmp_path, transform=self._synth_transforms())
        sample = dataset[0]
        model = PhaseFromMagnitude(num_filters=4, num_pool_layers=2)
        pred = model(sample["magnitude"][None])
        assert pred.shape == (1, 32, 32, 2)
        loss = (pred - sample["phase"][None]).square().mean()
        loss.backward()
        assert torch.isfinite(loss)

    def test_complex_model_with_transform_output(self, tmp_path):
        from direct.data.datasets import FastMRIDataset
        self._make_h5_with_kspace(tmp_path)
        dataset = FastMRIDataset(data_root=tmp_path, transform=self._synth_transforms())
        sample = dataset[0]
        model = ComplexFromMagnitude(num_filters=4, num_pool_layers=2)
        pred = model(sample["magnitude"][None])
        assert pred.shape == (1, 32, 32, 2)
        loss = (pred - sample["complex_image"][None]).square().mean()
        loss.backward()
        assert torch.isfinite(loss)

    def test_factorized_model_with_transform_output(self, tmp_path):
        from direct.data.datasets import FastMRIDataset
        self._make_h5_with_kspace(tmp_path)
        dataset = FastMRIDataset(data_root=tmp_path, transform=self._synth_transforms())
        sample = dataset[0]
        model = FactorizedSynthesizer(num_coils=4, num_filters=4, num_pool_layers=2, map_size=8)
        magnitude = sample["magnitude"][None]
        output = model(magnitude)
        diag = synthesis_diagnostics(output)
        assert diag["rss_nmse"] < 1e-10
        assert output.kspace.shape[1] == 4


def test_synthesis_eval_metrics():
    from direct.nn.synthesis.synthesis_engine import _build_synthesis_metrics, _per_case_phase_metrics

    fns = _build_synthesis_metrics(["phase_mae_deg", "phase_chordal", "phase_gradient"])
    assert set(fns) == {"phase_mae_deg_metric", "phase_chordal_metric", "phase_gradient_metric"}
    pred = torch.nn.functional.normalize(torch.randn(2, 8, 8, 2), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(2, 8, 8, 2), dim=-1)
    magnitude = torch.rand(2, 1, 8, 8)
    data = {"filename": ["a.h5", "b.h5"], "slice_no": torch.tensor([3, 4])}
    cases = _per_case_phase_metrics(pred, target, magnitude, None, data, fns)
    assert "a.h5_s3" in cases and "b.h5_s4" in cases
    assert torch.isfinite(cases["a.h5_s3"]["phase_mae_deg_metric"])


def test_phase_display_helpers():
    from direct.nn.synthesis.synthesis_engine import _mag_weighted_phase, _phase_hsv_rgb

    phase = torch.nn.functional.normalize(torch.randn(16, 16, 2), dim=-1)
    mag = torch.rand(16, 16)
    gray = _mag_weighted_phase(phase, mag)
    hsv = _phase_hsv_rgb(phase, mag)
    assert gray.shape == (1, 16, 16)
    assert hsv.shape == (3, 16, 16)
    assert gray.min() >= 0 and gray.max() <= 1
    assert hsv.min() >= 0 and hsv.max() <= 1


def test_synthesis_predict_writes_kspace(tmp_path):
    from types import SimpleNamespace

    from direct.nn.synthesis.synthesis_engine import PhaseFromMagnitudeEngine
    from direct.synthesis.physics import synthesis_diagnostics
    from direct.utils.writers import write_output_to_h5

    magnitude = phantom()
    maps = simulated_maps(magnitude, 4, 0)
    phase = torch.nn.functional.normalize(torch.randn(1, 32, 32, 2), dim=-1)
    data = {
        "magnitude": magnitude,
        "sensitivity_map": maps,
        "phase": phase,
        "filename": ["file1000002.h5"],
        "slice_no": torch.tensor([0]),
    }
    engine = PhaseFromMagnitudeEngine(
        cfg=SimpleNamespace(),
        model=PhaseFromMagnitude(num_filters=4, num_pool_layers=2),
        device="cpu",
    )
    engine.model.eval()
    output = engine.synthesize_from_batch(data)
    assert synthesis_diagnostics(output)["rss_nmse"] < 1e-8
    write_output_to_h5(
        (
            [(
                {
                    "kspace": output.kspace,
                    "sensitivity_map": output.sensitivity_map,
                    "phase": output.phase,
                    "magnitude": output.magnitude,
                    "metadata": {"engine": "test"},
                },
                None,
                "file1000002.h5",
            )],
            {"file1000002.h5": {"rss_nmse_metric": 0.0}},
        ),
        tmp_path,
    )
    import h5py
    with h5py.File(tmp_path / "file1000002.h5", "r") as handle:
        assert handle["kspace"].ndim == 4
        assert "reconstruction_rss" in handle
        assert handle.attrs["synthetic"]
    assert (tmp_path / "metrics_inference.json").exists()


def test_synthesis_predict_uses_simulated_coil_maps():
    from types import SimpleNamespace

    from direct.config.defaults import CoilSensitivitySimulationConfig, InferenceConfig
    from direct.nn.synthesis.synthesis_engine import PhaseFromMagnitudeEngine
    from direct.synthesis.physics import synthesis_diagnostics

    magnitude = phantom()
    data = {
        "magnitude": magnitude,
        "filename": ["file1000002.h5"],
        "slice_no": torch.tensor([0]),
    }
    engine = PhaseFromMagnitudeEngine(
        cfg=SimpleNamespace(
            inference=InferenceConfig(
                coil_sensitivity=CoilSensitivitySimulationConfig(mode="birdcage", num_coils=5, seed=0)
            )
        ),
        model=PhaseFromMagnitude(num_filters=4, num_pool_layers=2),
        device="cpu",
    )
    engine.model.eval()
    output = engine.synthesize_from_batch(data)
    assert output.sensitivity_map.shape == (1, 5, 32, 32, 2)
    assert synthesis_diagnostics(output)["rss_nmse"] < 1e-8
