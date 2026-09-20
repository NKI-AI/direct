"""Configuration for magnitude-conditioned synthesis models."""

from dataclasses import dataclass, field

from direct.config.defaults import ModelConfig


@dataclass
class PhaseFromMagnitudeConfig(ModelConfig):
    """magnitude → unit phase (chordal S¹ loss + optional gradient matching)."""

    model_name: str = "synthesis.synthesis.PhaseFromMagnitude"
    engine_name: str = "PhaseFromMagnitudeEngine"
    num_filters: int = 32
    num_pool_layers: int = 3
    # Weight on ‖∇û − ∇u‖² (local B0 / susceptibility structure).
    phase_gradient_weight: float = 0.1


@dataclass
class ComplexFromMagnitudeConfig(ModelConfig):
    """magnitude → SENSE complex (ℂ L2 + |z| L2 + chordal phase)."""

    model_name: str = "synthesis.synthesis.ComplexFromMagnitude"
    engine_name: str = "ComplexFromMagnitudeEngine"
    num_filters: int = 32
    num_pool_layers: int = 3
    complex_weight: float = 1.0
    magnitude_weight: float = 1.0
    phase_weight: float = 1.0
    phase_gradient_weight: float = 0.1


@dataclass
class FactorizedSynthesizerConfig(ModelConfig):
    """Joint phase + sensitivity map prediction + MRI forward model."""

    model_name: str = "synthesis.synthesis.FactorizedSynthesizer"
    engine_name: str = "FactorizedSynthesisEngine"
    num_coils: int = 15
    num_filters: int = 32
    num_pool_layers: int = 3
    map_size: int = 32


@dataclass
class PhaseDiffusionConfig(ModelConfig):
    """Conditional phase diffusion (DDPM on unit-phase coordinates)."""

    model_name: str = "synthesis.synthesis.PhaseDiffusion"
    engine_name: str = "PhaseDiffusionSynthesisEngine"
    num_filters: int = 32
    num_pool_layers: int = 3
    timesteps: int = 1000
    eval_timesteps: int = 25  # DDIM steps at validation (full val set, not ancestral 1000)


@dataclass
class PhaseFlowMatchingConfig(ModelConfig):
    """Conditional flow matching for phase (OT-CFM velocity field)."""

    model_name: str = "synthesis.synthesis.PhaseFlowMatching"
    engine_name: str = "PhaseFlowMatchingEngine"
    num_filters: int = 32
    num_pool_layers: int = 3
    num_integration_steps: int = 50
    eval_integration_steps: int = 10  # ODE steps at validation


@dataclass
class CycleConsistencySynthesizerConfig(ModelConfig):
    """Cycle-consistency synthesis."""

    model_name: str = "synthesis.synthesis.CycleConsistencySynthesizer"
    engine_name: str = "CycleConsistencySynthesisEngine"
    num_coils: int = 15
    num_filters: int = 32
    num_pool_layers: int = 3
    map_size: int = 32
    recon_model_name: str = "recurrentvarnet.recurrentvarnet.RecurrentVarNet"
    recon_checkpoint: str | None = None
    recon_weight: float = 1.0
    phase_weight: float = 0.1
    accelerations: list[int] = field(default_factory=lambda: [4, 8])
    center_fractions: list[float] = field(default_factory=lambda: [0.08, 0.04])
