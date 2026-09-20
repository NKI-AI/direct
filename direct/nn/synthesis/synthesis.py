"""Magnitude-to-multicoil k-space synthesis models for DIRECT.

Primary supervised pipelines (from fully sampled GT k-space labels):

1. **PhaseFromMagnitude**: magnitude → unit phase; loss vs SENSE phase label.
2. **ComplexFromMagnitude**: magnitude → complex image; loss vs SENSE complex label.

Additional / experimental:

3. **FactorizedSynthesizer**: jointly predicts phase + low-res sensitivity maps,
   then applies the MRI forward model.
4. **PhaseDiffusion**: conditional DDPM on unit-phase coordinates.
5. **CycleConsistencySynthesizer**: synthesize → undersample → reconstruct ≈ magnitude.

Tensors use DIRECT float32 real/imag-last convention:

- Magnitude: ``(batch, 1, height, width)``
- Phase / complex image: ``(batch, height, width, 2)``
- Sensitivity maps / k-space: ``(batch, coils, height, width, 2)``
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from direct.data import transforms as T
from direct.nn.unet.unet_2d import UnetModel2d
from direct.synthesis.physics import (
    SynthesisOutput,
    normalize_sensitivities,
    synthesize,
    unit_complex,
    validate_magnitude,
)


def _condition(magnitude: torch.Tensor, pools: int) -> torch.Tensor:
    """Normalize magnitude for U-Net conditioning.

    Args:
        magnitude: Nonneg float32 ``(B, 1, H, W)``.
        pools: Number of pooling layers; spatial dims must be >= 2^(pools+1).

    Returns:
        Normalized magnitude in [0, 1].
    """
    validate_magnitude(magnitude)
    if min(magnitude.shape[-2:]) < 2 ** (pools + 1):
        raise ValueError(f"Images must have both dimensions >= {2 ** (pools + 1)} for this U-Net depth.")
    return magnitude / magnitude.amax((2, 3), keepdim=True).clamp_min(1e-8)


def _weighted_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean with safe denominator."""
    return (error * weight).sum() / weight.expand_as(error).sum().clamp_min(1e-8)


class PhaseFromMagnitude(nn.Module):
    """Predict unit-complex phase from magnitude (pipeline A).

    Training: ``magnitude → phase``, loss vs SENSE phase label from GT k-space.
    """

    def __init__(
        self,
        num_filters: int = 32,
        num_pool_layers: int = 3,
        forward_operator: Any = None,
        backward_operator: Any = None,
    ):
        super().__init__()
        if num_filters < 1 or num_pool_layers < 1:
            raise ValueError("Positive filters and pool depth are required.")
        self.pools = num_pool_layers
        self.net = UnetModel2d(1, 2, num_filters, num_pool_layers, 0.0)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Predict unit phase ``(B, H, W, 2)`` from magnitude ``(B, 1, H, W)``."""
        pred = self.net(_condition(magnitude, self.pools)).permute(0, 2, 3, 1)
        return unit_complex(pred)


class ComplexFromMagnitude(nn.Module):
    """Predict SENSE-combined complex image from magnitude (pipeline B).

    Freely predicts complex (real, imag) to match the SENSE label; unlike
    :class:`PhaseFromMagnitude`, magnitude is not constrained to the RSS input.
    """

    def __init__(
        self,
        num_filters: int = 32,
        num_pool_layers: int = 3,
        forward_operator: Any = None,
        backward_operator: Any = None,
    ):
        super().__init__()
        if num_filters < 1 or num_pool_layers < 1:
            raise ValueError("Positive filters and pool depth are required.")
        self.pools = num_pool_layers
        self.net = UnetModel2d(1, 2, num_filters, num_pool_layers, 0.0)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Predict complex image ``(B, H, W, 2)`` from magnitude ``(B, 1, H, W)``."""
        return self.net(_condition(magnitude, self.pools)).permute(0, 2, 3, 1)


class FactorizedSynthesizer(nn.Module):
    """Predict phase and spatially coarse complex sensitivity maps in one pass.

    Uses two DIRECT U-Nets: one for phase prediction (magnitude → unit phase)
    and one for low-resolution sensitivity map prediction.  The forward model
    then computes ``k_c = FFT(S_c * m * exp(i*phi))`` with exact RSS preservation.

    Args:
        num_coils: Fixed coil count matching calibration data.
        num_filters: Initial U-Net channel width.
        num_pool_layers: Depth of the phase U-Net encoder/decoder.
        map_size: Square grid for sensitivity prediction; low res regularizes maps.
        forward_operator: Not used (compatibility with DIRECT's model init).
        backward_operator: Not used (compatibility with DIRECT's model init).
    """

    def __init__(
        self,
        num_coils: int = 15,
        num_filters: int = 32,
        num_pool_layers: int = 3,
        map_size: int = 32,
        forward_operator: Any = None,
        backward_operator: Any = None,
    ):
        super().__init__()
        if num_coils < 1 or num_filters < 1 or num_pool_layers < 1 or map_size < 8:
            raise ValueError("Positive coil count, filters, depth, and map_size >= 8 are required.")
        self.num_coils = num_coils
        self.pools = num_pool_layers
        self.map_size = map_size
        self.phase_net = UnetModel2d(1, 2, num_filters, num_pool_layers, 0.0)
        self.map_net = UnetModel2d(3, 2 * num_coils, num_filters, 2, 0.0)

    def forward(self, magnitude: torch.Tensor) -> SynthesisOutput:
        """Generate clean multicoil k-space from magnitude.

        Args:
            magnitude: Nonnegative float32 images ``(B, 1, H, W)``.

        Returns:
            :class:`SynthesisOutput` with predicted phase, maps, and k-space
            whose inverse-FFT RSS equals the input magnitude.
        """
        condition = _condition(magnitude, self.pools)
        phase = self.phase_net(condition).permute(0, 2, 3, 1)
        low = F.interpolate(condition, (self.map_size, self.map_size), mode="area")
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, self.map_size, device=magnitude.device),
            torch.linspace(-1, 1, self.map_size, device=magnitude.device),
            indexing="ij",
        )
        coords = torch.stack((yy, xx))[None].expand(magnitude.shape[0], -1, -1, -1)
        maps = self.map_net(torch.cat((low, coords), 1))
        maps = F.interpolate(maps, magnitude.shape[-2:], mode="bilinear", align_corners=False)
        maps = maps.reshape(magnitude.shape[0], self.num_coils, 2, *magnitude.shape[-2:]).permute(0, 1, 3, 4, 2)
        return synthesize(magnitude, phase, maps)


class PhaseDiffusion(nn.Module):
    """Conditional DDPM on Cartesian unit-phase coordinates.

    Operates on ``(cos(phi), sin(phi))`` in R^2 with a cosine noise schedule.
    The final sample is projected onto the unit circle.  Coil maps come from an
    external empirical bank in matching calibration gauge.

    A 32-dimensional sinusoidal time embedding conditions DIRECT's AdaIN U-Net.

    Args:
        num_filters: Initial U-Net channel width.
        num_pool_layers: Encoder/decoder depth.
        timesteps: Number of cosine-schedule diffusion steps (>= 2).
        forward_operator: Not used (DIRECT compatibility).
        backward_operator: Not used (DIRECT compatibility).
    """

    betas: torch.Tensor
    alpha_bar: torch.Tensor

    def __init__(
        self,
        num_filters: int = 32,
        num_pool_layers: int = 3,
        timesteps: int = 1000,
        eval_timesteps: int = 25,
        forward_operator: Any = None,
        backward_operator: Any = None,
    ):
        super().__init__()
        from direct.nn.adain.adain import NormType

        if timesteps < 2 or num_filters < 1 or num_pool_layers < 1:
            raise ValueError("timesteps >= 2 and positive U-Net width/depth are required.")
        self.timesteps = timesteps
        self.eval_timesteps = max(1, int(eval_timesteps))
        self.pools = num_pool_layers
        self.network = UnetModel2d(
            3, 2, num_filters, num_pool_layers, 0.0,
            norm_type=NormType.ADAIN,
            aux_in_features=32,
            adain_hidden_features=64,
        )
        times = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999).float()
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))

    def forward(
        self,
        noisy_phase: torch.Tensor,
        magnitude: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian noise at batch-specific diffusion times.

        Args:
            noisy_phase: Cartesian phase coordinates ``(B, 2, H, W)``.
            magnitude: Conditioning images ``(B, 1, H, W)``.
            times: Long tensor ``(B,)`` in ``[0, timesteps)``.

        Returns:
            Noise prediction ``(B, 2, H, W)``.
        """
        condition = _condition(magnitude, self.pools)
        if noisy_phase.shape != (magnitude.shape[0], 2, *magnitude.shape[-2:]):
            raise ValueError("noisy_phase must have shape (B, 2, H, W).")
        if times.shape != (magnitude.shape[0],) or times.dtype != torch.long:
            raise ValueError("times must be a long tensor of shape (B,).")
        if (times < 0).any() or (times >= self.timesteps).any():
            raise ValueError("Diffusion times are out of range.")
        frequencies = torch.exp(torch.arange(16, device=magnitude.device) * (-math.log(10000) / 15))
        angles = times[:, None].float() * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.network(torch.cat((noisy_phase, condition), dim=1), embedding)

    @torch.no_grad()
    def sample_phase(
        self,
        magnitude: torch.Tensor,
        generator: torch.Generator,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample unit-complex phase from noise, conditioned on magnitude.

        ``num_steps is None`` or ``>= timesteps`` uses ancestral DDPM.
        Fewer steps uses deterministic DDIM skipping of the cosine schedule.

        Args:
            magnitude: Images ``(B, 1, H, W)``.
            generator: Torch RNG on the same device.
            num_steps: Reverse steps. ``None`` means the full schedule.

        Returns:
            Unit phase ``(B, H, W, 2)``.
        """
        _condition(magnitude, self.pools)
        batch = magnitude.shape[0]
        value = torch.randn(
            batch, 2, *magnitude.shape[-2:],
            device=magnitude.device, generator=generator,
        )
        steps = self.timesteps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError("num_steps must be >= 1.")
        if steps >= self.timesteps:
            indices = list(range(self.timesteps - 1, -1, -1))
            ancestral = True
        else:
            indices = torch.linspace(self.timesteps - 1, 0, steps).long().tolist()
            ancestral = False

        for step_i, index in enumerate(indices):
            times = torch.full((batch,), index, device=magnitude.device, dtype=torch.long)
            epsilon = self(value, magnitude, times)
            alpha_bar = self.alpha_bar[index]
            clean = ((value - (1 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()).clamp(-1, 1)
            if ancestral:
                previous = self.alpha_bar[index - 1] if index else torch.ones_like(alpha_bar)
                beta = self.betas[index]
                alpha = 1 - beta
                mean = (previous.sqrt() * beta * clean + alpha.sqrt() * (1 - previous) * value) / (1 - alpha_bar)
                if index:
                    variance = beta * (1 - previous) / (1 - alpha_bar)
                    value = mean + variance.sqrt() * torch.randn(
                        value.shape, device=value.device, generator=generator
                    )
                else:
                    value = mean
            elif step_i + 1 >= len(indices):
                value = clean
            else:
                prev_ab = self.alpha_bar[indices[step_i + 1]]
                value = prev_ab.sqrt() * clean + (1 - prev_ab).sqrt() * epsilon
        return unit_complex(value.permute(0, 2, 3, 1))

    @torch.no_grad()
    def sample(
        self,
        magnitude: torch.Tensor,
        maps: torch.Tensor,
        generator: torch.Generator,
        num_steps: int | None = None,
    ) -> SynthesisOutput:
        """Sample phase and apply the MRI forward model.

        Args:
            magnitude: Images ``(B, 1, H, W)``.
            maps: Empirical sensitivity maps ``(B, C, H, W, 2)``.
            generator: Torch RNG on the same device.
            num_steps: Reverse steps. ``None`` uses the full ancestral schedule.

        Returns:
            Stochastic phase and clean multicoil k-space with preserved RSS.
        """
        maps = normalize_sensitivities(maps)
        identity = torch.zeros(magnitude.shape[0], *magnitude.shape[-2:], 2, device=magnitude.device)
        identity[..., 0] = 1
        synthesize(magnitude, identity, maps)
        return synthesize(magnitude, self.sample_phase(magnitude, generator, num_steps), maps)


class CycleConsistencySynthesizer(nn.Module):
    """Synthesis with cycle-consistency through a frozen reconstruction model.

    Predicts phase and sensitivity maps from magnitude, synthesizes multicoil
    k-space, then undersample-and-reconstructs to verify the output matches
    the input magnitude.  The reconstruction model is frozen; only synthesis
    parameters update.

    This enables training purely from magnitude images with no multicoil
    calibration targets, using the physical consistency constraint:

        magnitude → synthesize → undersample → reconstruct ≈ magnitude

    Args:
        num_coils: Fixed number of receiver coils.
        num_filters: Initial U-Net channel width.
        num_pool_layers: Encoder/decoder depth.
        map_size: Low-resolution grid for sensitivity prediction.
        forward_operator: FFT operator (used by reconstruction model).
        backward_operator: iFFT operator (used by reconstruction model).
    """

    def __init__(
        self,
        num_coils: int = 15,
        num_filters: int = 32,
        num_pool_layers: int = 3,
        map_size: int = 32,
        forward_operator: Any = None,
        backward_operator: Any = None,
    ):
        super().__init__()
        self.synthesizer = FactorizedSynthesizer(
            num_coils=num_coils,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            map_size=map_size,
        )
        self.forward_operator = forward_operator or T.fft2
        self.backward_operator = backward_operator or T.ifft2

    def forward(self, magnitude: torch.Tensor) -> SynthesisOutput:
        """Synthesize multicoil k-space from magnitude.

        Args:
            magnitude: Nonneg float32 ``(B, 1, H, W)``.

        Returns:
            :class:`SynthesisOutput` with phase, maps, and k-space.
        """
        return self.synthesizer(magnitude)


class PhaseFlowMatching(nn.Module):
    """Conditional Flow Matching for unit-phase generation.

    Learns a velocity field ``v(x_t, t | magnitude)`` that transports samples
    from Gaussian noise to the target phase distribution via an ODE:

        dx/dt = v(x_t, t | magnitude),    t ∈ [0, 1]

    Uses optimal transport conditional flow matching (OT-CFM):
        - x_0 ~ N(0, I)  (noise)
        - x_1 = target phase (cos φ, sin φ)
        - x_t = (1 - t) * x_0 + t * x_1  (linear interpolation)
        - v_target = x_1 - x_0  (constant velocity along the path)

    Simpler than diffusion: no noise schedule, direct velocity regression.
    At inference, integrate the ODE with Euler or RK4 and project onto S¹.

    Args:
        num_filters: Initial U-Net channel width.
        num_pool_layers: Encoder/decoder depth.
        num_integration_steps: ODE integration steps for sampling.
        forward_operator: Not used (DIRECT compatibility).
        backward_operator: Not used (DIRECT compatibility).
    """

    def __init__(
        self,
        num_filters: int = 32,
        num_pool_layers: int = 3,
        num_integration_steps: int = 50,
        eval_integration_steps: int = 10,
        forward_operator: Any = None,
        backward_operator: Any = None,
    ):
        super().__init__()
        from direct.nn.adain.adain import NormType

        if num_filters < 1 or num_pool_layers < 1:
            raise ValueError("Positive U-Net width and depth required.")
        if num_integration_steps < 1:
            raise ValueError("At least 1 integration step required.")
        self.pools = num_pool_layers
        self.eval_steps = max(1, int(eval_integration_steps))
        self.num_steps = num_integration_steps
        # Input: noisy phase (2) + magnitude condition (1) + time embedding (1) = 4 channels
        # Time is broadcast as a spatial constant channel
        self.network = UnetModel2d(
            4, 2, num_filters, num_pool_layers, 0.0,
            norm_type=NormType.ADAIN,
            aux_in_features=32,
            adain_hidden_features=64,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        magnitude: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity at interpolated state x_t and time t.

        Args:
            x_t: Interpolated phase coordinates ``(B, 2, H, W)``.
            magnitude: Conditioning images ``(B, 1, H, W)``.
            t: Float tensor ``(B,)`` in ``[0, 1]``.

        Returns:
            Velocity prediction ``(B, 2, H, W)``.
        """
        condition = _condition(magnitude, self.pools)
        if x_t.shape != (magnitude.shape[0], 2, *magnitude.shape[-2:]):
            raise ValueError("x_t must have shape (B, 2, H, W).")
        if t.shape != (magnitude.shape[0],) or not t.dtype.is_floating_point:
            raise ValueError("t must be a float tensor of shape (B,).")
        # Time embedding: sinusoidal
        frequencies = torch.exp(torch.arange(16, device=magnitude.device) * (-math.log(10000) / 15))
        angles = t[:, None].float() * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        # Broadcast t as spatial channel
        t_channel = t[:, None, None, None].expand(-1, 1, *magnitude.shape[-2:])
        net_input = torch.cat((x_t, condition, t_channel), dim=1)
        return self.network(net_input, embedding)

    @torch.no_grad()
    def sample(
        self,
        magnitude: torch.Tensor,
        maps: torch.Tensor,
        generator: torch.Generator,
        num_steps: int | None = None,
    ) -> SynthesisOutput:
        """Sample phase via Euler ODE integration, then apply MRI forward model.

        Args:
            magnitude: Images ``(B, 1, H, W)``.
            maps: Empirical sensitivity maps ``(B, C, H, W, 2)``.
            generator: Torch RNG on the same device.
            num_steps: Override default integration steps.

        Returns:
            Stochastic phase and clean multicoil k-space with preserved RSS.
        """
        _condition(magnitude, self.pools)
        maps = normalize_sensitivities(maps)
        steps = num_steps or self.num_steps
        dt = 1.0 / steps

        # Start from noise
        x = torch.randn(
            magnitude.shape[0], 2, *magnitude.shape[-2:],
            device=magnitude.device, generator=generator,
        )

        # Euler integration from t=0 to t=1
        for i in range(steps):
            t = torch.full((magnitude.shape[0],), i * dt, device=magnitude.device)
            v = self(x, magnitude, t)
            x = x + dt * v

        # Project onto unit circle
        phase = unit_complex(x.permute(0, 2, 3, 1))
        return synthesize(magnitude, phase, maps)
