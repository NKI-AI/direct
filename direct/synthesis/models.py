"""Supervised factor synthesis and conditional phase diffusion using DIRECT U-Nets."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from direct.data import transforms as T
from direct.nn.adain.adain import NormType
from direct.nn.unet.unet_2d import UnetModel2d
from direct.synthesis.physics import (
    SynthesisOutput,
    normalize_sensitivities,
    synthesize,
    unit_complex,
    validate_magnitude,
)


def _condition(magnitude: torch.Tensor, pools: int) -> torch.Tensor:
    validate_magnitude(magnitude)
    if min(magnitude.shape[-2:]) < 2 ** (pools + 1):
        raise ValueError(f"Images must have both dimensions >= {2 ** (pools + 1)} for this U-Net depth.")
    return magnitude / magnitude.amax((2, 3), keepdim=True).clamp_min(1e-8)


def _weighted_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (error * weight).sum() / weight.expand_as(error).sum().clamp_min(1e-8)


class FactorizedSynthesizer(nn.Module):
    """Predict phase and spatially coarse complex sensitivity maps in one pass.

    Args:
        num_coils: Fixed coil count with a consistent ordering in training data.
        num_filters: Initial DIRECT U-Net width.
        num_pool_layers: Depth of the phase U-Net.
        map_size: Square map prediction grid; low resolution regularizes maps.

    Raises:
        ValueError: If architecture dimensions are invalid.

    Notes:
        This deterministic baseline cannot represent all valid phase/coil
        realizations. It should be trained per coil configuration and protocol.
    """

    def __init__(self, num_coils: int, num_filters: int = 32, num_pool_layers: int = 3, map_size: int = 32):
        super().__init__()
        if num_coils < 1 or num_filters < 1 or num_pool_layers < 1 or map_size < 8:
            raise ValueError("Positive coil count, filters, depth, and map_size >= 8 are required.")
        self.num_coils = num_coils
        self.pools = num_pool_layers
        self.map_size = map_size
        self.phase_net = UnetModel2d(1, 2, num_filters, num_pool_layers, 0.0)
        self.map_net = UnetModel2d(3, 2 * num_coils, num_filters, 2, 0.0)

    def forward(self, magnitude: torch.Tensor) -> SynthesisOutput:
        """Generate clean k-space on the input image grid.

        Args:
            magnitude: Nonnegative float32 images (B, 1, H, W).

        Returns:
            Predicted phase, maps, and clean k-space with exact input RSS.
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

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Supervise gauge-fixed factors and their combined coil images.

        Args:
            batch: Prepared magnitude, phase, sensitivity_map and weight tensors.

        Returns:
            Total loss and phase, map, and coil-image components.

        Raises:
            ValueError: If the target coil count differs from the model.
        """
        if batch["sensitivity_map"].shape[1] != self.num_coils:
            raise ValueError("Training coil count differs from the factorized model.")
        output = self(batch["magnitude"])
        weight = batch["weight"][:, 0]
        phase_loss = _weighted_mean((output.phase - batch["phase"]).square().sum(-1), weight)
        map_loss = _weighted_mean((output.sensitivity_map - batch["sensitivity_map"]).square().sum(-1).sum(1), weight)
        predicted = T.complex_multiplication(output.sensitivity_map, output.phase[:, None])
        target = T.complex_multiplication(batch["sensitivity_map"], batch["phase"][:, None])
        coil_loss = _weighted_mean((predicted - target).square().sum(-1).sum(1), weight.square())
        return {
            "loss": coil_loss + 0.25 * (phase_loss + map_loss),
            "coil": coil_loss,
            "phase": phase_loss,
            "maps": map_loss,
        }


class PhaseDiffusion(nn.Module):
    """Conditional DDPM on Cartesian unit-phase coordinates.

    Args:
        num_filters: Initial DIRECT U-Net width.
        num_pool_layers: U-Net depth.
        timesteps: Number of cosine-schedule diffusion steps, at least two.

    Notes:
        A 32-dimensional sinusoidal time embedding conditions DIRECT's AdaIN.
        Diffusion operates in R^2; only the final sample is projected to the
        unit circle. This is not a wrapped-angle or manifold diffusion model.
        Coil maps come from an empirical bank in the same calibration gauge.
    """

    betas: torch.Tensor
    alpha_bar: torch.Tensor

    def __init__(self, num_filters: int = 32, num_pool_layers: int = 3, timesteps: int = 1000):
        super().__init__()
        if timesteps < 2 or num_filters < 1 or num_pool_layers < 1:
            raise ValueError("timesteps >= 2 and positive U-Net width/depth are required.")
        self.timesteps = timesteps
        self.pools = num_pool_layers
        self.network = UnetModel2d(
            3,
            2,
            num_filters,
            num_pool_layers,
            0.0,
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

    def forward(self, noisy_phase: torch.Tensor, magnitude: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Predict Gaussian noise at batch-specific diffusion times.

        Args:
            noisy_phase: Cartesian phase coordinates (B, 2, H, W).
            magnitude: Conditioning images (B, 1, H, W).
            times: Long tensor (B,) with values in [0, timesteps).

        Returns:
            Noise prediction (B, 2, H, W).

        Raises:
            ValueError: If phase or time dimensions or values are invalid.
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

    def loss(self, batch: dict[str, torch.Tensor], generator: torch.Generator) -> dict[str, torch.Tensor]:
        """Compute conditional epsilon-prediction loss at random diffusion times.

        Args:
            batch: Prepared magnitude, phase and support-weight tensors.
            generator: Generator on the training device.

        Returns:
            Scalar loss dictionary. A small background weight trains all pixels.
        """
        clean = batch["phase"].permute(0, 3, 1, 2)
        times = torch.randint(self.timesteps, (clean.shape[0],), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        alpha = self.alpha_bar[times, None, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        prediction = self(noisy, batch["magnitude"], times)
        return {"loss": _weighted_mean((prediction - noise).square(), 0.05 + batch["weight"])}

    @torch.no_grad()
    def sample(self, magnitude: torch.Tensor, maps: torch.Tensor, generator: torch.Generator) -> SynthesisOutput:
        """Run the full ancestral DDPM reverse chain, then apply MRI physics.

        Args:
            magnitude: Images (B, 1, H, W).
            maps: Empirical maps (B, C, H, W, 2), matching the training domain.
            generator: Generator on the same device.

        Returns:
            Stochastic phase and clean multicoil k-space with preserved RSS.

        Raises:
            ValueError: If magnitude or map shapes are invalid.
        """
        _condition(magnitude, self.pools)
        maps = normalize_sensitivities(maps)
        # Validate the forward-model boundary before running an expensive chain.
        identity = torch.zeros(magnitude.shape[0], *magnitude.shape[-2:], 2, device=magnitude.device)
        identity[..., 0] = 1
        synthesize(magnitude, identity, maps)
        value = torch.randn(magnitude.shape[0], 2, *magnitude.shape[-2:], device=magnitude.device, generator=generator)
        for index in range(self.timesteps - 1, -1, -1):
            times = torch.full((magnitude.shape[0],), index, device=magnitude.device, dtype=torch.long)
            epsilon = self(value, magnitude, times)
            alpha_bar = self.alpha_bar[index]
            previous = self.alpha_bar[index - 1] if index else torch.ones_like(alpha_bar)
            beta = self.betas[index]
            alpha = 1 - beta
            # Cartesian unit-phase targets lie in [-1, 1]; clipping x0 controls
            # the large final cosine-schedule beta without projecting each step.
            clean = ((value - (1 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()).clamp(-1, 1)
            mean = (previous.sqrt() * beta * clean + alpha.sqrt() * (1 - previous) * value) / (1 - alpha_bar)
            if index:
                variance = beta * (1 - previous) / (1 - alpha_bar)
                noise = torch.randn(value.shape, device=value.device, generator=generator)
                value = mean + variance.sqrt() * noise
            else:
                value = mean
        return synthesize(magnitude, unit_complex(value.permute(0, 2, 3, 1)), maps)
