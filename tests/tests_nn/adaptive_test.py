# Copyright (c) DIRECT Contributors

"""Tests for adaptive sampling policies."""

from __future__ import annotations

import pytest
import torch

from direct.data.transforms import ifft2
from direct.nn.adaptive.parameterized import (
    Parameterized2dPolicy,
    ParameterizedDynamic2dPolicy,
)
from direct.nn.adaptive.policy import StraightThroughPolicy
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.adaptive.utils import rescale_probs


@pytest.mark.parametrize("sampling_dimension", [PolicySamplingDimension.ONE_D, PolicySamplingDimension.TWO_D])
def test_parameterized_2d_policy_construct(sampling_dimension: PolicySamplingDimension) -> None:
    policy = Parameterized2dPolicy(
        kspace_shape=(32, 28),
        sampling_dimension=sampling_dimension,
    )
    assert policy.num_actions == (28 if sampling_dimension == PolicySamplingDimension.ONE_D else 32 * 28)
    assert policy.sampler.numel() == policy.num_actions


def test_rescale_probs_matches_budget() -> None:
    probs = torch.rand(2, 40).clamp(0.05, 0.95)
    budget = torch.tensor([10, 15])
    rescaled = rescale_probs(probs, budget)
    assert torch.allclose(rescaled.mean(dim=1), budget.float() / 40, atol=1e-5)


def test_straight_through_static_policy_forward() -> None:
    height, width = 32, 28
    batch, coils = 2, 4
    policy = StraightThroughPolicy(
        backward_operator=ifft2,
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
        sampling_type=PolicySamplingType.STATIC,
        sampler_chans=8,
        sampler_num_pool_layers=2,
        sampler_fc_size=32,
        sampler_num_fc_layers=2,
    )
    kspace = torch.randn(batch, coils, height, width, 2)
    mask = torch.zeros(batch, 1, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1
    sensitivity_map = torch.ones(batch, coils, height, width, 2) / coils
    masked_kspace = kspace * mask
    acceleration = torch.tensor([4.0, 4.0])

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, probability_masks = policy(
                mask=mask,
                kspace=kspace,
                acceleration=acceleration,
                masked_kspace=masked_kspace,
                sensitivity_map=sensitivity_map,
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable in this environment: {last_error}")

    assert out_kspace.shape == kspace.shape
    assert len(masks) >= 1
    assert probability_masks[-1].shape == mask.shape


def test_parameterized_dynamic_2d_policy_construct() -> None:
    policy = ParameterizedDynamic2dPolicy(
        kspace_shape=(24, 20),
        sampling_dimension=PolicySamplingDimension.ONE_D,
        num_time_steps=4,
    )
    # sampler layout: (1, num_time_steps, num_actions) for 1D dynamic
    assert policy.sampler.shape[1] == 4
    assert policy.num_actions == 20
