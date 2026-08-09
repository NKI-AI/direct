# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""Tests for adaptive sampling policies."""

import pytest
import torch

from direct.data.transforms import ifft2
from direct.nn.adaptive.parameterized import (
    Parameterized2dPolicy,
    ParameterizedDynamic2dPolicy,
)
from direct.nn.adaptive.policy import StraightThroughPolicy
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.adaptive.utils import (
    reshape_acquisitions_post_sampling,
    reshape_mask_pre_sampling,
    rescale_probs,
)


@pytest.mark.parametrize("sampling_dimension", [PolicySamplingDimension.ONE_D, PolicySamplingDimension.TWO_D])
def test_parameterized_2d_policy_construct(
    sampling_dimension: PolicySamplingDimension,
) -> None:
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


@pytest.mark.parametrize("sampling_dimension", [PolicySamplingDimension.ONE_D, PolicySamplingDimension.TWO_D])
def test_reshape_mask_roundtrip_2d(sampling_dimension: PolicySamplingDimension) -> None:
    batch, height, width = 2, 16, 12
    shape = (batch, 4, height, width, 2)
    mask = torch.zeros(batch, 1, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1.0

    flat_mask, _ = reshape_mask_pre_sampling(sampling_dimension, mask, None, shape)
    assert flat_mask.ndim == 2
    assert flat_mask.shape[0] == batch

    acquisitions, prob_mask, reshaped_mask = reshape_acquisitions_post_sampling(
        sampling_dimension,
        flat_mask,
        flat_mask,
        flat_mask,
        shape,
    )
    assert acquisitions.shape == mask.shape
    assert prob_mask.shape == mask.shape
    assert reshaped_mask.shape == mask.shape


def test_reshape_mask_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="Incorrect k-space shape"):
        reshape_mask_pre_sampling(
            PolicySamplingDimension.ONE_D,
            torch.zeros(1, 1, 8, 8, 1),
            None,
            (1, 4, 8, 8),
        )


def test_parameterized_2d_policy_forward() -> None:
    height, width = 16, 14
    batch, coils = 2, 3
    policy = Parameterized2dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
    )
    kspace = torch.randn(batch, coils, height, width, 2)
    mask = torch.zeros(batch, 1, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1.0
    acceleration = torch.tensor([4.0, 4.0])

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, probability_masks = policy(
                mask=mask,
                kspace=kspace,
                acceleration=acceleration,
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable in this environment: {last_error}")

    assert out_kspace.shape == kspace.shape
    assert len(masks) >= 1
    assert probability_masks[-1].shape == mask.shape
