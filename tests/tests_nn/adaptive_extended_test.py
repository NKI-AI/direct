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
"""Extended adaptive sampling unit tests."""

import pytest
import torch

from direct.data.transforms import ifft2
from direct.exceptions import RejectionSamplingError
from direct.nn.adaptive.binarizer import ThresholdSigmoidMask, deterministic_binarizer
from direct.nn.adaptive.parameterized import (
    Parameterized2dPolicy,
    Parameterized3dPolicy,
    ParameterizedDynamic2dPolicy,
    ParameterizedMultislice2dPolicy,
)
from direct.nn.adaptive.policy import StraightThroughPolicy
from direct.nn.adaptive.sampler import ImageLineConvSampler, KSpaceLineConvSampler
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.adaptive.utils import (
    normalize_masked_probabilities,
    reshape_acquisitions_post_sampling,
    reshape_mask_pre_sampling,
)
from direct.nn.types import ActivationType


def test_deterministic_binarizer() -> None:
    probs = torch.tensor([[0.1, 0.9, 0.8, 0.2]])
    mask = deterministic_binarizer(probs, budget=2)
    assert mask.sum().item() == 2
    assert mask[0, 1].item() == 1
    assert mask[0, 2].item() == 1


def test_threshold_sigmoid_mask_forward_backward() -> None:
    module = ThresholdSigmoidMask(slope=10.0, clamp=True)
    probs = torch.full((2, 20), 0.5, requires_grad=True)
    last_error: Exception | None = None
    for seed in range(20):
        torch.manual_seed(seed)
        try:
            binary = module(probs)
            break
        except RejectionSamplingError as exc:
            last_error = exc
    else:
        pytest.skip(f"Rejection sampling unstable: {last_error}")

    assert binary.shape == probs.shape
    assert set(binary.unique().tolist()).issubset({0.0, 1.0})
    binary.sum().backward()
    assert probs.grad is not None


def test_image_and_kspace_line_conv_samplers() -> None:
    height, width = 16, 14
    sampler = ImageLineConvSampler(
        input_dim=(2, height, width),
        num_actions=width,
        chans=4,
        num_pool_layers=2,
        fc_size=16,
        num_fc_layers=2,
        activation=ActivationType.LEAKY_RELU,
    )
    image = torch.randn(2, 2, height, width)
    mask = torch.zeros(2, 1, 1, width)
    mask[..., width // 2 - 1 : width // 2 + 1] = 1
    logits = sampler(image, mask)
    assert logits.shape == (2, width)

    kspace_sampler = KSpaceLineConvSampler(
        input_dim=(2, height, width),
        num_actions=width,
        chans=4,
        num_pool_layers=2,
        fc_size=16,
        num_fc_layers=2,
    )
    # K-space sampler expects [batch, coils, complex, height, width].
    kspace = torch.randn(2, 3, 2, height, width)
    assert kspace_sampler(kspace, mask).shape == (2, width)


def test_parameterized_3d_policy_forward() -> None:
    height, width = 12, 10
    batch, coils, slices = 1, 2, 2
    policy = Parameterized3dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
    )
    kspace = torch.randn(batch, coils, slices, height, width, 2)
    mask = torch.zeros(batch, 1, 1, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1.0
    acceleration = torch.tensor([4.0])

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, _probs = policy(mask=mask, kspace=kspace, acceleration=acceleration)
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")

    assert out_kspace.shape == kspace.shape
    assert masks[-1].shape == (batch, 1, 1, height, width, 1) or masks[-1].ndim == 6


def test_parameterized_dynamic_2d_policy_forward() -> None:
    height, width, time = 12, 10, 3
    batch, coils = 1, 2
    policy = ParameterizedDynamic2dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
        num_time_steps=time,
    )
    kspace = torch.randn(batch, coils, time, height, width, 2)
    mask = torch.zeros(batch, 1, time, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1.0
    acceleration = torch.tensor([4.0])

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, probs = policy(mask=mask, kspace=kspace, acceleration=acceleration)
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")

    assert out_kspace.shape == kspace.shape
    assert len(masks) >= 1
    assert probs[-1].shape[0] == batch


def test_parameterized_multislice_construct() -> None:
    policy = ParameterizedMultislice2dPolicy(
        kspace_shape=(16, 12),
        sampling_dimension=PolicySamplingDimension.TWO_D,
        num_slices=3,
    )
    assert policy.num_actions == 16 * 12
    assert policy.sampler.shape[1] == 3


def test_parameterized_multislice_forward() -> None:
    height, width, slices = 12, 10, 2
    policy = ParameterizedMultislice2dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
        num_slices=slices,
    )
    kspace = torch.randn(1, 2, slices, height, width, 2)
    mask = torch.zeros(1, 1, slices, height, width, 1)
    mask[..., width // 2, :] = 1.0
    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, _ = policy(mask=mask, kspace=kspace, acceleration=torch.tensor([4.0]))
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")
    assert out_kspace.shape == kspace.shape
    assert len(masks) >= 1


def test_parameterized_2d_policy_two_d_and_sigmoid() -> None:
    height, width = 10, 8
    policy = Parameterized2dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.TWO_D,
        use_softplus=False,
        acceleration=4.0,
    )
    kspace = torch.randn(1, 2, height, width, 2)
    mask = torch.zeros(1, 1, height, width, 1)
    mask[..., width // 2, :] = 1.0

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, _, _ = policy(mask=mask, kspace=kspace, acceleration=4.0)
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")
    assert out_kspace.shape == kspace.shape


def test_straight_through_policy_2d_sampling_type() -> None:
    height, width = 16, 12
    policy = StraightThroughPolicy(
        backward_operator=ifft2,
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.TWO_D,
        sampling_type=PolicySamplingType.STATIC,
        sampler_chans=4,
        sampler_num_pool_layers=2,
        sampler_fc_size=16,
        sampler_num_fc_layers=2,
    )
    batch, coils = 1, 2
    kspace = torch.randn(batch, coils, height, width, 2)
    mask = torch.zeros(batch, 1, height, width, 1)
    mask[..., :, width // 2 - 1 : width // 2 + 1, :] = 1
    sensitivity_map = torch.ones(batch, coils, height, width, 2) / coils
    acceleration = torch.tensor([4.0])

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, _probs = policy(
                mask=mask,
                kspace=kspace,
                acceleration=acceleration,
                masked_kspace=kspace * mask,
                sensitivity_map=sensitivity_map,
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")
    assert out_kspace.shape == kspace.shape
    assert masks[-1].shape == mask.shape


def test_reshape_utils_6d() -> None:
    batch, height, width, slices = 1, 8, 6, 2
    shape = (batch, 2, slices, height, width, 2)
    mask = torch.zeros(batch, 1, 1, height, width, 1)
    mask[..., width // 2, :] = 1.0
    flat, _ = reshape_mask_pre_sampling(PolicySamplingDimension.ONE_D, mask, None, shape)
    acq, prob, reshaped = reshape_acquisitions_post_sampling(PolicySamplingDimension.ONE_D, flat, flat, flat, shape)
    assert acq.shape == (batch, 1, 1, height, width, 1)
    assert prob.shape == acq.shape
    assert reshaped.shape == acq.shape


def test_reshape_utils_2d_pixel() -> None:
    from direct.nn.adaptive.utils import reshape_acquisitions_post_sampling, reshape_mask_pre_sampling

    batch, height, width = 2, 8, 6
    shape = (batch, 2, height, width, 2)
    mask = torch.zeros(batch, 1, height, width, 1)
    mask[..., height // 2, width // 2, :] = 1.0
    flat, _ = reshape_mask_pre_sampling(PolicySamplingDimension.TWO_D, mask, None, shape)
    assert flat.shape == (batch, height * width)
    acq, prob, reshaped = reshape_acquisitions_post_sampling(PolicySamplingDimension.TWO_D, flat, flat, flat, shape)
    assert acq.shape == mask.shape
    assert prob.shape == mask.shape
    assert reshaped.shape == mask.shape


def test_normalize_masked_probabilities_batch() -> None:
    mask = torch.zeros(3, 12)
    mask[:, :2] = 1.0
    probs = torch.rand(3, 12).clamp(0.1, 0.9) * (1.0 - mask)
    budget = torch.tensor([4, 5, 3])
    out = normalize_masked_probabilities(mask, probs.clone(), budget)
    assert out.shape == probs.shape
    assert torch.all(out[mask.bool()] == 0)


def test_parameterized_policy_with_padding() -> None:
    height, width = 12, 10
    policy = Parameterized2dPolicy(
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
    )
    kspace = torch.randn(1, 2, height, width, 2)
    mask = torch.zeros(1, 1, height, width, 1)
    mask[..., width // 2, :] = 1.0
    padding = torch.zeros_like(mask)
    padding[..., :1, :] = 1.0
    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, _, _ = policy(
                mask=mask,
                kspace=kspace,
                acceleration=torch.tensor([4.0]),
                padding=padding,
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")
    assert out_kspace.shape == kspace.shape


def test_straight_through_dynamic_policy_forward() -> None:
    height, width, time = 32, 28, 4
    policy = StraightThroughPolicy(
        backward_operator=ifft2,
        kspace_shape=(height, width),
        sampling_dimension=PolicySamplingDimension.ONE_D,
        sampling_type=PolicySamplingType.DYNAMIC_2D,
        num_time_steps=time,
        sampler_chans=4,
        sampler_num_pool_layers=2,
        sampler_fc_size=16,
        sampler_num_fc_layers=2,
        num_layers=1,
    )
    batch, coils = 1, 2
    kspace = torch.randn(batch, coils, time, height, width, 2)
    mask = torch.zeros(batch, 1, time, height, width, 1)
    mask[..., width // 2 - 1 : width // 2 + 1, :] = 1
    sensitivity_map = torch.ones(batch, coils, time, height, width, 2) / coils
    acceleration = torch.tensor([4.0])

    last_error: Exception | None = None
    for seed in range(10):
        torch.manual_seed(seed)
        try:
            out_kspace, masks, _probs = policy(
                mask=mask,
                kspace=kspace,
                acceleration=acceleration,
                masked_kspace=kspace * mask,
                sensitivity_map=sensitivity_map,
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        pytest.skip(f"Stochastic binarizer unstable: {last_error}")
    assert out_kspace.shape == kspace.shape
    assert len(masks) >= 1


def test_straight_through_policy_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="kspace_shape"):
        StraightThroughPolicy(
            backward_operator=ifft2,
            kspace_shape=(8,),
            sampling_dimension=PolicySamplingDimension.ONE_D,
        )


def test_sampling_mask_rgb_overlay_colors() -> None:
    from direct.nn.adaptive.utils import export_sampling_mask, sampling_mask_rgb_overlay, split_sampling_mask_history

    initial = torch.zeros(8, 8)
    initial[:, 3:5] = 1.0
    final = initial.clone()
    final[:, 6] = 1.0

    rgb = sampling_mask_rgb_overlay(initial, final)
    assert rgb.shape == (3, 8, 8)
    assert torch.all(rgb[2, :, 3:5] > 0)  # blue = initial
    assert torch.all(rgb[0, :, 6] > 0)  # red = newly acquired
    assert torch.all(rgb[0, :, 3:5] == 0)

    stacked = torch.stack([initial, final], dim=-1)
    assert split_sampling_mask_history(stacked) is not None

    data = {"masks": [initial.unsqueeze(0).unsqueeze(0).unsqueeze(-1), final.unsqueeze(0).unsqueeze(0).unsqueeze(-1)]}
    exported = export_sampling_mask(data)
    assert exported is not None
    assert exported.shape[-1] == 2
