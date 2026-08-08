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

"""Configuration dataclasses for adaptive sampling policies in :mod:`direct.nn.adaptive`."""

from dataclasses import dataclass

from omegaconf import MISSING

from direct.config.defaults import ModelConfig
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.types import ActivationType


@dataclass
class PolicyConfig(ModelConfig):
    """Base configuration shared by all adaptive sampling policies."""

    sampling_dimension: PolicySamplingDimension = MISSING
    st_slope: float = 10
    st_clamp: bool = False
    use_softplus: bool = True
    slope: float = 10
    fix_sign_leakage: bool = True
    acceleration: float | None = None


@dataclass
class ParameterizedPolicyConfig(PolicyConfig):
    """Configuration for learnable parameterized sampling policies."""

    kspace_shape: tuple[int, ...] = MISSING


@dataclass
class Parameterized2dPolicyConfig(ParameterizedPolicyConfig):
    """Configuration for parameterized 2D sampling policies."""


@dataclass
class Parameterized3dPolicyConfig(ParameterizedPolicyConfig):
    """Configuration for parameterized 3D sampling policies."""


@dataclass
class ParameterizedDynamic2dPolicyConfig(ParameterizedPolicyConfig):
    """Configuration for parameterized dynamic 2D sampling policies."""

    num_time_steps: int = MISSING
    non_uniform: bool = False


@dataclass
class ParameterizedMultislice2dPolicyConfig(ParameterizedPolicyConfig):
    """Configuration for parameterized multislice 2D sampling policies."""

    num_slices: int = MISSING
    non_uniform: bool = False


@dataclass
class StraightThroughPolicyConfig(PolicyConfig):
    """Configuration for straight-through estimator sampling policies."""

    kspace_shape: tuple[int, ...] = MISSING
    num_layers: int = 2
    kspace_sampler: bool = False
    sampler_detach_mask: bool = False
    sampler_chans: int = 16
    sampler_num_pool_layers: int = 4
    sampler_fc_size: int = 256
    sampler_drop_prob: float = 0
    sampler_num_fc_layers: int = 3
    sampler_activation: ActivationType = ActivationType.LEAKY_RELU
    sampling_type: PolicySamplingType = PolicySamplingType.STATIC
    num_time_steps: int | None = None
    num_slices: int | None = None
