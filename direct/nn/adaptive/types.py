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

"""Enumerations for adaptive k-space sampling policies."""

from direct.types import DirectEnum


class PolicySamplingDimension(DirectEnum):
    """Supported k-space sampling dimensions."""

    ONE_D = "1D"
    TWO_D = "2D"


class PolicySamplingType(DirectEnum):
    """Supported adaptive sampling strategies."""

    STATIC = "static"
    DYNAMIC_2D = "dynamic_2d"
    DYNAMIC_2D_NON_UNIFORM = "dynamic_2d_non_uniform"
    MULTISLICE_2D = "multislice_2d"
    MULTISLICE_2D_NON_UNIFORM = "multislice_2d_non_uniform"
