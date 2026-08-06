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

"""Contains the configuration of MEDL models."""

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class MEDLConfig(ModelConfig):
    """Base configuration shared by MEDL reconstruction models."""

    iterations: int = 4
    num_layers: int = 3
    unet_num_filters: int = 18
    unet_num_pool_layers: int = 4
    unet_dropout: float = 0.0
    unet_norm: bool = False


@dataclass
class MEDL2DConfig(MEDLConfig):
    """Configuration for 2D MEDL models."""


@dataclass
class MEDL3DConfig(MEDLConfig):
    """Configuration for 3D MEDL models."""
