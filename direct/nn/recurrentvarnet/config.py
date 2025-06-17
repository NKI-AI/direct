# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
from dataclasses import dataclass
from typing import Optional, tuple

from direct.config.defaults import ModelConfig
from direct.nn.types import InitType


@dataclass
class RecurrentVarNetConfig(ModelConfig):
    num_steps: int = 15  # :math:`T`
    recurrent_hidden_channels: int = 64
    recurrent_num_layers: int = 4  # :math:`n_l`
    no_parameter_sharing: bool = True
    learned_initializer: bool = True
    initializer_initialization: Optional[str] = InitType.SENSE
    initializer_channels: Optional[tuple[int, ...]] = (32, 32, 64, 64)  # :math:`n_d`
    initializer_dilations: Optional[tuple[int, ...]] = (1, 1, 2, 4)  # :math:`p`
    initializer_multiscale: int = 1
    normalized: bool = False


@dataclass
class RecurrentVarNet3dConfig(ModelConfig):
    num_steps: int = 10  # :math:`T`
    recurrent_hidden_channels: int = 64
    recurrent_num_layers: int = 4  # :math:`n_l`
    no_parameter_sharing: bool = True
    learned_initializer: bool = True
    initializer_initialization: Optional[str] = InitType.SENSE
    initializer_channels: Optional[tuple[int, ...]] = (32, 32, 64, 64)  # :math:`n_d`
    initializer_dilations: Optional[tuple[int, ...]] = (1, 1, 2, 4)  # :math:`p`
    initializer_multiscale: int = 1
    normalized: bool = False
