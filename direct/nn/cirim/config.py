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

from direct.config.defaults import ModelConfig


@dataclass
class CIRIMConfig(ModelConfig):
    time_steps: int = 8  # :math:`T`
    depth: int = 2
    recurrent_hidden_channels: int = 64
    num_cascades: int = 8
    no_parameter_sharing: bool = True
