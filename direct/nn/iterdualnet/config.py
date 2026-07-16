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
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.conv.modulated import ModConvActivation, ModConvType


@dataclass
class IterDualNetConfig(ModelConfig):
    num_iter: int = 10
    image_normunet: bool = False
    kspace_normunet: bool = False
    image_unet_num_filters: int = 8
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    kspace_unet_num_filters: int = 8
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout: float = 0.0
    image_no_parameter_sharing: bool = True
    kspace_no_parameter_sharing: bool = False
    compute_per_coil: bool = True
    conv_modulation: ModConvType = ModConvType.NONE
    aux_in_features: Optional[int] = None
    auxiliary_features: Optional[tuple[str, ...]] = None
    log_aux: bool = False
    fc_hidden_features: Optional[tuple[int, ...]] = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: Optional[int] = None
