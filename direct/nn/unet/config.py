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
from direct.nn.conv.modulated import ModConvActivation, ModConvType
from direct.nn.types import InitType


@dataclass
class UnetModel2dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    modulation: ModConvType = ModConvType.NONE
    aux_in_features: int | None = None
    fc_hidden_features: int | None = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: int | None = None


class NormUnetModel2dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    norm_groups: int = 2
    modulation: ModConvType = ModConvType.NONE
    aux_in_features: int | None = None
    fc_hidden_features: int | None = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: int | None = None


@dataclass
class Unet2dConfig(ModelConfig):
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    skip_connection: bool = False
    normalized: bool = False
    image_initialization: InitType = InitType.ZERO_FILLED
    conv_modulation: ModConvType = ModConvType.NONE
    aux_in_features: int | None = None
    auxiliary_features: tuple[str, ...] | None = None
    log_aux: bool = False
    fc_hidden_features: tuple[int, ...] | None = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: int | None = None


@dataclass
class UnetModel3dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0


class NormUnetModel3dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    norm_groups: int = 2
