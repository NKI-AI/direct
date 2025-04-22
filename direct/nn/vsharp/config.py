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

from __future__ import annotations

from dataclasses import dataclass

from direct.config.defaults import ModelConfig
from direct.nn.types import ActivationType, InitType, ModelName


@dataclass
class VSharpNetConfig(ModelConfig):
    num_steps: int = 10
    num_steps_dc_gd: int = 8
    image_init: InitType = InitType.SENSE
    no_parameter_sharing: bool = True
    auxiliary_steps: int = 0
    image_model_architecture: ModelName = ModelName.UNET
    initializer_channels: tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
    initializer_activation: ActivationType = ActivationType.PRELU
    image_resnet_hidden_channels: int = 128
    image_resnet_num_blocks: int = 15
    image_resnet_batchnorm: bool = True
    image_resnet_scale: float = 0.1
    image_unet_num_filters: int = 32
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    image_didn_hidden_channels: int = 16
    image_didn_num_dubs: int = 6
    image_didn_num_convs_recon: int = 9
    image_conv_hidden_channels: int = 64
    image_conv_n_convs: int = 15
    image_conv_activation: str = ActivationType.RELU
    image_conv_batchnorm: bool = False


@dataclass
class VSharpNet3DConfig(ModelConfig):
    num_steps: int = 8
    num_steps_dc_gd: int = 6
    image_init: InitType = InitType.SENSE
    no_parameter_sharing: bool = True
    auxiliary_steps: int = -1
    initializer_channels: tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
    initializer_activation: ActivationType = ActivationType.PRELU
    unet_num_filters: int = 32
    unet_num_pool_layers: int = 4
    unet_dropout: float = 0.0
    unet_norm: bool = False
