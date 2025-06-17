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
from direct.nn.types import ActivationType, InitType, ModelName


@dataclass
class MRIVarSplitNetConfig(ModelConfig):
    num_steps_reg: int = 8
    num_steps_dc: int = 8
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    kspace_no_parameter_sharing: bool = True
    image_model_architecture: str = ModelName.UNET
    kspace_model_architecture: Optional[str] = None
    image_resnet_hidden_channels: Optional[int] = 128
    image_resnet_num_blocks: Optional[int] = 15
    image_resnet_batchnorm: Optional[bool] = True
    image_resnet_scale: Optional[float] = 0.1
    image_unet_num_filters: Optional[int] = 32
    image_unet_num_pool_layers: Optional[int] = 4
    image_unet_dropout: Optional[float] = 0.0
    image_didn_hidden_channels: Optional[int] = 16
    image_didn_num_dubs: Optional[int] = 6
    image_didn_num_convs_recon: Optional[int] = 9
    kspace_resnet_hidden_channels: Optional[int] = 64
    kspace_resnet_num_blocks: Optional[int] = 1
    kspace_resnet_batchnorm: Optional[bool] = True
    kspace_resnet_scale: Optional[float] = 0.1
    kspace_unet_num_filters: Optional[int] = 16
    kspace_unet_num_pool_layers: Optional[int] = 4
    kspace_unet_dropout: Optional[float] = 0.0
    kspace_didn_hidden_channels: Optional[int] = 8
    kspace_didn_num_dubs: Optional[int] = 6
    kspace_didn_num_convs_recon: Optional[int] = 9
    image_conv_hidden_channels: Optional[int] = 64
    image_conv_n_convs: Optional[int] = 15
    image_conv_activation: Optional[str] = ActivationType.RELU
    image_conv_batchnorm: Optional[bool] = False
    kspace_conv_hidden_channels: Optional[int] = 64
    kspace_conv_n_convs: Optional[int] = 15
    kspace_conv_activation: Optional[str] = ActivationType.PRELU
    kspace_conv_batchnorm: Optional[bool] = False


@dataclass
class MRIVarSplitNet3DConfig(ModelConfig):
    num_steps_reg: int = 8
    num_steps_dc: int = 8
    image_init: InitType = InitType.SENSE
    no_parameter_sharing: bool = True
    image_model_num_filters: int = 16
    image_model_num_pool_layers: int = 4
    image_model_dropout: float = 0.0
