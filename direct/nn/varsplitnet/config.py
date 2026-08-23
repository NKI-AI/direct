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

"""direct.nn.varsplitnet.config module."""

from dataclasses import dataclass

from direct.config.defaults import ModelConfig
from direct.nn.types import ActivationType, ModelName


@dataclass
class MRIVarSplitNetConfig(ModelConfig):
    """MRIVarSplitNetConfig."""

    num_steps_reg: int = 8
    num_steps_dc: int = 8
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    kspace_no_parameter_sharing: bool = True
    image_model_architecture: str = ModelName.UNET
    kspace_model_architecture: str | None = None
    image_resnet_hidden_channels: int | None = 128
    image_resnet_num_blocks: int | None = 15
    image_resnet_batchnorm: bool | None = True
    image_resnet_scale: float | None = 0.1
    image_unet_num_filters: int | None = 32
    image_unet_num_pool_layers: int | None = 4
    image_unet_dropout: float | None = 0.0
    image_didn_hidden_channels: int | None = 16
    image_didn_num_dubs: int | None = 6
    image_didn_num_convs_recon: int | None = 9
    kspace_resnet_hidden_channels: int | None = 64
    kspace_resnet_num_blocks: int | None = 1
    kspace_resnet_batchnorm: bool | None = True
    kspace_resnet_scale: float | None = 0.1
    kspace_unet_num_filters: int | None = 16
    kspace_unet_num_pool_layers: int | None = 4
    kspace_unet_dropout: float | None = 0.0
    kspace_didn_hidden_channels: int | None = 8
    kspace_didn_num_dubs: int | None = 6
    kspace_didn_num_convs_recon: int | None = 9
    image_conv_hidden_channels: int | None = 64
    image_conv_n_convs: int | None = 15
    image_conv_activation: str | None = ActivationType.RELU
    image_conv_batchnorm: bool | None = False
    kspace_conv_hidden_channels: int | None = 64
    kspace_conv_n_convs: int | None = 15
    kspace_conv_activation: str | None = ActivationType.PRELU
    kspace_conv_batchnorm: bool | None = False
