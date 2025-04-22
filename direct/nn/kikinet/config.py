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
class KIKINetConfig(ModelConfig):
    num_iter: int = 10
    image_model_architecture: str = "MWCNN"
    kspace_model_architecture: str = "UNET"
    image_mwcnn_hidden_channels: int = 16
    image_mwcnn_num_scales: int = 4
    image_mwcnn_bias: bool = True
    image_mwcnn_batchnorm: bool = False
    image_unet_num_filters: int = 8
    image_unet_num_pool_layers: int = 4
    image_unet_dropout_probability: float = 0.0
    kspace_conv_hidden_channels: int = 16
    kspace_conv_n_convs: int = 4
    kspace_conv_batchnorm: bool = False
    kspace_didn_hidden_channels: int = 64
    kspace_didn_num_dubs: int = 6
    kspace_didn_num_convs_recon: int = 9
    kspace_unet_num_filters: int = 8
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout_probability: float = 0.0
    normalize: bool = False
