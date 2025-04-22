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
from direct.nn.conjgradnet.conjgrad import CGUpdateType
from direct.nn.types import ActivationType, InitType, ModelName


@dataclass
class ConjGradNetConfig(ModelConfig):
    num_steps: int = 8
    image_init: str = InitType.ZEROS
    no_parameter_sharing: bool = True
    cg_tol: float = 1e-7
    cg_iters: int = 10
    cg_param_update_type: str = CGUpdateType.FR
    denoiser_architecture: str = ModelName.RESNET
    resnet_hidden_channels: int = 128
    resnet_num_blocks: int = 15
    resenet_batchnorm: bool = True
    resenet_scale: Optional[float] = 0.1
    unet_num_filters: Optional[int] = 32
    unet_num_pool_layers: Optional[int] = 4
    unet_dropout: Optional[float] = 0.0
    didn_hidden_channels: Optional[int] = 16
    didn_num_dubs: Optional[int] = 6
    didn_num_convs_recon: Optional[int] = 9
    conv_hidden_channels: Optional[int] = 64
    conv_n_convs: Optional[int] = 15
    conv_activation: Optional[str] = ActivationType.RELU
    conv_batchnorm: Optional[bool] = False
