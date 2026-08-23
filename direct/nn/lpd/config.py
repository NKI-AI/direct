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
"""direct.nn.lpd.config module."""

from dataclasses import dataclass

from direct.config.defaults import ModelConfig
from direct.nn.conv.modulated import ModConvActivation, ModConvType


@dataclass
class LPDNetConfig(ModelConfig):
    """LPDNetConfig."""

    num_iter: int = 25
    num_primal: int = 5
    num_dual: int = 5
    primal_model_architecture: str = "MWCNN"
    dual_model_architecture: str = "DIDN"
    primal_mwcnn_hidden_channels: int = 16
    primal_mwcnn_num_scales: int = 4
    primal_mwcnn_bias: bool = True
    primal_mwcnn_batchnorm: bool = False
    primal_unet_num_filters: int = 8
    primal_unet_num_pool_layers: int = 4
    primal_unet_dropout_probability: float = 0.0
    dual_conv_hidden_channels: int = 16
    dual_conv_n_convs: int = 4
    dual_conv_batchnorm: bool = False
    dual_didn_hidden_channels: int = 64
    dual_didn_num_dubs: int = 6
    dual_didn_num_convs_recon: int = 9
    dual_unet_num_filters: int = 8
    dual_unet_num_pool_layers: int = 4
    dual_unet_dropout_probability: float = 0.0
    conv_modulation: ModConvType = ModConvType.NONE
    aux_in_features: int | None = None
    auxiliary_features: tuple[str, ...] | None = None
    log_aux: bool = False
    fc_hidden_features: tuple[int, ...] | None = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: int | None = None
