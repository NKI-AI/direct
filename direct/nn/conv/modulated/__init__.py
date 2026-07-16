# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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
"""Modulated convolution layers and auxiliary conditioning utilities."""

from direct.nn.conv.modulated.auxiliary_data import (
    AUXILIARY_FEATURE_REGISTRY,
    DEFAULT_AUXILIARY_FEATURE_NAMES,
    AuxiliaryFeature,
    ModulationConfig,
    prepare_auxiliary_data,
    register_auxiliary_feature,
    resolve_auxiliary_features,
)
from direct.nn.conv.modulated.factory import (
    ModulationParams,
    mod_conv2d,
    mod_conv3d,
    mod_conv_transpose2d,
    mod_conv_transpose3d,
)
from direct.nn.conv.modulated.modulated_conv import (
    ModConv2d,
    ModConv2dBias,
    ModConv3d,
    ModConvActivation,
    ModConvTranspose2d,
    ModConvTranspose3d,
    ModConvType,
)

__all__ = [
    "ModulationParams",
    "mod_conv2d",
    "mod_conv3d",
    "mod_conv_transpose2d",
    "mod_conv_transpose3d",
    "AUXILIARY_FEATURE_REGISTRY",
    "AuxiliaryFeature",
    "DEFAULT_AUXILIARY_FEATURE_NAMES",
    "ModConv2d",
    "ModConv2dBias",
    "ModConv3d",
    "ModConvActivation",
    "ModConvTranspose2d",
    "ModConvTranspose3d",
    "ModConvType",
    "ModulationConfig",
    "prepare_auxiliary_data",
    "register_auxiliary_feature",
    "resolve_auxiliary_features",
]
