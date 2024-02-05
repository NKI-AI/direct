# Copyright (c) DIRECT Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.conv.modulated_conv import ModConvActivation, ModConvType
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
    conv_modulation: ModConvType = ModConvType.NONE
    aux_in_features: int = 2
    fc_hidden_features: Optional[int] = None
    fc_groups: int = 1
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID
    num_weights: Optional[int] = None
    modulation_at_input: bool = False
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
