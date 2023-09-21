# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.types import ActivationType, ModelName


@dataclass
class VSharpNetConfig(ModelConfig):
    num_steps: int = 10
    num_steps_dc_gd: int = 8
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    auxiliary_steps: int = 0
    image_model_architecture: ModelName = ModelName.unet
    initializer_channels: tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
    initializer_activation: ActivationType = ActivationType.prelu
    image_resnet_hidden_channels: int = 128
    image_resnet_num_blocks: int = 15
    image_resnet_batchnorm: bool = True
    image_resnet_scale: float = 0.1
    image_unet_num_filters: int = 32
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    image_unet_cwn_conv: bool = False
    image_didn_hidden_channels: int = 16
    image_didn_num_dubs: int = 6
    image_didn_num_convs_recon: int = 9
    image_conv_hidden_channels: int = 64
    image_conv_n_convs: int = 15
    image_conv_activation: str = ActivationType.relu
    image_conv_batchnorm: bool = False
    kspace_model_architecture: Optional[str] = None
    kspace_resnet_hidden_channels: int = 64
    kspace_resnet_num_blocks: int = 1
    kspace_resnet_batchnorm: bool = True
    kspace_resnet_scale: float = 0.1
    kspace_unet_num_filters: int = 16
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout: float = 0.0
    kspace_didn_hidden_channels: int = 8
    kspace_didn_num_dubs: int = 6
    kspace_didn_num_convs_recon: int = 9
    kspace_conv_hidden_channels: int = 64
    kspace_conv_n_convs: int = 15
    kspace_conv_activation: str = ActivationType.prelu
    kspace_conv_batchnorm: bool = False


@dataclass
class VSharpNet3DConfig(ModelConfig):
    num_steps: int = 8
    num_steps_dc_gd: int = 6
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    auxiliary_steps: int = -1
    initializer_channels: tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
    initializer_activation: ActivationType = ActivationType.prelu
    unet_num_filters: int = 32
    unet_num_pool_layers: int = 4
    unet_dropout: float = 0.0
    unet_cwn_conv: bool = False
    unet_norm: bool = False
