# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.types import ActivationType, ModelName


@dataclass
class MRIVarSplitNetConfig(ModelConfig):
    num_steps_reg: int = 8
    num_steps_dc: int = 8
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    kspace_no_parameter_sharing: bool = True
    image_model_architecture: str = ModelName.unet
    kspace_model_architecture: Optional[str] = None
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
    image_conv_hidden_channels: int = 64
    image_conv_n_convs: int = 15
    image_conv_activation: str = ActivationType.relu
    image_conv_batchnorm: bool = False
    kspace_conv_hidden_channels: int = 64
    kspace_conv_n_convs: int = 15
    kspace_conv_activation: str = ActivationType.prelu
    kspace_conv_batchnorm: bool = False
