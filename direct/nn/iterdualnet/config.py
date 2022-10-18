# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class IterDualNetConfig(ModelConfig):
    num_iter: int = 10
    use_norm_unet: bool = False
    image_unet_num_filters: int = 8
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    kspace_unet_num_filters: int = 8
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout: float = 0.0
    no_parameter_sharing: bool = True
    compute_per_coil: bool = True


@dataclass
class IterDualNetSSLConfig(IterDualNetConfig):
    pass
