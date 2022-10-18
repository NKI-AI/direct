# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig


@dataclass
class ResNetConfig(ModelConfig):
    in_channels: int = 2
    out_channels: Optional[int] = None
    hidden_channels: int = 32
    num_blocks: int = 15
    batchnorm: bool = True
    scale: Optional[float] = 0.1
    image_init: str = "sense"


@dataclass
class MRIResNetConjGradConfig(ModelConfig):
    num_steps: int = 8
    resnet_hidden_channels: int = 128
    resnet_num_blocks: int = 15
    resenet_batchnorm: bool = True
    resenet_scale: Optional[float] = 0.1
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    cg_iters: int = 10
