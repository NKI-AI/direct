# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from typing import Tuple, List, Any

from direct.config.defaults import ModelConfig


@dataclass
class Unet2DConfig(ModelConfig):
    in_channels: int = 1
    out_channels: int = 1
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
