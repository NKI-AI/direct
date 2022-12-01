# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Tuple

from direct.config.defaults import ModelConfig


@dataclass
class DenseNetConfig(ModelConfig):
    in_channels: int = 1
    num_classes: int = 1000
    num_layers: Tuple[int, ...] = (6, 6, 6, 6)
    bottleneck_channels: int = 2
    expansion: int = 8
    activation: str = "relu"
