# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from typing import Tuple, List, Any

from direct.config.defaults import ModelConfig


@dataclass
class RIMConfig(ModelConfig):
    hidden_channels: int = 16
    length: int = 8
    depth: int = 2
    steps: int = 1
    no_parameter_sharing: bool = False
    instance_norm: bool = False
    dense_connect: bool = False
    replication_padding: bool = True
    learned_initializer: bool = False
    initializer_channels: Tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: Tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1


@dataclass
class RIM3dConfig(ModelConfig):
    hidden_channels: int = 16
    length: int = 8
    depth: int = 2
    steps: int = 1
    no_parameter_sharing: bool = False
    instance_norm: bool = False
    dense_connect: bool = False
    replication_padding: bool = True
    # learned_initializer: bool = False
    # initializer_channels: Tuple[int, ...] = (32, 32, 64, 64)
    # initializer_dilations: Tuple[int, ...] = (1, 1, 2, 4)
    # initializer_multiscale: int = 1
    z_reduction_frequency: int = 0
