# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional, Tuple

from direct.config.defaults import ModelConfig


@dataclass
class RecurrentVarNetConfig(ModelConfig):
    num_steps: int = 15
    recurrent_hidden_channels: int = 64
    recurrent_num_layers: int = 4
    no_parameter_sharing: bool = True
    learned_initializer: bool = True
    initializer_initialization: Optional[str] = "zero_filled"
    initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64)
    initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
