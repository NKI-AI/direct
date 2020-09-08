# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from direct.config.defaults import ModelConfig


@dataclass
class UnetModel2dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
