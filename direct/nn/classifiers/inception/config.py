# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class InceptionConfig(ModelConfig):
    in_channels: int = 1
    num_classes: int = 1000
    hidden_channels: int = 8
    activation_name: str = "relu"
