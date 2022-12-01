# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class ResNetConfig(ModelConfig):
    in_channels: int = 1
    num_classes: int = 1000


@dataclass
class ResNet18Config(ResNetConfig):
    pass


@dataclass
class ResNet34Config(ResNetConfig):
    pass


@dataclass
class ResNet50Config(ResNetConfig):
    pass


@dataclass
class ResNet101Config(ResNetConfig):
    pass


@dataclass
class ResNet151Config(ResNetConfig):
    pass
