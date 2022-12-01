# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class VGGConfig(ModelConfig):
    in_channels: int = 1
    num_classes: int = 1000
    batchnorm: bool = False


@dataclass
class VGG11Config(VGGConfig):
    pass


@dataclass
class VGG13Config(VGGConfig):
    pass


@dataclass
class VGG16Config(VGGConfig):
    pass


@dataclass
class VGG19Config(VGGConfig):
    pass
