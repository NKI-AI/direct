# Copyright (c) DIRECT Contributors

"""Contains the configuration of MEDL models."""


from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class MEDLConfig(ModelConfig):
    iterations: int = (4,)
    num_layers: int = 3
    unet_num_filters: int = 18
    unet_num_pool_layers: int = 4
    unet_dropout: float = 0.0
    unet_norm: bool = False


@dataclass
class MEDL2DConfig(MEDLConfig):
    pass


@dataclass
class MEDL3DConfig(MEDLConfig):
    pass
