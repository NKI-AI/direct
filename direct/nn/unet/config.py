# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.conv.modulated_conv import ModConvActivation


@dataclass
class UnetModel2dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    modulation: bool = False
    aux_in_features: Optional[int] = None
    fc_hidden_features: Optional[int] = None
    fc_groups: Optional[int] = None
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID


class NormUnetModel2dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    norm_groups: int = 2
    modulation: bool = False
    aux_in_features: Optional[int] = None
    fc_hidden_features: Optional[int] = None
    fc_groups: Optional[int] = None
    fc_activation: ModConvActivation = ModConvActivation.SIGMOID


@dataclass
class UnetModel3dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0


class NormUnetModel3dConfig(ModelConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    norm_groups: int = 2


@dataclass
class Unet2dConfig(ModelConfig):
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
    skip_connection: bool = False
    normalized: bool = False
    image_initialization: str = "zero_filled"
