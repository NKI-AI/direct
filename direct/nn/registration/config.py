"""Configuration for the UnetRegistrationModel."""

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class UnetRegistrationModelConfig(ModelConfig):
    max_seq_len: int = 12
    unet_num_filters: int = 16
    unet_num_pool_layers: int = 4
    unet_dropout_probability: float = 0.0
    unet_normalized: bool = False
