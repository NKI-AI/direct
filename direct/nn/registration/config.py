"""Configuration for the UnetRegistrationModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.registration.demons import DemonsFilterType


@dataclass
class RegistrationModelConfig(ModelConfig):
    warp_num_integration_steps: int = 1
    train_end_to_end: bool = False
    # TODO: Needs to be defined outside of the config
    reg_loss_factor: float = 1.0  # Regularization loss weight factor


@dataclass
class OpticalFlowILKRegistration2dModelConfig(RegistrationModelConfig):
    radius: int = 5
    num_warp: int = 3
    gaussian: bool = False
    prefilter: bool = True


@dataclass
class OpticalFlowTVL1Registration2dModelConfig(RegistrationModelConfig):
    attachment: float = 15
    tightness: float = 0.3
    num_warp: int = 3
    num_iter: int = 5
    tol: float = 1e-2
    prefilter: bool = False


@dataclass
class DemonsRegistration2dModelConfig(RegistrationModelConfig):
    demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES
    demons_num_iterations: int = 50
    demons_smooth_displacement_field: bool = True
    demons_standard_deviations: float = 1.0
    demons_intensity_difference_threshold: Optional[float] = None
    demons_maximum_rms_error: Optional[float] = None


@dataclass
class UnetRegistration2dModelConfig(RegistrationModelConfig):
    max_seq_len: int = 12
    unet_num_filters: int = 16
    unet_num_pool_layers: int = 4
    unet_dropout_probability: float = 0.0
    unet_normalized: bool = False


@dataclass
class VxmDenseConfig(RegistrationModelConfig):
    inshape: tuple = (512, 246)
    nb_unet_features: int = 16
    nb_unet_levels: int = 4
    nb_unet_conv_per_level: int = 1
    int_downsize: int = 2


@dataclass
class ViTRegistration2dModelConfig(RegistrationModelConfig):
    max_seq_len: int = 12
    average_size: tuple[int, int] = (320, 320)
    patch_size: tuple[int, int] = (16, 16)
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = 9
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    gpsa_interval: tuple[int, int] = (-1, -1)
    locality_strength: float = 1.0
    use_pos_embedding: bool = True
