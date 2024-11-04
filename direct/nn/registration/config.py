"""Configuration for the UnetRegistrationModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.registration.demons import DemonsFilterType


@dataclass
class RegistrationModelConfig(ModelConfig):
    warp_num_integration_steps: int = 1


@dataclass
class OpticalFlowILKRegistration2dModelConfig(RegistrationModelConfig):
    radius: int = 7
    num_warp: int = 10
    gaussian: bool = False
    prefilter: bool = True


@dataclass
class OpticalFlowTVL1Registration2dModelConfig(RegistrationModelConfig):
    attachment: float = 15
    tightness: float = 0.3
    num_warp: int = 5
    num_iter: int = 10
    tol: float = 1e-3
    prefilter: bool = True


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
    train_end_to_end: bool = True


@dataclass
class VxmDenseModelConfig(RegistrationModelConfig):
    inshape: tuple = (512, 246)
    nb_unet_features: int = 16
    nb_unet_levels: int = 4
    nb_unet_conv_per_level: int = 1
    int_downsize: int = 2
