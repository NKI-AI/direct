# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration dataclasses for registration models in :mod:`direct.nn.registration`."""

from dataclasses import dataclass

from direct.config.defaults import ModelConfig
from direct.registration.demons import DemonsFilterType


@dataclass
class RegistrationModelConfig(ModelConfig):
    """Base configuration shared by all registration models."""

    warp_num_integration_steps: int = 1
    train_end_to_end: bool = True
    decoupled_training: bool = False
    rec_loss_factor: float = 1.0
    reg_loss_factor: float = 1.0
    # If True, also apply predicted DF to GT moving (`target`) and add photometric loss vs reference.
    reg_loss_on_target: bool = False


@dataclass
class OpticalFlowILKRegistration2dModelConfig(RegistrationModelConfig):
    """Configuration for iterative Lucas-Kanade optical-flow registration."""

    radius: int = 5
    num_warp: int = 3
    gaussian: bool = False
    prefilter: bool = True


@dataclass
class OpticalFlowTVL1Registration2dModelConfig(RegistrationModelConfig):
    """Configuration for TV-L1 optical-flow registration."""

    attachment: float = 15
    tightness: float = 0.3
    num_warp: int = 3
    num_iter: int = 5
    tol: float = 1e-2
    prefilter: bool = False


@dataclass
class DemonsRegistration2dModelConfig(RegistrationModelConfig):
    """Configuration for SimpleITK demons registration."""

    demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES
    demons_num_iterations: int = 50
    demons_smooth_displacement_field: bool = True
    demons_standard_deviations: float = 1.0
    demons_intensity_difference_threshold: float | None = None
    demons_maximum_rms_error: float | None = None


@dataclass
class UnetRegistration2dModelConfig(RegistrationModelConfig):
    """Configuration for UNet-based dense displacement-field prediction."""

    max_seq_len: int = 12
    unet_num_filters: int = 16
    unet_num_pool_layers: int = 4
    unet_dropout_probability: float = 0.0
    unet_normalized: bool = False


@dataclass
class VxmDenseConfig(RegistrationModelConfig):
    """Configuration for the VoxelMorph dense registration network."""

    inshape: tuple = (512, 246)
    nb_unet_features: int = 16
    nb_unet_levels: int = 4
    nb_unet_conv_per_level: int = 1
    int_downsize: int = 2


@dataclass
class ViTRegistration2dModelConfig(RegistrationModelConfig):
    """Configuration for Vision Transformer-based dense displacement-field prediction."""

    max_seq_len: int = 12
    average_size: tuple[int, int] = (320, 320)
    patch_size: tuple[int, int] = (16, 16)
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = 9
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    gpsa_interval: tuple[int, int] = (-1, -1)
    locality_strength: float = 1.0
    use_pos_embedding: bool = True


UnetRegistrationModelConfig = UnetRegistration2dModelConfig
