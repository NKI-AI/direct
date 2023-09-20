# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.types import ActivationType, ModelName


@dataclass
class VSharpNetConfig(ModelConfig):
    num_steps: int = 10
    num_steps_dc_gd: int = 8
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    auxiliary_steps: int = 0
    image_model_architecture: ModelName = ModelName.unet
    initializer_channels: tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
    initializer_activation: ActivationType = ActivationType.prelu
    image_resnet_hidden_channels: int = 128
    image_resnet_num_blocks: int = 15
    image_resnet_batchnorm: bool = True
    image_resnet_scale: float = 0.1
    image_unet_num_filters: int = 32
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    image_unet_cwn_conv: bool = False
    image_didn_hidden_channels: int = 16
    image_didn_num_dubs: int = 6
    image_didn_num_convs_recon: int = 9
    image_conv_hidden_channels: int = 64
    image_conv_n_convs: int = 15
    image_conv_activation: str = ActivationType.relu
    image_conv_batchnorm: bool = False
    image_uformer_patch_size: int = 256
    image_uformer_embedding_dim: int = 32
    image_uformer_encoder_depths: tuple[int, ...] = (2, 2, 2, 2)
    image_uformer_encoder_num_heads: tuple[int, ...] = (1, 2, 4, 8)
    image_uformer_bottleneck_depth: int = 2
    image_uformer_bottleneck_num_heads: int = 16
    image_uformer_win_size: int = 8
    image_uformer_mlp_ratio: float = 4.0
    image_uformer_qkv_bias: bool = True
    image_uformer_qk_scale: Optional[float] = None
    image_uformer_drop_rate: float = 0.0
    image_uformer_attn_drop_rate: float = 0.0
    image_uformer_drop_path_rate: float = 0.1
    image_uformer_patch_norm: bool = True
    image_uformer_shift_flag: bool = True
    image_uformer_modulator: bool = False
    image_uformer_cross_modulator: bool = False
    image_uformer_normalized: bool = True
    kspace_model_architecture: Optional[str] = None
    kspace_resnet_hidden_channels: int = 64
    kspace_resnet_num_blocks: int = 1
    kspace_resnet_batchnorm: bool = True
    kspace_resnet_scale: float = 0.1
    kspace_unet_num_filters: int = 16
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout: float = 0.0
    kspace_didn_hidden_channels: int = 8
    kspace_didn_num_dubs: int = 6
    kspace_didn_num_convs_recon: int = 9
    kspace_conv_hidden_channels: int = 64
    kspace_conv_n_convs: int = 15
    kspace_conv_activation: str = ActivationType.prelu
    kspace_conv_batchnorm: bool = False
    kspace_uformer_patch_size: int = 256
    kspace_uformer_embedding_dim: int = 32
    kspace_uformer_encoder_depths: tuple[int, ...] = (2, 2, 2)
    kspace_uformer_encoder_num_heads: tuple[int, ...] = (1, 2, 4)
    kspace_uformer_bottleneck_depth: int = 2
    kspace_uformer_bottleneck_num_heads: int = 8
    kspace_uformer_win_size: int = 8
    kspace_uformer_mlp_ratio: float = 4.0
    kspace_uformer_qkv_bias: bool = True
    kspace_uformer_qk_scale: Optional[float] = None
    kspace_uformer_drop_rate: float = 0.0
    kspace_uformer_attn_drop_rate: float = 0.0
    kspace_uformer_drop_path_rate: float = 0.1
    kspace_uformer_patch_norm: bool = True
    kspace_uformer_shift_flag: bool = True
    kspace_uformer_modulator: bool = False
    kspace_uformer_cross_modulator: bool = False
    kspace_uformer_normalized: bool = True
    image_vision_transformer_average_img_size: int = 320
    image_vision_transformer_patch_size: int = 10
    image_vision_transformer_embedding_dim: int = 64
    image_vision_transformer_depth: int = 8
    image_vision_transformer_num_heads: int = 9
    image_vision_transformer_mlp_ratio: float = 4.0
    image_vision_transformer_qkv_bias: bool = False
    image_vision_transformer_qk_scale: Optional[float] = None
    image_vision_transformer_drop_rate: float = (0.0,)
    image_vision_transformer_attn_drop_rate: float = (0.0,)
    image_vision_transformer_dropout_path_rate: float = 0.0
    image_vision_transformer_gpsa_interval: tuple[int, int] = (-1, -1)
    image_vision_transformer_locality_strength: float = 1.0
    image_vision_transformer_use_pos_embedding: bool = True
    image_vision_transformer_normalized: bool = True
    kspace_vision_transformer_average_img_size: int = 320
    kspace_vision_transformer_patch_size: int = 10
    kspace_vision_transformer_embedding_dim: int = 64
    kspace_vision_transformer_depth: int = 8
    kspace_vision_transformer_num_heads: int = 9
    kspace_vision_transformer_mlp_ratio: float = 4.0
    kspace_vision_transformer_qkv_bias: bool = False
    kspace_vision_transformer_qk_scale: Optional[float] = None
    kspace_vision_transformer_drop_rate: float = (0.0,)
    kspace_vision_transformer_attn_drop_rate: float = (0.0,)
    kspace_vision_transformer_dropout_path_rate: float = 0.0
    kspace_vision_transformer_gpsa_interval: tuple[int, int] = (-1, -1)
    kspace_vision_transformer_locality_strength: float = 1.0
    kspace_vision_transformer_use_pos_embedding: bool = True
    kspace_vision_transformer_normalized: bool = True


@dataclass
class VSharpNet3DConfig(ModelConfig):
    num_steps: int = 8
    num_steps_dc_gd: int = 6
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    auxiliary_steps: int = -1
    initializer_channels: tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: tuple[int, ...] = (1, 1, 2, 4)
    initializer_multiscale: int = 1
    initializer_activation: ActivationType = ActivationType.prelu
    unet_num_filters: int = 32
    unet_num_pool_layers: int = 4
    unet_dropout: float = 0.0
    unet_cwn_conv: bool = False
    unet_norm: bool = False
