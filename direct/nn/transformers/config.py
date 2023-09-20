# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional, Tuple

from direct.config.defaults import ModelConfig


@dataclass
class UFormerModelConfig(ModelConfig):
    in_channels: int = 2
    out_channels: Optional[int] = None
    patch_size: int = 128
    embedding_dim: int = 16
    encoder_depths: tuple[int, ...] = (2, 2, 2)
    encoder_num_heads: tuple[int, ...] = (1, 2, 4)
    bottleneck_depth: int = 2
    bottleneck_num_heads: int = 8
    win_size: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    patch_norm: bool = True
    token_projection: str = "linear"
    token_mlp: str = "leff"
    shift_flag: bool = True
    modulator: bool = False
    cross_modulator: bool = False
    normalized: bool = True


@dataclass
class VariationalUFormerConfig(ModelConfig):
    num_steps: int = 5
    patch_size: int = 128
    embedding_dim: int = 16
    encoder_depths: tuple[int, ...] = (2, 2, 2)
    encoder_num_heads: tuple[int, ...] = (1, 2, 4)
    bottleneck_depth: int = 2
    bottleneck_num_heads: int = 8
    win_size: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    patch_norm: bool = True
    token_projection: str = "linear"
    token_mlp: str = "leff"
    shift_flag: bool = True
    modulator: bool = False
    cross_modulator: bool = False
    no_weight_sharing: bool = True


@dataclass
class MRIUFormerConfig(ModelConfig):
    patch_size: int = 128
    embedding_dim: int = 16
    encoder_depths: tuple[int, ...] = (2, 2, 2, 2)
    encoder_num_heads: tuple[int, ...] = (1, 2, 4, 8)
    bottleneck_depth: int = 2
    bottleneck_num_heads: int = 16
    win_size: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    patch_norm: bool = True
    token_projection: str = "linear"
    token_mlp: str = "leff"
    shift_flag: bool = True
    modulator: bool = False
    cross_modulator: bool = False


@dataclass
class ImageDomainUFormerConfig(MRIUFormerConfig):
    pass


@dataclass
class KSpaceDomainUFormerConfig(MRIUFormerConfig):
    multicoil_input_mode: str = "sense_sum"
    patch_size: int = 64
    embedding_dim: int = 16
    encoder_depths: tuple[int, ...] = (2, 2, 2)
    encoder_num_heads: tuple[int, ...] = (1, 2, 4)
    bottleneck_depth: int = 2
    bottleneck_num_heads: int = 8


@dataclass
class MRITransformerConfig(ModelConfig):
    num_gradient_descent_steps: int = 8
    average_img_size: int = 320
    patch_size: int = 10
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = 9
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    gpsa_interval: Tuple[int, int] = (-1, -1)
    locality_strength: float = 1.0
    use_pos_embedding: bool = True


@dataclass
class ImageDomainVisionTransformerConfig(ModelConfig):
    use_mask: bool = True
    average_img_size: int = 320
    patch_size: int = 10
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = 9
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    gpsa_interval: Tuple[int, int] = (-1, -1)
    locality_strength: float = 1.0
    use_pos_embedding: bool = True
