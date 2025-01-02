# Copyright (c) DIRECT Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from direct.config.defaults import ModelConfig
from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType


@dataclass
class UFormerModelConfig(ModelConfig):
    patch_size: int = 256
    embedding_dim: int = 32
    encoder_depths: Tuple[int, ...] = (2, 2, 2, 2)
    encoder_num_heads: Tuple[int, ...] = (1, 2, 4, 8)
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
    token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.LINEAR
    token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.LEFF
    shift_flag: bool = True
    modulator: bool = False
    cross_modulator: bool = False
    normalized: bool = True


@dataclass
class VisionTransformer2DConfig(ModelConfig):
    average_img_size: int | tuple[int, int] = MISSING
    patch_size: int | tuple[int, int] = 16
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = (9,)
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    use_gpsa: bool = True
    locality_strength: float = 1.0
    use_pos_embedding: bool = True
    normalized: bool = True


@dataclass
class VisionTransformer3DConfig(ModelConfig):
    average_img_size: int | tuple[int, int, int] = MISSING
    patch_size: int | tuple[int, int, int] = 16
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = (9,)
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    use_gpsa: bool = True
    locality_strength: float = 1.0
    use_pos_embedding: bool = True
    normalized: bool = True
