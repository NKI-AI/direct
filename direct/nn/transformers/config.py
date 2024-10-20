# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig


@dataclass
class MRIViTConfig(ModelConfig):
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
    normalized: bool = True


@dataclass
class ImageDomainMRIViT2DConfig(MRIViTConfig):
    average_size: tuple[int, int] = (320, 320)
    patch_size: tuple[int, int] = (16, 16)


@dataclass
class ImageDomainMRIViT3DConfig(MRIViTConfig):
    average_size: tuple[int, int] = (320, 320, 320)
    patch_size: tuple[int, int] = (16, 16, 16)


@dataclass
class KSpaceDomainMRIViT2DConfig(MRIViTConfig):
    average_size: tuple[int, int] = (320, 320)
    patch_size: tuple[int, int] = (16, 16)


@dataclass
class KSpaceDomainMRIViT3DConfig(MRIViTConfig):
    average_size: tuple[int, int] = (320, 320, 320)
    patch_size: tuple[int, int] = (16, 16, 16)
