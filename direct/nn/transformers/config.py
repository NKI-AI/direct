# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from direct.config.defaults import ModelConfig
from direct.constants import COMPLEX_SIZE
from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType


@dataclass
class UFormerModelConfig(ModelConfig):
    in_channels: int = COMPLEX_SIZE
    out_channels: Optional[int] = None
    patch_size: int = 256
    embedding_dim: int = 32
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
    token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.LINEAR
    token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.LEFF
    shift_flag: bool = True
    modulator: bool = False
    cross_modulator: bool = False
    normalized: bool = True


@dataclass
class ImageDomainMRIUFormerConfig(ModelConfig):
    patch_size: int = 256
    embedding_dim: int = 32
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
    token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.LINEAR
    token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.LEFF
    shift_flag: bool = True
    modulator: bool = False
    cross_modulator: bool = False
    normalized: bool = True


@dataclass
class MRIViTConfig(ModelConfig):
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = 9
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
class VisionTransformer2DConfig(MRIViTConfig):
    in_channels: int = COMPLEX_SIZE
    out_channels: Optional[int] = None
    average_img_size: tuple[int, int] = MISSING
    patch_size: tuple[int, int] = (16, 16)


@dataclass
class VisionTransformer3DConfig(MRIViTConfig):
    in_channels: int = COMPLEX_SIZE
    out_channels: Optional[int] = None
    average_img_size: tuple[int, int, int] = MISSING
    patch_size: tuple[int, int, int] = (16, 16, 16)


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
    compute_per_coil: bool = True


@dataclass
class KSpaceDomainMRIViT3DConfig(MRIViTConfig):
    average_size: tuple[int, int] = (320, 320, 320)
    patch_size: tuple[int, int] = (16, 16, 16)
    compute_per_coil: bool = True
