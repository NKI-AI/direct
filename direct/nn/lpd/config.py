# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class LPDNetConfig(ModelConfig):
    num_iter: int = 25
    num_primal: int = 5
    num_dual: int = 5
    primal_model_architecture: str = "MWCNN"
    dual_model_architecture: str = "DIDN"
    primal_mwcnn_hidden_channels: int = 16
    primal_mwcnn_num_scales: int = 4
    primal_mwcnn_bias: bool = True
    primal_mwcnn_batchnorm: bool = False
    primal_unet_num_filters: int = 8
    primal_unet_num_pool_layers: int = 4
    primal_unet_dropout_probability: float = 0.0
    primal_uformer_patch_size: int = 128
    primal_uformer_embedding_dim: int = 8
    primal_uformer_encoder_depths: list[int, ...] = (2, 2, 2)
    primal_uformer_encoder_num_heads: list[int, ...] = (2, 4, 8)
    primal_uformer_bottleneck_depth: int = 2
    primal_uformer_bottleneck_num_heads: int = 16
    primal_uformer_win_size: int = 8
    dual_conv_hidden_channels: int = 16
    dual_conv_n_convs: int = 4
    dual_conv_batchnorm: bool = False
    dual_didn_hidden_channels: int = 64
    dual_didn_num_dubs: int = 6
    dual_didn_num_convs_recon: int = 9
    dual_unet_num_filters: int = 8
    dual_unet_num_pool_layers: int = 4
    dual_unet_dropout_probability: float = 0.0
    dual_uformer_patch_size: int = 256
    dual_uformer_embedding_dim: int = 32
    dual_uformer_encoder_depths: list[int, ...] = (2, 2, 2)
    dual_uformer_encoder_num_heads: list[int, ...] = (2, 4, 8)
    dual_uformer_bottleneck_depth: int = 2
    dual_uformer_bottleneck_num_heads: int = 16
    dual_uformer_win_size: int = 8
