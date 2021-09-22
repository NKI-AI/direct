# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class XPDNetConfig(ModelConfig):
    num_primal: int = 5
    num_dual: int = 1
    num_iter: int = 10
    mwcnn_hidden_channels: int = 16
    mwcnn_num_scales: int = 4
    mwcnn_bias: bool = True
    mwcnn_batchnorm: bool = False
