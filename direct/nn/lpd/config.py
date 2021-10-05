# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class LPDNet(ModelConfig):
    num_iters: int = 25
    num_primal: int = 5
    num_dual: int = 5
    n_hidden: int = 32
