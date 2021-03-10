# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from torch.nn import Module


@dataclass
class BaseConfig(Module):
    def __init__(self):
        super(BaseConfig, self).__init__()
        pass
