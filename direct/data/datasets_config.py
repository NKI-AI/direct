# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from typing import Tuple

from direct.config.defaults import BaseConfig


@dataclass
class TransformsConfig(BaseConfig):
    crop: Tuple[int, int] = (320, 320)


@dataclass
class DatasetConfig(BaseConfig):
    name: str = 'FastMRI'
    transforms: BaseConfig = TransformsConfig()
