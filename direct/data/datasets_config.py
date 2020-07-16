# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass, field
from typing import Tuple

from direct.config.defaults import BaseConfig
from direct.common.subsample_config import MaskingConfig


@dataclass
class TransformsConfig(BaseConfig):
    crop: Tuple[int] = field(default_factory=lambda: (320, 320))
    estimate_sensitivity_maps: bool = False
    masking: MaskingConfig = MaskingConfig()


@dataclass
class DatasetConfig(BaseConfig):
    name: str = 'FastMRI'
    transforms: BaseConfig = TransformsConfig()
