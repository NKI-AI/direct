# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from direct.config.defaults import BaseConfig
from direct.common.subsample_config import MaskingConfig


@dataclass
class TransformsConfig(BaseConfig):
    crop: Tuple[int, int] = field(default_factory=lambda: (320, 320))
    estimate_sensitivity_maps: bool = False
    pad_coils: Optional[int] = None
    masking: MaskingConfig = MaskingConfig()


@dataclass
class DatasetConfig(BaseConfig):
    name: str = "FastMRI"
    lists: List[str] = field(default_factory=lambda: [])
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None
