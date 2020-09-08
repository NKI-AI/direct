# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from direct.config.defaults import BaseConfig
from direct.common.subsample_config import MaskingConfig


@dataclass
class TransformsConfig(BaseConfig):
    crop: Optional[Tuple[int, int]] = field(default_factory=lambda: (320, 320))
    crop_type: str = "uniform"
    estimate_sensitivity_maps: bool = False
    pad_coils: Optional[int] = None
    masking: MaskingConfig = MaskingConfig()


@dataclass
class DatasetConfig(BaseConfig):
    name: str = "FastMRI"
    lists: List[str] = field(default_factory=lambda: [])
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None
    kspace_context: int = 0


@dataclass
class FastMRIConfig(DatasetConfig):
    pass_mask: bool = True


@dataclass
class CalgaryCampinasConfig(DatasetConfig):
    pass
