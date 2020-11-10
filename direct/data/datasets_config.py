# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from direct.config.defaults import BaseConfig
from direct.common.subsample_config import MaskingConfig

from omegaconf import MISSING


@dataclass
class TransformsConfig(BaseConfig):
    crop: Optional[Tuple[int, int]] = field(default_factory=lambda: (320, 320))
    crop_type: str = "uniform"
    estimate_sensitivity_maps: bool = False
    estimate_body_coil_image: bool = False
    sensitivity_maps_gaussian: Optional[float] = 0.7
    image_center_crop: bool = True
    pad_coils: Optional[int] = None
    scaling_key: Optional[str] = None
    masking: MaskingConfig = MaskingConfig()


@dataclass
class DatasetConfig(BaseConfig):
    name: str = MISSING
    lists: List[str] = field(default_factory=lambda: [])
    regex_filter: Optional[str] = None
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None
    input_kspace_key: Optional[str] = None
    input_image_key: Optional[str] = None
    kspace_context: int = 0


@dataclass
class FastMRIConfig(DatasetConfig):
    pass_mask: bool = False
    pass_attrs: bool = True


@dataclass
class CalgaryCampinasConfig(DatasetConfig):
    pass_mask: bool = False
    crop_outer_slices: bool = False
