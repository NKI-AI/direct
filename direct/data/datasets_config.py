# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Classes holding the typed configurations for the datasets."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from omegaconf import MISSING

from direct.common.subsample_config import MaskingConfig
from direct.config.defaults import BaseConfig


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
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None


@dataclass
class H5SliceConfig(DatasetConfig):
    lists: List[str] = field(default_factory=lambda: [])
    regex_filter: Optional[str] = None
    input_kspace_key: Optional[str] = None
    input_image_key: Optional[str] = None
    kspace_context: int = 0
    pass_mask: bool = False


@dataclass
class FastMRIConfig(H5SliceConfig):
    pass_attrs: bool = True


@dataclass
class CalgaryCampinasConfig(H5SliceConfig):
    crop_outer_slices: bool = False


@dataclass
class FakeMRIBlobsConfig(DatasetConfig):
    pass_attrs: bool = True


@dataclass
class SheppLoganDatasetConfig(DatasetConfig):
    shape: Tuple[int, int, int] = (100, 100, 30)
    num_coils: int = 12
    seed: Optional[int] = None
    B0: float = 3.0
    zlimits: Tuple[float, float] = (-0.929, 0.929)


@dataclass
class SheppLoganProtonConfig(SheppLoganDatasetConfig):
    pass


@dataclass
class SheppLoganT1Config(SheppLoganDatasetConfig):
    pass


@dataclass
class SheppLoganT2Config(SheppLoganDatasetConfig):
    T2_star: bool = False
