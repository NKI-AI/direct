# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Classes holding the typed configurations for the datasets."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from omegaconf import MISSING

from direct.common.subsample_config import MaskingConfig
from direct.config.defaults import BaseConfig


@dataclass
class TransformsConfig(BaseConfig):
    masking: MaskingConfig = MaskingConfig()
    crop: Optional[Tuple[int, int]] = None
    crop_type: Optional[str] = "uniform"
    image_center_crop: bool = False
    padding_eps: float = 0.001
    estimate_sensitivity_maps: bool = True
    estimate_body_coil_image: bool = False
    sensitivity_maps_gaussian: Optional[float] = 0.7
    delete_acs_mask: bool = True
    delete_kspace: bool = True
    image_recon_type: str = "rss"
    pad_coils: Optional[int] = None
    scaling_key: Optional[str] = "masked_kspace"
    scale_percentile: Optional[float] = 0.99
    use_seed: bool = True


@dataclass
class DatasetConfig(BaseConfig):
    name: str = MISSING
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None


@dataclass
class H5SliceConfig(DatasetConfig):
    regex_filter: Optional[str] = None
    input_kspace_key: Optional[str] = None
    input_image_key: Optional[str] = None
    kspace_context: int = 0
    pass_mask: bool = False
    data_root: Optional[str] = None
    filenames_filter: Optional[List[str]] = None
    filenames_lists: Optional[List[str]] = None
    filenames_lists_root: Optional[str] = None


@dataclass
class FastMRIConfig(H5SliceConfig):
    pass_attrs: bool = True


@dataclass
class AnnotatedFastMRIConfig(FastMRIConfig):
    multiple_annotation_policy: str = "all"


@dataclass
class AnnotatedFastMRIBrainConfig(AnnotatedFastMRIConfig):
    pass


@dataclass
class AnnotatedFastMRIKneeConfig(AnnotatedFastMRIConfig):
    pass


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
