# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Classes holding the typed configurations for the datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from direct.common.subsample_config import MaskingConfig
from direct.config.defaults import BaseConfig
from direct.data.transforms import RescaleMode


@dataclass
class CropTransformConfig(BaseConfig):
    crop: Optional[str] = None
    crop_type: Optional[str] = "uniform"
    image_center_crop: bool = False


@dataclass
class SensitivityMapEstimationTransformConfig(BaseConfig):
    estimate_sensitivity_maps: bool = True
    sensitivity_maps_type: str = "rss_estimate"
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6
    sensitivity_maps_espirit_crop: Optional[float] = 0.95
    sensitivity_maps_espirit_max_iters: Optional[int] = 30
    sensitivity_maps_gaussian: Optional[float] = 0.7


@dataclass
class AugmentationTransformsConfig(BaseConfig):
    rescale: Optional[tuple[int, int]] = None
    rescale_mode: RescaleMode = RescaleMode.BILINEAR
    pad: Optional[tuple[int, int]] = None
    random_rotation: bool = False
    random_rotation_degrees: tuple[int, ...] = (-90, 90)
    random_rotation_probability: Optional[float] = 0.5
    random_flip: bool = False
    random_flip_type: Optional[str] = "random"
    random_flip_probability: Optional[float] = 0.5
    random_reverse: bool = False
    random_reverse_probability: Optional[float] = 0.5
    compress_coils: Optional[int] = None


@dataclass
class NormalizationTransformConfig(BaseConfig):
    scaling_key: Optional[str] = "masked_kspace"
    scale_percentile: Optional[float] = 0.99


@dataclass
class TransformsConfig(BaseConfig):
    masking: Optional[MaskingConfig] = MaskingConfig()
    cropping: CropTransformConfig = CropTransformConfig()
    random_augmentations: Optional[AugmentationTransformsConfig] = AugmentationTransformsConfig()
    compute_and_apply_padding: bool = True
    padding_eps: float = 0.001
    estimate_body_coil_image: bool = False
    sensitivity_map_estimation: SensitivityMapEstimationTransformConfig = SensitivityMapEstimationTransformConfig()
    normalization: NormalizationTransformConfig = NormalizationTransformConfig()
    use_acs_as_mask: bool = False
    delete_acs_mask: bool = True
    delete_kspace: bool = True
    image_recon_type: str = "rss"
    pad_coils: Optional[int] = None
    use_seed: bool = True
    target_accelerations: Optional[tuple[float, ...]] = None


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
    filenames_filter: Optional[list[str]] = None
    filenames_lists: Optional[list[str]] = None
    filenames_lists_root: Optional[str] = None


@dataclass
class CMRxReconConfig(DatasetConfig):
    regex_filter: Optional[str] = None
    data_root: Optional[str] = None
    filenames_filter: Optional[list[str]] = None
    filenames_lists: Optional[list[str]] = None
    filenames_lists_root: Optional[str] = None
    kspace_key: str = "kspace_full"
    compute_mask: bool = False
    extra_keys: Optional[list[str]] = None
    kspace_context: Optional[str] = None


@dataclass
class FastMRIConfig(H5SliceConfig):
    pass_attrs: bool = True


@dataclass
class FastMRI3dConfig(DatasetConfig):
    data_root: Optional[str] = None
    filenames_filter: Optional[list[str]] = None
    filenames_lists: Optional[list[str]] = None
    filenames_lists_root: Optional[str] = None
    extra_keys: Optional[list[str]] = None
    pass_attrs: bool = False
    kspace_context: Optional[int] = None


@dataclass
class CalgaryCampinasConfig(H5SliceConfig):
    crop_outer_slices: bool = False


@dataclass
class FakeMRIBlobsConfig(DatasetConfig):
    pass_attrs: bool = True


@dataclass
class SheppLoganDatasetConfig(DatasetConfig):
    shape: tuple[int, int, int] = (100, 100, 30)
    num_coils: int = 12
    seed: Optional[int] = None
    B0: float = 3.0
    zlimits: tuple[float, float] = (-0.929, 0.929)


@dataclass
class SheppLoganProtonConfig(SheppLoganDatasetConfig):
    pass


@dataclass
class SheppLoganT1Config(SheppLoganDatasetConfig):
    pass


@dataclass
class SheppLoganT2Config(SheppLoganDatasetConfig):
    T2_star: bool = False
