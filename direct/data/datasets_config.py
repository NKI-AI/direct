# Copyright (c) DIRECT Contributors

"""Classes holding the typed configurations for the datasets."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from omegaconf import MISSING

from direct.common.subsample_config import MaskingConfig
from direct.config.defaults import BaseConfig
from direct.data.mri_transforms import (
    HalfSplitType,
    MaskSplitterType,
    RandomFlipType,
    ReconstructionType,
    SensitivityMapType,
    TransformsType,
)


@dataclass
class CropTransformConfig(BaseConfig):
    crop: Optional[str] = None
    crop_type: Optional[str] = "uniform"
    image_center_crop: bool = False


@dataclass
class SensitivityMapEstimationTransformConfig(BaseConfig):
    estimate_sensitivity_maps: bool = True
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.RSS_ESTIMATE
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6
    sensitivity_maps_espirit_crop: Optional[float] = 0.95
    sensitivity_maps_espirit_max_iters: Optional[int] = 30
    sensitivity_maps_gaussian: Optional[float] = 0.7


@dataclass
class RandomAugmentationTransformsConfig(BaseConfig):
    random_rotation_degrees: Tuple[int, ...] = (-90, 90)
    random_rotation_probability: float = 0.0
    random_flip_type: Optional[RandomFlipType] = RandomFlipType.RANDOM
    random_flip_probability: float = 0.0
    random_reverse_probability: float = 0.0


@dataclass
class NormalizationTransformConfig(BaseConfig):
    scaling_key: Optional[str] = "masked_kspace"
    scale_percentile: Optional[float] = 0.99


@dataclass
class TransformsConfig(BaseConfig):
    """Configuration for the transforms.

    Attributes
    ----------
    masking : MaskingConfig
        Configuration for the masking.
    cropping : CropTransformConfig
        Configuration for the cropping.
    random_augmentations : RandomAugmentationTransformsConfig
        Configuration for the random augmentations.
    padding_eps : float
        Padding epsilon. Default is 0.001.
    estimate_body_coil_image : bool
        Estimate body coil image. Default is False.
    sensitivity_map_estimation : SensitivityMapEstimationTransformConfig
        Configuration for the sensitivity map estimation.
    normalization : NormalizationTransformConfig
        Configuration for the normalization.
    delete_acs_mask : bool
        Delete ACS mask after its use. Default is True.
    delete_kspace : bool
        Delete k-space after its use. This should be set to False if the k-space is needed for the loss computation.
        Default is True.
    image_recon_type : ReconstructionType
        Image reconstruction type. Default is ReconstructionType.RSS.
    pad_coils : int, optional
        Pad coils. Default is None.
    use_seed : bool
        Use seed for the transforms. Typically this should be set to True for reproducibility (e.g. inference),
        and False for training. Default is True.
    transforms_type : TransformsType
        Type of transforms.  By default the transforms are set for supervised learning (`TransformsType.SUPERVISED`).
        To use SSL transforms, set transforms_type to `SSL_SSDU`. This will require additional parameters to be set:
        mask_split_ratio, mask_split_acs_region, mask_split_keep_acs, mask_split_type, mask_split_gaussian_std.
        Default is `TransformsType.SUPERVISED`.
    mask_split_ratio : Tuple[float, ...]
        Ratio of the mask to split into input and target mask. Ignored if transforms_type is not `SSL_SSDU`.
        Default is (0.4,).
    mask_split_acs_region : Tuple[int, int]
        Region of the ACS k-space to keep in the input mask. Ignored if transforms_type is not `SSL_SSDU`.
        Default is (0, 0).
    mask_split_keep_acs : bool, optional
        Keep ACS in both masks, input and target. Ignored if transforms_type is not `SSL_SSDU`. Default is False.
    mask_split_type : MaskSplitterType
        Type of mask splitting if transforms_type is `SSL_SSDU`. Ignored if transforms_type is not SSL_SSDU.
        Default is `MaskSplitterType.GAUSSIAN`.
    mask_split_gaussian_std : float
        Standard deviation of the Gaussian mask splitter. Ignored if mask_split_type is not `MaskSplitterType.GAUSSIAN`.
        Ignored if transforms_type is not `SSL_SSDU`. Default is 3.0.
    mask_split_half_direction : HalfSplitType
        Direction to split the mask if mask_split_type is `MaskSplitterType.HALF`.
        Ignored if MaskSplitterType is not `HALF` or transforms_type is not `SSL_SSDU`.
        Default is `HalfSplitType.VERTICAL`.
    """

    masking: Optional[MaskingConfig] = MaskingConfig()
    cropping: CropTransformConfig = CropTransformConfig()
    random_augmentations: RandomAugmentationTransformsConfig = RandomAugmentationTransformsConfig()
    padding_eps: float = 0.001
    estimate_body_coil_image: bool = False
    sensitivity_map_estimation: SensitivityMapEstimationTransformConfig = SensitivityMapEstimationTransformConfig()
    normalization: NormalizationTransformConfig = NormalizationTransformConfig()
    delete_acs_mask: bool = True
    delete_kspace: bool = True
    image_recon_type: ReconstructionType = ReconstructionType.RSS
    pad_coils: Optional[int] = None
    use_seed: bool = True
    transforms_type: TransformsType = TransformsType.SUPERVISED
    # Next attributes are for the mask splitter in case of transforms_type is set to SSL_SSDU
    mask_split_ratio: Tuple[float, ...] = (0.4,)
    mask_split_acs_region: Tuple[int, int] = (0, 0)
    mask_split_keep_acs: Optional[bool] = False
    mask_split_type: MaskSplitterType = MaskSplitterType.GAUSSIAN
    mask_split_gaussian_std: float = 3.0
    mask_split_half_direction: HalfSplitType = HalfSplitType.VERTICAL


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
class CMRxReconConfig(DatasetConfig):
    regex_filter: Optional[str] = None
    data_root: Optional[str] = None
    filenames_filter: Optional[List[str]] = None
    filenames_lists: Optional[List[str]] = None
    filenames_lists_root: Optional[str] = None
    kspace_key: str = "kspace_full"
    compute_mask: bool = False
    extra_keys: Optional[List[str]] = None
    kspace_context: Optional[str] = None


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
