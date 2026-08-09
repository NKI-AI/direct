# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes holding the typed configurations for the datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

from direct.common.subsample_config import MaskingConfig
from direct.config import BaseConfig
from direct.data.mri_transforms import (
    DemonsFilterType,
    HalfSplitType,
    MaskSplitterType,
    RandomFlipType,
    ReconstructionType,
    RegistrationSimulateReferenceType,
    RescaleMode,
    SensitivityMapType,
    TransformKey,
    TransformsType,
)


@dataclass
class CropTransformConfig(BaseConfig):
    crop: str | None = None
    crop_type: str | None = "uniform"
    image_center_crop: bool = False


@dataclass
class SensitivityMapEstimationTransformConfig(BaseConfig):
    estimate_sensitivity_maps: bool = True
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.RSS_ESTIMATE
    sensitivity_maps_espirit_threshold: float | None = 0.05
    sensitivity_maps_espirit_kernel_size: int | None = 6
    sensitivity_maps_espirit_crop: float | None = 0.95
    sensitivity_maps_espirit_max_iters: int | None = 30
    sensitivity_maps_gaussian: float | None = 0.7


@dataclass
class AugmentationTransformConfig(BaseConfig):
    rescale: tuple[int, ...] | None = None
    rescale_mode: RescaleMode | None = RescaleMode.NEAREST
    rescale_2d_if_3d: bool | None = False
    pad: tuple[int, ...] | None = None


@dataclass
class RandomAugmentationTransformsConfig(BaseConfig):
    random_rotation_degrees: tuple[int, ...] = (-90, 90)
    random_rotation_probability: float = 0.0
    random_flip_type: RandomFlipType | None = RandomFlipType.RANDOM
    random_flip_probability: float = 0.0
    random_reverse_probability: float = 0.0


@dataclass
class NormalizationTransformConfig(BaseConfig):
    scaling_key: str | None = "masked_kspace"
    scale_percentile: float | None = 0.99


@dataclass
class RegistrationTransformConfig(BaseConfig):
    registration: bool = False
    registration_simulate_reference: RegistrationSimulateReferenceType | None = None
    registration_simulate_elastic_sigma: float = 3.0
    registration_simulate_elastic_points: int = 3
    registration_simulate_elastic_rotate: float = 0.0
    registration_simulate_elastic_zoom: float = 0.0
    registration_estimate_displacement: bool = True
    registration_simulate_reference_from_key_index: int = 0
    registration_moving_key: TransformKey = TransformKey.TARGET
    demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES
    demons_num_iterations: int = 100
    demons_smooth_displacement_field: bool = True
    demons_standard_deviations: float = 1.5
    demons_intensity_difference_threshold: float | None = None
    demons_maximum_rms_error: float | None = None


@dataclass
class TransformsConfig(BaseConfig):
    """Configuration for the transforms.

    Attributes
    ----------
    masking : MaskingConfig
        Configuration for the masking.
    target_acceleration : float, optional
        Target acceleration to override the sampled acceleration with. Default is None.
    cropping : CropTransformConfig
        Configuration for the cropping.
    augmentation : AugmentationTransformConfig
        Configuration for the augmentation. Currently only rescale and pad are supported.
    random_augmentations : RandomAugmentationTransformsConfig
        Configuration for the random augmentations. Currently only random rotation, flip and reverse are supported.
    padding_eps : float
        Padding epsilon. Default is 0.001.
    estimate_body_coil_image : bool
        Estimate body coil image. Default is False.
    sensitivity_map_estimation : SensitivityMapEstimationTransformConfig
        Configuration for the sensitivity map estimation.
    normalization : NormalizationTransformConfig
        Configuration for the normalization.
    use_acs_as_mask : bool
        Use the ACS mask as the sampling mask. Default is False.
    delete_acs_mask : bool
        Delete ACS mask after its use. Default is True.
    delete_kspace : bool
        Delete k-space after its use. This should be set to False if the k-space is needed for the loss computation.
        Default is True.
    image_recon_type : ReconstructionType
        Image reconstruction type. Default is ReconstructionType.RSS.
    compress_coils : int, optional
        Number of coils to compress input k-space. It is not recommended to be used in combination with `pad_coils`.
        Default is None.
    pad_coils : int, optional
        Pad coils. Default is None.
    registration : RegistrationTransformConfig
        Configuration for the registration transforms.
    use_seed : bool
        Use seed for the transforms. Typically this should be set to True for reproducibility (e.g. inference),
        and False for training. Default is True.
    transforms_type : TransformsType
        Type of transforms.  By default the transforms are set for supervised learning (`TransformsType.SUPERVISED`).
        To use SSL transforms, set transforms_type to `SSL_SSDU`. This will require additional parameters to be set:
        mask_split_ratio, mask_split_acs_region, mask_split_keep_acs, mask_split_type, mask_split_gaussian_std.
        Default is `TransformsType.SUPERVISED`.
    mask_split_ratio : tuple[float, ...]
        Ratio of the mask to split into input and target mask. Ignored if transforms_type is not `SSL_SSDU`.
        Default is (0.4,).
    mask_split_acs_region : tuple[int, int]
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

    masking: MaskingConfig | None = field(default_factory=MaskingConfig)
    target_acceleration: float | None = None
    # Paper adaptive DYNAMIC sampling: independent init/ACS mask per time/slice frame.
    dynamic_mask: bool = False
    cropping: CropTransformConfig = field(default_factory=CropTransformConfig)
    augmentation: AugmentationTransformConfig = field(default_factory=AugmentationTransformConfig)
    random_augmentations: RandomAugmentationTransformsConfig = field(default_factory=RandomAugmentationTransformsConfig)
    padding_eps: float = 0.001
    estimate_body_coil_image: bool = False
    sensitivity_map_estimation: SensitivityMapEstimationTransformConfig = field(
        default_factory=SensitivityMapEstimationTransformConfig
    )
    normalization: NormalizationTransformConfig = field(default_factory=NormalizationTransformConfig)
    use_acs_as_mask: bool = False
    delete_acs_mask: bool = True
    delete_kspace: bool = True
    image_recon_type: ReconstructionType = ReconstructionType.RSS
    compress_coils: int | None = None
    pad_coils: int | None = None
    registration: RegistrationTransformConfig = field(default_factory=RegistrationTransformConfig)
    use_seed: bool = True
    transforms_type: TransformsType = TransformsType.SUPERVISED
    # Next attributes are for the mask splitter in case of transforms_type is set to SSL_SSDU
    mask_split_ratio: tuple[float, ...] = (0.4,)
    mask_split_acs_region: tuple[int, int] = (0, 0)
    mask_split_keep_acs: bool | None = False
    mask_split_type: MaskSplitterType = MaskSplitterType.GAUSSIAN
    mask_split_gaussian_std: float = 3.0
    mask_split_half_direction: HalfSplitType = HalfSplitType.VERTICAL


@dataclass
class DatasetConfig(BaseConfig):
    name: str = MISSING
    transforms: BaseConfig = field(default_factory=TransformsConfig)
    text_description: str | None = None


@dataclass
class H5SliceConfig(DatasetConfig):
    regex_filter: str | None = None
    input_kspace_key: str | None = None
    input_image_key: str | None = None
    kspace_context: int = 0
    pass_mask: bool = False
    data_root: str | None = None
    filenames_filter: list[str] | None = None
    filenames_lists: list[str] | None = None
    filenames_lists_root: str | None = None


@dataclass
class CMRxReconConfig(DatasetConfig):
    data_root: str | None = None
    filenames_filter: list[str] | None = None
    filenames_lists: list[str] | None = None
    filenames_lists_root: str | None = None
    kspace_key: str = "kspace_full"
    compute_mask: bool = False
    extra_keys: list[str] | None = None
    kspace_context: str | None = None


@dataclass
class FastMRIConfig(H5SliceConfig):
    pass_attrs: bool = True


@dataclass
class CalgaryCampinasConfig(H5SliceConfig):
    crop_outer_slices: bool = False


@dataclass
class FakeMRIBlobsConfig(DatasetConfig):
    pass_attrs: bool = True
    # If set (e.g. True / "time"), each sample is a full volume (T/S, coils, H, W).
    kspace_context: bool | str | int | None = None


@dataclass
class SheppLoganDatasetConfig(DatasetConfig):
    shape: tuple[int, int, int] = (100, 100, 30)
    num_coils: int = 12
    seed: int | None = None
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
