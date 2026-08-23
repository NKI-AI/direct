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
"""The `direct.data.mri_transforms` module contains mri transformations utilized to transform or augment k-space data,
used for DIRECT's training pipeline. They can be also used individually by importing them into python scripts.
"""

import contextlib
import copy
import functools
import logging
import random
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch

from direct.algorithms.mri_algorithms import EspiritCalibration
from direct.data import transforms as T
from direct.exceptions import ItemNotFoundException
from direct.registration.elastic_deformation import RandomElasticDeformationModule
from direct.registration.registration import (
    DemonsFilterType,
    DisplacementModule,
    DisplacementTransformType,
)
from direct.ssl.ssl import (
    GaussianMaskSplitterModule,
    HalfMaskSplitterModule,
    HalfSplitType,
    MaskSplitterType,
    SSLTransformMaskPrefixes,
    UniformMaskSplitterModule,
)
from direct.types import DirectEnum, IntegerListOrTupleString, KspaceKey, TransformKey
from direct.utils import DirectModule, DirectTransform
from direct.utils.asserts import assert_complex

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temp_seed(rng, seed):
    """Temp seed.

    Args:
        rng: Rng.
        seed: Seed.
    """
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class Compose(DirectTransform):
    """Compose several transformations together, for instance ClipAndScale and a flip.

    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
    additional dependencies.
    """

    def __init__(self, transforms: Iterable[Callable]) -> None:
        """Inits :class:`Compose`.

        Args:
            transforms: List of transforms.
        """
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`Compose`.

        Args:
            sample: Dict sample.

        Returns:
            Dict sample transformed by `transforms`.
        """
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def __repr__(self):
        """Representation of :class:`Compose`."""
        repr_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            repr_string += "\n"
            repr_string += f"    {transform},"
        repr_string = repr_string[:-1] + "\n)"
        return repr_string


class RandomRotation(DirectTransform):
    r"""Random :math:`k`-space rotation.

    Performs a random rotation with probability :math:`p`. Rotation degrees must be multiples of 90.
    """

    def __init__(
        self,
        degrees: Sequence[int] | None = (-90, 90),
        p: float = 0.5,
        keys_to_rotate: tuple[TransformKey, ...] = (TransformKey.KSPACE,),
    ) -> None:
        r"""Inits :class:`RandomRotation`.

        Args:
            degrees: Degrees of rotation. Must be a multiple of 90. If len(degrees) > 1, then a degree will be chosen at random.
                Default is ``(-90, 90)``.
            p: Probability of rotation. Default is ``0.5``.
            keys_to_rotate: Keys to rotate. Default is ``"kspace"``.
        """
        super().__init__()

        if degrees is None:
            degrees = (-90, 90)
        assert all(degree % 90 == 0 for degree in degrees)

        self.degrees = degrees
        self.p = p
        self.keys_to_rotate = keys_to_rotate

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`RandomRotation`.

        Args:
            sample: Dict sample.

        Returns:
            Sample with rotated values of `keys_to_rotate`.
        """
        if random.SystemRandom().random() <= self.p:
            degree = random.SystemRandom().choice(self.degrees)
            k = degree // 90
            for key in self.keys_to_rotate:
                if key in sample:
                    value = T.view_as_complex(sample[key].clone())
                    sample[key] = T.view_as_real(torch.rot90(value, k=k, dims=(-2, -1)))

            # If rotated by multiples of (n + 1) * 90 degrees, reconstruction size also needs to change
            reconstruction_size = sample.get("reconstruction_size", None)
            if reconstruction_size and (k % 2) == 1:
                sample["reconstruction_size"] = (
                    reconstruction_size[:-3] + reconstruction_size[-3:-1][::-1] + reconstruction_size[-1:]
                )

        return sample


class RandomFlipType(DirectEnum):
    """RandomFlipType."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    RANDOM = "random"
    BOTH = "both"


class RandomFlip(DirectTransform):
    r"""Random k-space flip transform.

    Performs a random flip with probability :math:`p`. Flip can be horizontal, vertical, or a random choice of the two.
    """

    def __init__(
        self,
        flip: RandomFlipType | None = RandomFlipType.RANDOM,
        p: float = 0.5,
        keys_to_flip: tuple[TransformKey, ...] = (TransformKey.KSPACE,),
    ) -> None:
        r"""Inits :class:`RandomFlip`.

        Args:
            flip: Horizontal, vertical, or random choice of the two. Default is ``RandomFlipType.RANDOM``.
            p: Probability of flip. Default is ``0.5``.
            keys_to_flip: Keys to flip. Default is ``"kspace"``.
        """
        super().__init__()

        self.flip = flip
        self.p = p
        self.keys_to_flip = keys_to_flip

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`RandomFlip`.

        Args:
            sample: Dict sample.

        Returns:
            Sample with flipped values of `keys_to_flip`.
        """
        if random.SystemRandom().random() <= self.p:
            dims = (
                (-2,)
                if self.flip == "horizontal"
                else (
                    (-1,)
                    if self.flip == "vertical"
                    else ((-2, -1) if self.flip == "both" else (random.SystemRandom().choice([-2, -1]),))
                )
            )

            for key in self.keys_to_flip:
                if key in sample:
                    value = T.view_as_complex(sample[key].clone())
                    value = torch.flip(value, dims=dims)
                    sample[key] = T.view_as_real(value)

        return sample


class RandomReverse(DirectTransform):
    r"""Random reverse of the order along a given dimension of a PyTorch tensor."""

    def __init__(
        self,
        dim: int = 1,
        p: float = 0.5,
        keys_to_reverse: tuple[TransformKey, ...] = (TransformKey.KSPACE,),
    ) -> None:
        r"""Inits :class:`RandomReverse`.

        Args:
            dim: Dimension along to perform reversion. Typically, this is for time or slice dimension. Default is ``2``.
            p: Probability of flip. Default is ``0.5``.
            keys_to_reverse: Keys to reverse. Default is ``"kspace"``.
        """
        super().__init__()

        self.dim = dim
        self.p = p
        self.keys_to_reverse = keys_to_reverse

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`RandomReverse`.

        Args:
            sample: Dict sample.

        Returns:
            Sample with flipped values of `keys_to_flip`.
        """
        if random.SystemRandom().random() <= self.p:
            dim = self.dim
            for key in self.keys_to_reverse:
                if key in sample:
                    tensor = sample[key].clone()

                    if dim < 0:
                        dim += tensor.dim()

                    tensor = T.view_as_complex(tensor)

                    index = [slice(None)] * tensor.dim()
                    index[dim] = torch.arange(tensor.size(dim) - 1, -1, -1, dtype=torch.long)

                    tensor = tensor[tuple(index)]

                    sample[key] = T.view_as_real(tensor)

        return sample


def _mask_func_supports_return_acceleration(mask_func: Callable) -> bool:
    """Check whether a mask callable can return sampled acceleration metadata."""
    from direct.common.subsample import BaseMaskFunc

    return isinstance(mask_func, BaseMaskFunc)


class CreateSamplingMask(DirectTransform):
    """Data Transformer for training MRI reconstruction models.

    Creates sampling mask.
    """

    def __init__(
        self,
        mask_func: Callable,
        shape: tuple[int, ...] | None = None,
        use_seed: bool = True,
        return_acs: bool = False,
        dynamic_mask: bool | None = None,
    ) -> None:
        """Inits :class:`CreateSamplingMask`.

        Args:
            mask_func: A function which creates a sampling mask of the appropriate shape.
            shape: Sampling mask shape. Default is ``None``.
            use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
                the same mask every time. Default is ``True``.
            return_acs: If True, it will generate an ACS mask. Default is ``False``.
            dynamic_mask: If True and k-space is 5D ``(coil, time/slice, height, width, complex)``, generate an independent
                init/ACS mask per temporal/slice frame (paper adaptive dynamic sampling). Default is ``None/False``.
        """
        super().__init__()
        self.mask_func = mask_func
        self.shape = shape
        self.use_seed = use_seed
        self.return_acs = return_acs
        self.dynamic_mask = bool(dynamic_mask)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`CreateSamplingMask`.

        Args:
            sample: Dict sample.

        Returns:
            Sample with `sampling_mask` key.
        """
        # Default: drop coil dim only. For 5D cine this yields (time, height, width, complex),
        # which MaskFuncMode.DYNAMIC / MULTISLICE require (>= 4 dims).
        # With dynamic_mask=True we instead build independent spatial masks per frame below.
        if not self.shape:
            if sample["kspace"].ndim == 5 and self.dynamic_mask:
                shape = sample["kspace"].shape[-3:]
            else:
                shape = sample["kspace"].shape[1:]
        elif any(_ is None for _ in self.shape):  # Allow None as values.
            if sample["kspace"].ndim == 5 and self.dynamic_mask:
                kspace_shape = list(sample["kspace"].shape[-3:-1])
            else:
                kspace_shape = list(sample["kspace"].shape[1:-1])
            shape = tuple(_ if _ else kspace_shape[idx] for idx, _ in enumerate(self.shape)) + (2,)
        else:
            shape = self.shape + (2,)

        supports_acc = _mask_func_supports_return_acceleration(self.mask_func)
        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))

        def _spatial_mask(mask: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
            """Normalize mask to (1, height, width, 1) before stacking time/slice frames."""
            mask = mask.reshape(-1, spatial_shape[0], spatial_shape[1], 1)
            if mask.shape[0] != 1:
                # Keep leading singleton; FastMRI mask funcs may insert an extra singleton.
                mask = mask[:1]
            return mask

        if sample["kspace"].ndim == 5 and self.dynamic_mask:
            # Paper init2-style: STATIC spatial mask func, independent seed per frame.
            spatial_shape = shape if len(shape) == 3 else shape[-3:]
            nz = sample["kspace"].shape[1]
            sampling_masks = []
            accelerations = []
            center_fractions = []
            acs_masks = [] if self.return_acs else None

            if seed is not None:
                np.random.seed(seed)
                dynamic_seeds = [int(_) for _ in np.random.randint(0, 10000, nz)]
            else:
                dynamic_seeds = [None for _ in range(nz)]

            for frame_seed in dynamic_seeds:
                mask_kwargs = {"shape": spatial_shape, "seed": frame_seed, "return_acs": False}
                if supports_acc:
                    mask_kwargs["return_acceleration"] = True
                    sampling_mask_z, acceleration_z, center_fraction_z = self.mask_func(**mask_kwargs)
                    accelerations.append(acceleration_z)
                    center_fractions.append(center_fraction_z)
                else:
                    sampling_mask_z = self.mask_func(**mask_kwargs)
                if "padding" in sample:
                    sampling_mask_z = T.apply_padding(sampling_mask_z, sample["padding"])
                sampling_mask_z = _spatial_mask(sampling_mask_z, spatial_shape)
                sampling_masks.append(sampling_mask_z.to(sample["kspace"].dtype))

                if self.return_acs:
                    acs_z = self.mask_func(shape=spatial_shape, seed=frame_seed, return_acs=True).to(
                        sample["kspace"].dtype
                    )
                    if "padding" in sample:
                        acs_z = T.apply_padding(acs_z, sample["padding"])
                    acs_z = _spatial_mask(acs_z, spatial_shape)
                    acs_masks.append(acs_z)  # ty: ignore[unresolved-attribute]

            sampling_mask = torch.stack(sampling_masks, dim=1)
            if self.return_acs:
                sample["acs_mask"] = torch.stack(acs_masks, dim=1)  # ty: ignore[invalid-argument-type]

            if supports_acc:
                sample["acceleration"] = torch.tensor(accelerations, dtype=sample["kspace"].dtype)
                sample["center_fraction"] = torch.tensor(center_fractions, dtype=sample["kspace"].dtype)
            else:
                acceleration = [
                    np.prod(sampling_mask[0, i].shape) / sampling_mask[0, i].sum()  # ty: ignore[unsupported-operator]
                    for i in range(sampling_mask.shape[1])
                ]
                sample["acceleration"] = torch.tensor(acceleration, dtype=torch.float32).unsqueeze(0)
                if self.return_acs:
                    center_fraction = [
                        sample["acs_mask"][0, i].sum() / np.prod(sample["acs_mask"][0, i].shape)
                        for i in range(sample["acs_mask"].shape[1])
                    ]
                    sample["center_fraction"] = torch.tensor(center_fraction, dtype=torch.float32).unsqueeze(0)
        else:
            mask_kwargs = {"shape": shape, "seed": seed, "return_acs": False}
            if supports_acc:
                mask_kwargs["return_acceleration"] = True
                sampling_mask, acceleration, center_fraction = self.mask_func(**mask_kwargs)
                sample["acceleration"] = torch.tensor([acceleration], dtype=sample["kspace"].dtype)
                sample["center_fraction"] = torch.tensor([center_fraction], dtype=sample["kspace"].dtype)
            else:
                sampling_mask = self.mask_func(**mask_kwargs)

            if "padding" in sample:
                sampling_mask = T.apply_padding(sampling_mask, sample["padding"])

            if sample["kspace"].ndim == 5 and sampling_mask.ndim == 4:
                sampling_mask = sampling_mask.unsqueeze(1)

            if not supports_acc:
                if sampling_mask.ndim == 5:
                    acceleration = [
                        np.prod(sampling_mask[0, i].shape) / sampling_mask[0, i].sum()
                        for i in range(sampling_mask.shape[1])
                    ]
                    sample["acceleration"] = torch.tensor(acceleration, dtype=torch.float32).unsqueeze(0)
                else:
                    sample["acceleration"] = (np.prod(sampling_mask.shape) / sampling_mask.sum()).unsqueeze(0)

            if self.return_acs:
                acs_mask = self.mask_func(shape=shape, seed=seed, return_acs=True)
                if "padding" in sample:
                    acs_mask = T.apply_padding(acs_mask, sample["padding"])
                if sample["kspace"].ndim == 5 and acs_mask.ndim == 4:
                    acs_mask = acs_mask.unsqueeze(1)
                sample["acs_mask"] = acs_mask
                if "center_fraction" not in sample:
                    if sample["acs_mask"].ndim == 5:
                        center_fraction = [
                            sample["acs_mask"][0, i].sum() / np.prod(sample["acs_mask"][0, i].shape)
                            for i in range(sample["acs_mask"].shape[1])
                        ]
                        sample["center_fraction"] = torch.tensor(center_fraction, dtype=torch.float32).unsqueeze(0)
                    else:
                        sample["center_fraction"] = (
                            sample["acs_mask"].sum() / np.prod(sample["acs_mask"].shape)
                        ).unsqueeze(0)

        sample["sampling_mask"] = sampling_mask
        return sample


class ApplyMaskModule(DirectModule):
    """Data Transformer for training MRI reconstruction models.

    Masks the input k-space (with key `input_kspace_key`) using a sampling mask with key `sampling_mask_key` onto
    a new masked k-space with key `target_kspace_key`.
    """

    def __init__(
        self,
        sampling_mask_key: str = "sampling_mask",
        input_kspace_key: KspaceKey = KspaceKey.KSPACE,
        target_kspace_key: KspaceKey = KspaceKey.MASKED_KSPACE,
    ) -> None:
        """Inits :class:`ApplyMaskModule`.

        Args:
            sampling_mask_key: Default is ``"sampling_mask"``.
            input_kspace_key: Default is ``KspaceKey.KSPACE``.
            target_kspace_key: Default KspaceKey.MASKED_KSPACE.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.sampling_mask_key = sampling_mask_key
        self.input_kspace_key = input_kspace_key
        self.target_kspace_key = target_kspace_key

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`ApplyMaskModule`.

        Applies mask with key `sampling_mask_key` onto kspace `input_kspace_key`. Result is stored as a tensor with
        key `target_kspace_key`.

        Args:
            sample: Dict sample containing keys `sampling_mask_key` and `input_kspace_key`.

        Returns:
            Sample with (new) key `target_kspace_key`.
        """
        if self.input_kspace_key not in sample:
            raise ValueError(f"Key {self.input_kspace_key} corresponding to `input_kspace_key` not found in sample.")
        input_kspace = sample[self.input_kspace_key]

        if self.sampling_mask_key not in sample:
            raise ValueError(f"Key {self.sampling_mask_key} corresponding to `sampling_mask_key` not found in sample.")
        sampling_mask = sample[self.sampling_mask_key]

        target_kspace, _ = T.apply_mask(input_kspace, sampling_mask)
        sample[self.target_kspace_key] = target_kspace
        return sample


class CropKspace(DirectTransform):
    """Data Transformer for training MRI reconstruction models.

    Crops the k-space by:
        * It first projects the k-space to the image-domain via the backward operator,
        * It crops the back-projected k-space to specified shape or key,
        * It transforms the cropped back-projected k-space to the k-space domain via the forward operator.
    """

    def __init__(
        self,
        crop: str | tuple[int, ...] | list[int],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        image_space_center_crop: bool = False,
        random_crop_sampler_type: str | None = "uniform",
        random_crop_sampler_use_seed: bool | None = True,
        random_crop_sampler_gaussian_sigma: list[float] | None = None,
    ) -> None:
        """Inits :class:`CropKspace`.

        Args:
            crop: Shape to crop the input to or a string pointing to a crop key (e.g. `reconstruction_size`).
            forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered). Default is
                :class:`direct.data.transforms.fft2`.
            backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered). Default is
                :class:`direct.data.transforms.ifft2`.
            image_space_center_crop: If set, the crop in the data will be taken in the center
            random_crop_sampler_type: If "uniform" the random cropping will be done by uniformly sampling `crop`, as opposed to
                `gaussian` which will sample from a gaussian distribution. If `image_space_center_crop` is True, then this is
                ignored. Default is ``"uniform"``.
            random_crop_sampler_use_seed: If true, a pseudo-random number based on the filename is computed so that every slice
                of the volume is cropped the same way. Default is ``True``.
            random_crop_sampler_gaussian_sigma: Standard variance of the gaussian when `random_crop_sampler_type` is `gaussian`.
                If `image_space_center_crop` is True, then this is ignored. Default is ``None``.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.image_space_center_crop = image_space_center_crop

        if not (isinstance(crop, (Iterable, str))):
            raise ValueError(
                f"Invalid input for `crop`. Received {crop}. Can be a list of tuple of integers or a string."
            )
        self.crop = crop

        if image_space_center_crop:
            self.crop_func = T.complex_center_crop
        else:
            self.crop_func = functools.partial(
                T.complex_random_crop,
                sampler=(random_crop_sampler_type if random_crop_sampler_type is not None else "uniform"),
                sigma=random_crop_sampler_gaussian_sigma,
            )
            self.random_crop_sampler_use_seed = random_crop_sampler_use_seed

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`CropKspace`.

        Args:
            sample: Dict sample containing key `kspace`.

        Returns:
            Cropped and masked sample.
        """

        kspace = sample["kspace"]  # shape (coil, [slice/time], height, width, complex=2)

        dim = self.spatial_dims.TWO_D if kspace.ndim == 4 else self.spatial_dims.THREE_D

        backprojected_kspace = self.backward_operator(kspace, dim=dim)  # shape (coil, height, width, complex=2)

        if isinstance(self.crop, IntegerListOrTupleString):
            crop_shape = IntegerListOrTupleString(self.crop)
        elif isinstance(self.crop, str):
            assert self.crop in sample, f"Not found {self.crop} key in sample."
            crop_shape = sample[self.crop][:-1]
        else:
            if kspace.ndim == 5 and len(self.crop) == 2:
                crop_shape = (kspace.shape[1],) + tuple(self.crop)
            else:
                crop_shape = tuple(self.crop)

        cropper_args: dict[str, Any] = {
            "data_list": [backprojected_kspace],
            "crop_shape": crop_shape,
            "contiguous": False,
        }
        if not self.image_space_center_crop:
            cropper_args["seed"] = (
                None if not self.random_crop_sampler_use_seed else tuple(map(ord, str(sample["filename"])))
            )
        cropped_backprojected_kspace = self.crop_func(**cropper_args)

        if "sampling_mask" in sample:
            crop_shape_tuple: tuple[int, ...] = tuple(crop_shape)  # ty: ignore[invalid-argument-type]
            mask_crop_shape: tuple[int, ...] = (1,) + crop_shape_tuple[1:] if kspace.ndim == 5 else crop_shape_tuple
            sample["sampling_mask"] = T.complex_center_crop(sample["sampling_mask"], mask_crop_shape)
            sample["acs_mask"] = T.complex_center_crop(sample["acs_mask"], mask_crop_shape)

        # Compute new k-space for the cropped_backprojected_kspace
        # shape (coil, [slice/time], new_height, new_width, complex=2)
        sample["kspace"] = self.forward_operator(cropped_backprojected_kspace, dim=dim)  # The cropped kspace

        return sample


class RescaleMode(DirectEnum):
    """RescaleMode."""

    AREA = "area"
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    TRILINEAR = "trilinear"


class RescaleKspace(DirectTransform):
    """Rescale k-space (downsample/upsample) module.

    Rescales the k-space:
    * It first projects the k-space to the image-domain via the backward operator,
    * It rescales the back-projected k-space to specified shape,
    * It transforms the rescaled back-projected k-space to the k-space domain via the forward operator.

    Args:
        shape: Shape to rescale the input. Must be correspond to (height, width).
        forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered). Default is
            :class:`direct.data.transforms.fft2`.
        backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered). Default is
            :class:`direct.data.transforms.ifft2`.
        rescale_mode: Mode to be used for rescaling. Can be RescaleMode.AREA, RescaleMode.BICUBIC, RescaleMode.BILINEAR,
            RescaleMode.NEAREST, RescaleMode.NEAREST_EXACT, or RescaleMode.TRILINEAR. Note that not all modes are supported for
            2D or 3D data. Default is ``RescaleMode.NEAREST``.
        kspace_key: K-space key. Default is ``KspaceKey.KSPACE``.
        rescale_2d_if_3d: If True and input k-space data is 3D, rescaling will be done only on the height and width
            dimensions. Default is ``False``.

    Notes:
        If the input k-space data is 3D, rescaling will be done only on the height and width dimensions if
        `rescale_2d_if_3d` is set to True.
    """

    def __init__(
        self,
        shape: tuple[int, int] | list[int],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        rescale_mode: RescaleMode | None = RescaleMode.NEAREST,
        kspace_key: KspaceKey = KspaceKey.KSPACE,
        rescale_2d_if_3d: bool | None = None,
    ) -> None:
        """Inits :class:`RescaleKspace`.

        Args:
            shape: Shape to rescale the input. Must be correspond to (height, width).
            forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered). Default is
                :class:`direct.data.transforms.fft2`.
            backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered). Default is
                :class:`direct.data.transforms.ifft2`.
            rescale_mode: Mode to be used for rescaling. Can be RescaleMode.AREA, RescaleMode.BICUBIC, RescaleMode.BILINEAR,
                RescaleMode.NEAREST, RescaleMode.NEAREST_EXACT, or RescaleMode.TRILINEAR. Note that not all modes are supported for
                2D or 3D data. Default is ``RescaleMode.NEAREST``.
            kspace_key: K-space key. Default is ``KspaceKey.KSPACE``.
            rescale_2d_if_3d: If True and input k-space data is 3D, rescaling will be done only on the height and width
                dimensions, by combining the slice/time dimension with the batch dimension. Default is ``False``.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        if len(shape) not in [2, 3]:
            raise ValueError(
                f"Shape should be a list or tuple of two integers if 2D or three integers if 3D. Received: {shape}."
            )
        self.shape = shape
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.rescale_mode = rescale_mode
        self.kspace_key = kspace_key

        self.rescale_2d_if_3d = rescale_2d_if_3d
        if rescale_2d_if_3d and len(shape) == 3:
            raise ValueError("Shape cannot have a length of 3 when rescale_2d_if_3d is set to True.")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`RescaleKspace`.

        Args:
            sample: Dict sample containing key `kspace`.

        Returns:
            Cropped and masked sample.
        """
        kspace = sample[self.kspace_key]  # shape (coil, [slice/time], height, width, complex=2)

        dim = self.spatial_dims.TWO_D if kspace.ndim == 4 else self.spatial_dims.THREE_D

        backprojected_kspace = self.backward_operator(kspace, dim=dim)

        if kspace.ndim == 5 and self.rescale_2d_if_3d:
            backprojected_kspace = backprojected_kspace.permute(1, 0, 2, 3, 4)

        if (kspace.ndim == 4) or (kspace.ndim == 5 and not self.rescale_2d_if_3d):
            backprojected_kspace = backprojected_kspace.unsqueeze(0)

        rescaled_backprojected_kspace = T.complex_image_resize(
            backprojected_kspace,
            self.shape,
            self.rescale_mode if self.rescale_mode is not None else "nearest",
        )

        if (kspace.ndim == 4) or (kspace.ndim == 5 and not self.rescale_2d_if_3d):
            rescaled_backprojected_kspace = rescaled_backprojected_kspace.squeeze(0)

        if kspace.ndim == 5 and self.rescale_2d_if_3d:
            rescaled_backprojected_kspace = rescaled_backprojected_kspace.permute(1, 0, 2, 3, 4)

        # Compute new k-space from rescaled_backprojected_kspace
        # shape (coil, [slice/time if rescale_2d_if_3d else new_slc_or_time], new_height, new_width, complex=2)
        sample[self.kspace_key] = self.forward_operator(rescaled_backprojected_kspace, dim=dim)  # The rescaled kspace

        return sample


class PadKspace(DirectTransform):
    """Pad k-space with zeros to desired shape module.

    Rescales the k-space by:
    * It first projects the k-space to the image-domain via the backward operator,
    * It pads the back-projected k-space to specified shape,
    * It transforms the rescaled back-projected k-space to the k-space domain via the forward operator.

    Args:
        pad_shape: Shape to zero-pad the input. Must be correspond to (height, width) or (slice/time, height, width).
        forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered). Default is
            :class:`direct.data.transforms.fft2`.
        backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered). Default is
            :class:`direct.data.transforms.ifft2`.
        kspace_key: K-space key. Default is ``KspaceKey.KSPACE``.
    """

    def __init__(
        self,
        pad_shape: tuple[int, ...] | list[int],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        kspace_key: KspaceKey = KspaceKey.KSPACE,
    ) -> None:
        """Inits :class:`RescaleKspace`.

        Args:
            pad_shape: Shape to zero-pad the input. Must be correspond to (height, width) or (slice/time, height, width).
            forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered). Default is
                :class:`direct.data.transforms.fft2`.
            backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered). Default is
                :class:`direct.data.transforms.ifft2`.
            kspace_key: K-space key. Default is ``KspaceKey.KSPACE``.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        if len(pad_shape) not in [2, 3]:
            raise ValueError(f"Shape should be a list or tuple of two or three integers. Received: {pad_shape}.")

        self.shape = pad_shape
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`PadKspace`.

        Args:
            sample: Dict sample containing key `kspace`.

        Returns:
            Cropped and masked sample.
        """
        kspace = sample[self.kspace_key]  # shape (coil, [slice or time], height, width, complex=2)
        shape = kspace.shape

        sample["original_size"] = shape[1:-1]

        dim = self.spatial_dims.TWO_D if kspace.ndim == 4 else self.spatial_dims.THREE_D

        backprojected_kspace = self.backward_operator(kspace, dim=dim)
        backprojected_kspace = T.view_as_complex(backprojected_kspace)

        padded_backprojected_kspace = T.pad_tensor(backprojected_kspace, self.shape)
        padded_backprojected_kspace = T.view_as_real(padded_backprojected_kspace)

        # shape (coil, [slice or time], height, width, complex=2)
        sample[self.kspace_key] = self.forward_operator(padded_backprojected_kspace, dim=dim)  # The padded kspace

        return sample


class ComputeZeroPadding(DirectTransform):
    r"""Computes zero padding present in multi-coil kspace input.

    Zero-padding is computed from multi-coil kspace with no signal contribution, i.e. its magnitude
    is really close to zero:

    .. math ::

        \text{padding} = \sum_{i=1}^{n_c} |y_i| < \frac{1}{n_x \cdot n_y}
        \sum_{j=1}^{n_x \cdot n_y} \big\{\sum_{i=1}^{n_c} |y_i|\big\} * \epsilon.
    """

    def __init__(
        self,
        kspace_key: KspaceKey = KspaceKey.KSPACE,
        padding_key: str = "padding",
        eps: float | None = 0.0001,
    ) -> None:
        """Inits :class:`ComputeZeroPadding`.

        Args:
            kspace_key: K-space key. Default is ``KspaceKey.KSPACE``.
            padding_key: Target key. Default is ``"padding"``.
            eps: Epsilon to multiply sum of signals. If really high, probably no padding will be produced. Default is
                ``0.0001``.
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.padding_key = padding_key
        self.eps = eps

    def __call__(self, sample: dict[str, Any], coil_dim: int = 0) -> dict[str, Any]:
        """Updates sample with a key `padding_key` with value a binary tensor.

        Non-zero entries indicate samples in kspace with key `kspace_key` which have minor contribution, i.e. padding.

        Args:
            sample: Dict sample containing key `kspace_key`.
            coil_dim: Coil dimension. Default is ``0``.

        Returns:
            Dict sample containing key `padding_key`.
        """
        if self.eps is None:
            return sample
        shape = sample[self.kspace_key].shape

        kspace = T.modulus(sample[self.kspace_key].clone()).sum(coil_dim)

        if len(shape) == 5:  # Check if 3D data
            # Assumes that slice dim is 0
            kspace = kspace.sum(0)

        padding = (kspace < (torch.mean(kspace) * self.eps)).to(kspace.device)

        if len(shape) == 5:
            padding = padding.unsqueeze(0)

        padding = padding.unsqueeze(coil_dim).unsqueeze(-1)
        sample[self.padding_key] = padding

        return sample


class ApplyZeroPadding(DirectTransform):
    """Applies zero padding present in multi-coil kspace input."""

    def __init__(self, kspace_key: KspaceKey = KspaceKey.KSPACE, padding_key: str = "padding") -> None:
        """Inits :class:`ApplyZeroPadding`.

        Args:
            kspace_key: K-space key. Default is ``KspaceKey.KSPACE``.
            padding_key: Target key. Default is ``"padding"``.
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.padding_key = padding_key

    def __call__(self, sample: dict[str, Any], coil_dim: int = 0) -> dict[str, Any]:
        """Applies zero padding on `kspace_key` with value a binary tensor.

        Args:
            sample: Dict sample containing key `kspace_key`.
            coil_dim: Coil dimension. Default is ``0``.

        Returns:
            Dict sample containing key `padding_key`.
        """

        sample[self.kspace_key] = T.apply_padding(sample[self.kspace_key], sample[self.padding_key])

        return sample


class ReconstructionType(DirectEnum):
    """Reconstruction method for :class:`ComputeImage` transform."""

    IFFT = "ifft"
    RSS = "rss"
    COMPLEX = "complex"
    COMPLEX_MOD = "complex_mod"
    SENSE = "sense"
    SENSE_MOD = "sense_mod"


class ComputeImageModule(DirectModule):
    """Compute Image transform."""

    def __init__(
        self,
        kspace_key: KspaceKey,
        target_key: str,
        backward_operator: Callable,
        type_reconstruction: ReconstructionType = ReconstructionType.RSS,
    ) -> None:
        """Inits :class:`ComputeImageModule`.

        Args:
            kspace_key: K-space key.
            target_key: Target key.
            backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            type_reconstruction: Type of reconstruction. Can be ReconstructionType.RSS, ReconstructionType.COMPLEX,
                ReconstructionType.COMPLEX_MOD, ReconstructionType.SENSE, ReconstructionType.SENSE_MOD or ReconstructionType.IFFT.
                Default is ``ReconstructionType.RSS``.
        """
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        self.target_key = target_key
        self.type_reconstruction = type_reconstruction

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        r"""Forward pass of :class:`ComputeImageModule`.

        Args:
            sample: Contains key kspace_key with value a torch.Tensor of shape (coil, \*spatial_dims, complex=2).

        Returns:
            Contains key target_key with value a torch.Tensor of shape (\*spatial_dims) if `type_reconstruction` is
                ReconstructionType.RSS, ReconstructionType.COMPLEX_MOD, ReconstructionType.SENSE_MOD, and of shape (\*spatial_dims,
                complex_dim=2) otherwise.
        """
        kspace_data = sample[self.kspace_key]
        dim = self.spatial_dims.TWO_D if kspace_data.ndim == 5 else self.spatial_dims.THREE_D
        # Get complex-valued data solution
        image = self.backward_operator(kspace_data, dim=dim)
        if self.type_reconstruction == ReconstructionType.IFFT:
            sample[self.target_key] = image
        elif self.type_reconstruction in [
            ReconstructionType.COMPLEX,
            ReconstructionType.COMPLEX_MOD,
        ]:
            sample[self.target_key] = image.sum(self.coil_dim)
        elif self.type_reconstruction == ReconstructionType.RSS:
            sample[self.target_key] = T.root_sum_of_squares(image, dim=self.coil_dim)
        else:
            if "sensitivity_map" not in sample:
                raise ItemNotFoundException(
                    "sensitivity map",
                    "Sensitivity map is required for SENSE reconstruction.",
                )
            sample[self.target_key] = T.complex_multiplication(T.conjugate(sample["sensitivity_map"]), image).sum(
                self.coil_dim
            )
        if self.type_reconstruction in [
            ReconstructionType.COMPLEX_MOD,
            ReconstructionType.SENSE_MOD,
        ]:
            sample[self.target_key] = T.modulus(sample[self.target_key], self.complex_dim)
        return sample


class EstimateBodyCoilImage(DirectTransform):
    """Estimates body coil image."""

    def __init__(self, mask_func: Callable, backward_operator: Callable, use_seed: bool = True) -> None:
        """Inits :class:`EstimateBodyCoilImage'.

        Args:
            mask_func: A function which creates a sampling mask of the appropriate shape.
            backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
                the same mask every time. Default is ``True``.
        """
        super().__init__()
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.backward_operator = backward_operator

    def __call__(self, sample: dict[str, Any], coil_dim: int = 0) -> dict[str, Any]:
        """Calls :class:`EstimateBodyCoilImage`.

        Args:
            sample: Contains key kspace_key with value a torch.Tensor of shape (coil, ..., complex=2).
            coil_dim: Coil dimension. Default is ``0``.

        Returns:
            Contains key `"body_coil_image`.
        """
        kspace = sample["kspace"]

        # We need to create an ACS mask based on the shape of this kspace, as it can be cropped.
        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        kspace_shape = tuple(sample["kspace"].shape[-3:])
        acs_mask = self.mask_func(shape=kspace_shape, seed=seed, return_acs=True)

        kspace = acs_mask * kspace + 0.0
        dim = self.spatial_dims.TWO_D if kspace.ndim == 4 else self.spatial_dims.THREE_D
        acs_image = self.backward_operator(kspace, dim=dim)

        sample["body_coil_image"] = T.root_sum_of_squares(acs_image, dim=coil_dim)
        return sample


class SensitivityMapType(DirectEnum):
    """SensitivityMapType."""

    ESPIRIT = "espirit"
    RSS_ESTIMATE = "rss_estimate"
    UNIT = "unit"


class EstimateSensitivityMapModule(DirectModule):
    """Data Transformer for training MRI reconstruction models.

    Estimates sensitivity maps given masked k-space data using one of three methods:

    *   Unit: unit sensitivity map in case of single coil acquisition.
    *   RSS-estimate: sensitivity maps estimated by using the root-sum-of-squares of the autocalibration-signal.
    *   ESPIRIT: sensitivity maps estimated with the ESPIRIT method [#]_. Note that this is currently not
        implemented for 3D data, and attempting to use it in such cases will result in a NotImplementedError.

    References:
        .. [#] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT--an eigenvalue
            approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
            doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.
    """

    def __init__(
        self,
        kspace_key: KspaceKey = KspaceKey.ACS_KSPACE,
        backward_operator: Callable = T.ifft2,
        type_of_map: SensitivityMapType | None = SensitivityMapType.RSS_ESTIMATE,
        gaussian_sigma: float | None = None,
        espirit_threshold: float | None = 0.05,
        espirit_kernel_size: int | None = 6,
        espirit_crop: float | None = 0.95,
        espirit_max_iters: int | None = 30,
    ) -> None:
        """Inits :class:`EstimateSensitivityMapModule`.

        Args:
            kspace_key: K-space key to compute the ACS image from. If `kspace_key` is not `KspaceKey.ACS_KSPACE`, the ACS mask
                should be provided in the sample. Default is ``KspaceKey.ACS_KSPACE``.
            backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            type_of_map: Type of map to estimate. Can be SensitivityMapType.RSS_ESTIMATE, SensitivityMapType.UNIT or
                SensitivityMapType.ESPIRIT. Default is ``SensitivityMapType.RSS_ESTIMATE``.
            gaussian_sigma: If non-zero, acs_image well be calculated
            espirit_threshold: Threshold for the calibration matrix when `type_of_map` is set to `SensitivityMapType.ESPIRIT`.
                Default is ``0.05``.
            espirit_kernel_size: Kernel size for the calibration matrix when `type_of_map` is set to
                `SensitivityMapType.ESPIRIT`. Default is ``6``.
            espirit_crop: Output eigenvalue cropping threshold when `type_of_map` is set to `SensitivityMapType.ESPIRIT`.
                Default is ``0.95``.
            espirit_max_iters: Power method iterations when `type_of_map` is set to `SensitivityMapType.ESPIRIT`. Default is
                ``30``.
        """
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        self.type_of_map = type_of_map

        # RSS estimate attributes
        self.gaussian_sigma = gaussian_sigma
        # Espirit attributes
        if type_of_map == SensitivityMapType.ESPIRIT:
            self.espirit_calibrator = EspiritCalibration(
                backward_operator,
                espirit_threshold if espirit_threshold is not None else 0.05,
                espirit_kernel_size if espirit_kernel_size is not None else 6,
                espirit_crop,
                espirit_max_iters if espirit_max_iters is not None else 100,
                kspace_key,
            )
        self.espirit_threshold = espirit_threshold
        self.espirit_kernel_size = espirit_kernel_size
        self.espirit_crop = espirit_crop
        self.espirit_max_iters = espirit_max_iters

    def estimate_acs_image(self, sample: dict[str, Any], width_dim: int = -2) -> torch.Tensor:
        """Estimates the autocalibration (ACS) image by sampling the k-space using the ACS mask.

        Args:
            sample: Sample dictionary,
            width_dim: Dimension corresponding to width. Default is ``-2``.

        Returns:
            Estimate of the ACS image.
        """
        kspace_data = sample[self.kspace_key]

        if self.kspace_key != KspaceKey.ACS_KSPACE:
            if TransformKey.ACS_MASK not in sample:
                raise ValueError("ACS mask is required for estimating ACS image from k-space but not found.")
            kspace_data = kspace_data * sample[TransformKey.ACS_MASK]

        if self.gaussian_sigma == 0 or not self.gaussian_sigma:
            kspace_acs = kspace_data + 0.0  # + 0.0 removes the sign of zeros.
        else:
            gaussian_mask = torch.linspace(-1, 1, kspace_data.size(width_dim), dtype=kspace_data.dtype)
            gaussian_mask = torch.exp(-((gaussian_mask / self.gaussian_sigma) ** 2))
            gaussian_mask_shape = torch.ones(len(kspace_data.shape)).int()
            gaussian_mask_shape[width_dim] = kspace_data.size(width_dim)
            gaussian_mask = gaussian_mask.reshape(tuple(gaussian_mask_shape))
            kspace_acs = kspace_data * gaussian_mask + 0.0

        # Get complex-valued data solution
        # Shape (batch, [slice/time], coil, height, width, complex=2)
        dim = self.spatial_dims.TWO_D if kspace_data.ndim == 5 else self.spatial_dims.THREE_D
        acs_image = self.backward_operator(kspace_acs, dim=dim)

        return acs_image

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calculates sensitivity maps for the input sample.

        Args:
            sample: Must contain key matching kspace_key with value a (complex) torch.Tensor of shape (coil, height, width,
                complex=2).

        Returns:
            Sample with key "sensitivity_map" with value the estimated sensitivity map.
        """
        kspace = sample[self.kspace_key]  # shape (batch, coil, [slice/time], height, width, complex=2)

        if kspace.shape[self.coil_dim] == 1:
            warnings.warn(
                "Estimation of sensitivity map of Single-coil data. This warning will be displayed only once."
            )
        if "sensitivity_map" in sample:
            warnings.warn(
                "`sensitivity_map` is given, but will be overwritten. This warning will be displayed only once."
            )

        if self.type_of_map == SensitivityMapType.UNIT:
            sensitivity_map = torch.zeros(kspace.shape).float()
            # Assumes complex channel is last
            assert_complex(kspace, complex_last=True)
            sensitivity_map[..., 0] = 1.0
            # Shape (batch, coil, [slice/time], height, width, complex=2)
            sensitivity_map = sensitivity_map.to(kspace.device)

        elif self.type_of_map == SensitivityMapType.RSS_ESTIMATE:
            # Shape (batch, coil, [slice/time], height, width, complex=2)
            acs_image = self.estimate_acs_image(sample)
            # Shape (batch, [slice/time], height, width)
            acs_image_rss = T.root_sum_of_squares(acs_image, dim=self.coil_dim)
            # Shape (batch, 1, [slice/time], height, width, 1)
            acs_image_rss = acs_image_rss.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)
            # Shape (batch, coil, [slice/time], height, width, complex=2)
            sensitivity_map = T.safe_divide(acs_image, acs_image_rss)
        else:
            if sample[self.kspace_key].ndim > 5:
                raise NotImplementedError(
                    "EstimateSensitivityMapModule is not yet implemented for "
                    "Espirit sensitivity map estimation for 3D data."
                )
            sensitivity_map = self.espirit_calibrator(sample)

        sensitivity_map_norm = torch.sqrt(
            (sensitivity_map**2).sum(self.complex_dim).sum(self.coil_dim)
        )  # shape (batch, [slice/time], height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)

        sample[TransformKey.SENSITIVITY_MAP] = T.safe_divide(sensitivity_map, sensitivity_map_norm)
        return sample


class AddBooleanKeysModule(DirectModule):
    """Adds keys with boolean values to sample."""

    def __init__(self, keys: list[str], values: list[bool]) -> None:
        """Inits :class:`AddBooleanKeysModule`.

        Args:
            keys: A list of keys to be added.
            values: A list of values corresponding to the keys.
        """
        super().__init__()
        self.keys = keys
        self.values = values

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Adds boolean keys to the input sample dictionary.

        Args:
            sample: The input sample dictionary.

        Returns:
            The modified sample with added boolean keys.
        """
        for key, value in zip(self.keys, self.values):
            sample[key] = value

        return sample


class CopyKeysModule(DirectModule):
    """Copy keys to a new name from the sample if present."""

    def __init__(self, keys: list[str], new_keys: list[str]) -> None:
        """Inits :class:`CopyKeysModule`.

        Args:
            keys: Key(s) to copy.
            new_keys: Key(s) to create.
        """
        super().__init__()
        self.keys = keys
        self.new_keys = new_keys

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`CopyKeysModule`.

        Args:
            sample: Dictionary to look for keys and copy them with a new name.

        Returns:
            Dictionary with copied specified keys.
        """
        for key, new_key in zip(self.keys, self.new_keys):
            if key in sample:
                if isinstance(sample[key], np.ndarray):
                    sample[new_key] = sample[key].copy()  # Copy NumPy array
                elif isinstance(sample[key], torch.Tensor):
                    sample[new_key] = sample[key].detach().clone()  # Copy Torch tensor
                else:
                    sample[new_key] = copy.deepcopy(sample[key])
        return sample


class CompressCoilModule(DirectModule):
    """Compresses k-space coils using SVD."""

    def __init__(self, kspace_key: KspaceKey, num_coils: int) -> None:
        """Inits :class:`CompressCoilModule`.

        Args:
            kspace_key: K-space key.
            num_coils: Number of coils to compress.
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.num_coils = num_coils

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Performs coil compression to input k-space.

        Args:
            sample: Dict sample containing key `kspace_key`. Assumes coil dimension is first axis.

        Returns:
            Dict sample with `kspace_key` compressed to num_coils.
        """
        k_space = sample[self.kspace_key].clone()  # shape (batch, coil, [slice/time], height, width, complex=2)

        if k_space.shape[1] <= self.num_coils:
            return sample

        ndim = k_space.ndim

        k_space = torch.view_as_complex(k_space)

        if ndim == 6:  # If 3D sample reshape slice into batch dimension as sensitivities are computed 2D
            num_slice_or_time = k_space.shape[2]
            k_space = k_space.permute(0, 2, 1, 3, 4)
            k_space = k_space.reshape(k_space.shape[0] * num_slice_or_time, *k_space.shape[2:])

        shape = k_space.shape

        # Reshape the k-space data to combine spatial dimensions
        k_space_reshaped = k_space.reshape(shape[0], shape[1], -1)

        # Compute the coil combination matrix using Singular Value Decomposition (SVD)
        U, _, _ = torch.linalg.svd(k_space_reshaped, full_matrices=False)

        # Select the top ncoils_new singular vectors from the decomposition
        U_new = U[:, :, : self.num_coils]

        # Perform coil compression
        compressed_k_space = torch.matmul(U_new.transpose(1, 2), k_space_reshaped)

        # Reshape the compressed k-space back to its original shape
        compressed_k_space = compressed_k_space.reshape(shape[0], self.num_coils, *shape[2:])

        if ndim == 6:
            compressed_k_space = compressed_k_space.reshape(
                shape[0] // num_slice_or_time,
                num_slice_or_time,
                self.num_coils,
                *shape[2:],
            ).permute(0, 2, 1, 3, 4)

        compressed_k_space = torch.view_as_real(compressed_k_space)
        sample[self.kspace_key] = compressed_k_space  # shape (batch, new coil, [slice/time], height, width, complex=2)

        return sample


class DeleteKeysModule(DirectModule):
    """Remove keys from the sample if present."""

    def __init__(self, keys: list[str]) -> None:
        """Inits :class:`DeleteKeys`.

        Args:
            keys: Key(s) to delete.
        """
        super().__init__()
        self.keys = keys

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`DeleteKeys`.

        Args:
            sample: Dictionary to look for keys and remove them.

        Returns:
            Dictionary with deleted specified keys.
        """
        for key in self.keys:
            sample.pop(key, None)

        return sample


class IndexSelectionMode(DirectEnum):
    """IndexSelectionMode."""

    RANDOM = "random"
    CUSTOM = "custom"
    RANGE = "range"


class IndexSelectionModule(DirectModule):
    """Randomly selects indices from the sample.

    Args:
        key: Key to select indices from.
        mode: Mode of index selection. Can be IndexSelectionMode.RANDOM, IndexSelectionMode.CUSTOM or
            IndexSelectionMode.RANGE. Default is ``IndexSelectionMode.CUSTOM``.
        num_indices: Number of indices to select.
        out_key: Key to store the selected indices. If None, the indices are stored in the same key. Default is ``None``.
        index_dim: Dimension along which to select indices. Default is ``1``.
        use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default is ``True``.
    """

    def __init__(
        self,
        key: TransformKey,
        mode: IndexSelectionMode = IndexSelectionMode.CUSTOM,
        indices: list[int] | None = None,
        num_indices: int | None = None,
        out_key: TransformKey | None = None,
        index_dim: int = 0,
        use_seed: bool = True,
    ) -> None:
        """Inits :class:`IndexSelection`.

        Args:
            key: Key to select indices from.
            mode: Mode of index selection. Can be IndexSelectionMode.RANDOM, IndexSelectionMode.CUSTOM or
                IndexSelectionMode.RANGE. Default is ``IndexSelectionMode.CUSTOM``.
            indices: List of indices to select if mode is IndexSelectionMode.CUSTOM or range if mode is
                IndexSelectionMode.RANGE. Default is ``None``.
            num_indices: Number of indices to select if mode is IndexSelectionMode.RANDOM. Default is ``None``.
            out_key: Key to store the selected indices. If None, the indices are stored in the same key. Default is ``None``.
            index_dim: Dimension along which to select indices. Default is ``1``.
            use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
                the same mask every time. Default is ``True``.
        """
        super().__init__()
        self.key = key
        self.out_key = out_key if out_key is not None else key
        self.mode = mode
        self.indices = indices
        self.num_indices = num_indices
        self.index_dim = index_dim
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`IndexSelection`.

        Args:
            sample: Dictionary to look for key and select indices from.

        Returns:
            Dictionary with randomly selected indices.
        """
        if self.key not in sample:
            return sample

        if self.mode == IndexSelectionMode.RANDOM:
            seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
            with temp_seed(self.rng, seed):
                num_to_keep = max(min(self.num_indices, sample[self.key].shape[self.index_dim]), 1)  # ty: ignore[invalid-argument-type]
                start = self.rng.randint(0, sample[self.key].shape[self.index_dim] - num_to_keep)
                keep_indices = torch.arange(start, start + num_to_keep, device=sample[self.key].device)
        else:
            if self.mode == IndexSelectionMode.CUSTOM:
                keep_indices = torch.tensor(
                    [idx for idx in self.indices if np.abs(idx) < sample[self.key].shape[self.index_dim]],  # ty: ignore[not-iterable]
                    device=sample[self.key].device,
                )
            else:
                keep_indices = torch.arange(self.indices[0], self.indices[1], device=sample[self.key].device)  # ty: ignore[not-subscriptable]
            num_to_keep = len(keep_indices)

        sample[self.out_key] = sample[self.key].index_select(self.index_dim, keep_indices)

        if num_to_keep == 1:
            sample[self.out_key] = sample[self.out_key].squeeze(self.index_dim)

        return sample


class DropIndexModule(DirectModule):
    """Drop indices from the sample.

    Args:
        keys: Key(s) to drop indices from.
        index: Index to drop.
        index_dim: Dimension(s) along which to drop indices. If a list, must have the same length as `keys`. Default is
            ``1``.
        store_deleted_keys: Key(s) to store the deleted indices. If None, the deleted indices are not stored. If the length
            does not match `keys`, the remaining keys are set to None. Default is ``None``.
    """

    def __init__(
        self,
        keys: list[TransformKey],
        index: int,
        index_dim: int | list[int] = 1,
        store_deleted_keys: list[TransformKey] | None = None,
    ) -> None:
        """Inits :class:`DropIndexModule`.

        Args:
            keys: Key(s) to drop indices from.
            index: Index to drop.
            index_dim: Dimension(s) along which to drop indices. If a list, must have the same length as `keys`. Default is
                ``1``.
            store_deleted_keys: Key(s) to store the deleted indices. If None, the deleted indices are not stored. If the length
                does not match `keys`, the remaining keys are set to None. Default is ``None``.
        """
        super().__init__()
        self.keys = keys
        self.index = index
        self.index_dim = [index_dim] * len(keys) if isinstance(index_dim, int) else index_dim
        self.store_deleted_keys = store_deleted_keys
        if self.store_deleted_keys is not None and len(keys) > len(self.store_deleted_keys):
            self.store_deleted_keys = store_deleted_keys + [None] * (len(keys) - len(self.store_deleted_keys))  # ty: ignore[unsupported-operator]

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`DropIndexModule`.

        Args:
            sample: Dictionary to look for key and drop indices from.

        Returns:
            Dictionary with dropped index.
        """

        for i, key in enumerate(self.keys):
            if key not in sample:
                continue
            # Skip keys that do not have the requested index dimension (e.g. scalar
            # acceleration / center_fraction tensors, or a shared single mask).
            if self.index_dim[i] >= sample[key].ndim or sample[key].shape[self.index_dim[i]] == 1:
                continue
            if self.store_deleted_keys is not None:
                deleted_key = self.store_deleted_keys[i]
                if deleted_key:
                    sample[deleted_key] = sample[key].index_select(
                        self.index_dim[i],
                        torch.tensor(
                            [idx for idx in range(sample[key].shape[self.index_dim[i]]) if idx == self.index],
                            device=sample[key].device,
                        ),
                    )
            sample[key] = sample[key].index_select(
                self.index_dim[i],
                torch.tensor(
                    [idx for idx in range(sample[key].shape[self.index_dim[i]]) if idx != self.index],
                    device=sample[key].device,
                ),
            )

        return sample


class SqueezeKeyModule(DirectModule):
    """Squeeze the specified key(s) in the sample.

    Args:
        keys: Key(s) to squeeze.
        dim: Dimension to squeeze.
    """

    def __init__(self, keys: TransformKey, dim: int) -> None:
        """Inits :class:`SqueezeKeyModule`.

        Args:
            keys: Key(s) to squeeze.
            dim: Dimension to squeeze.
        """
        super().__init__()
        self.keys = keys
        self.dim = dim

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`SqueezeKeyModule`.

        Args:
            sample: Dictionary to look for keys to squeeze.

        Returns:
            Dictionary with squeezed specified keys.
        """
        for key in self.keys:
            if key in sample:
                sample[key] = sample[key].squeeze(self.dim)
        return sample


class RenameKeysModule(DirectModule):
    """Rename keys from the sample if present."""

    def __init__(self, old_keys: list[str], new_keys: list[str]) -> None:
        """Inits :class:`RenameKeys`.

        Args:
            old_keys: Key(s) to rename.
            new_keys: Key(s) to replace old keys.
        """
        super().__init__()
        self.old_keys = old_keys
        self.new_keys = new_keys

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`RenameKeys`.

        Args:
            sample: Dictionary to look for keys and rename them.

        Returns:
            Dictionary with renamed specified keys.
        """
        for old_key, new_key in zip(self.old_keys, self.new_keys):
            if old_key in sample:
                sample[new_key] = sample.pop(old_key)

        return sample


class PadCoilDimensionModule(DirectModule):
    """Pad the coils by zeros to a given number of coils.

    Useful if you want to collate volumes with different coil dimension.
    """

    def __init__(
        self,
        pad_coils: int | None = None,
        key: str = "masked_kspace",
        coil_dim: int = 1,
    ) -> None:
        """Inits :class:`PadCoilDimensionModule`.

        Args:
            pad_coils: Number of coils to pad to. Default is ``None``.
            key: Key to pad in sample. Default is ``"masked_kspace"``.
            coil_dim: Coil dimension along which the pad will be done. Default is ``0``.
        """
        super().__init__()
        self.num_coils = pad_coils
        self.key = key
        self.coil_dim = coil_dim

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`PadCoilDimensionModule`.

        Args:
            sample: Dictionary with key `self.key`.

        Returns:
            Dictionary with padded coils of sample[self.key] if self.num_coils is not None.
        """
        if not self.num_coils:
            return sample

        if self.key not in sample:
            return sample

        data = sample[self.key]

        curr_num_coils = data.shape[self.coil_dim]
        if curr_num_coils > self.num_coils:
            raise ValueError(
                f"Tried to pad to {self.num_coils} coils, but already have {curr_num_coils} for {sample['filename']}."
            )
        if curr_num_coils == self.num_coils:
            return sample

        shape = data.shape
        num_coils = shape[self.coil_dim]
        padding_data_shape = list(shape).copy()
        padding_data_shape[self.coil_dim] = max(self.num_coils - num_coils, 0)
        zeros = torch.zeros(padding_data_shape, dtype=data.dtype, device=data.device)
        sample[self.key] = torch.cat([zeros, data], dim=self.coil_dim)

        return sample


class ComputeScalingFactorModule(DirectModule):
    """Calculates scaling factor.

    Scaling factor is for the input data based on either to the percentile or to the maximum of `normalize_key`.
    """

    def __init__(
        self,
        normalize_key: None | TransformKey = TransformKey.MASKED_KSPACE,
        percentile: None | float = 0.99,
        scaling_factor_key: TransformKey = TransformKey.SCALING_FACTOR,
    ) -> None:
        """Inits :class:`ComputeScalingFactorModule`.

        Args:
            normalize_key: Key name to compute the data for. If the maximum has to be computed on the ACS, ensure the
                reconstruction on the ACS is available (typically `body_coil_image`). Default is ``"masked_kspace"``.
            percentile: Rescale data with the given percentile. If None, the division is done by the maximum. Default is
                ``0.99``.
            scaling_factor_key: Name of how the scaling factor will be stored. Default is ``"scaling_factor"``.
        """
        super().__init__()
        self.normalize_key = normalize_key
        self.percentile = percentile
        self.scaling_factor_key = scaling_factor_key

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`ComputeScalingFactorModule`.

        Args:
            sample: Sample with key `normalize_key` to compute scaling_factor.

        Returns:
            Sample with key `scaling_factor_key`.
        """
        if self.normalize_key == "scaling_factor":  # This is a real-valued given number
            scaling_factor = sample["scaling_factor"]
        elif not self.normalize_key:
            kspace = sample["masked_kspace"]
            scaling_factor = torch.tensor([1.0] * kspace.size(0), device=kspace.device, dtype=kspace.dtype)
        else:
            data = sample[self.normalize_key]
            scaling_factor: list | torch.Tensor = []
            # Compute the maximum and scale the input
            if self.percentile:
                for _ in range(data.size(0)):
                    # Used in case the k-space is padded (e.g. for batches)
                    non_padded_coil_data = data[_][data[_].sum(dim=tuple(range(1, data[_].ndim))).bool()]
                    tview = -1.0 * T.modulus(non_padded_coil_data).view(-1)
                    s, _ = torch.kthvalue(tview, int((1 - self.percentile) * tview.size()[0]) + 1)
                    scaling_factor += [-1.0 * s]
                scaling_factor = torch.tensor(scaling_factor, dtype=data.dtype, device=data.device)
            else:
                scaling_factor = T.modulus(data).amax(dim=list(range(data.ndim))[1:-1])
        sample[self.scaling_factor_key] = scaling_factor
        return sample


class NormalizeModule(DirectModule):
    """Normalize the input data."""

    def __init__(
        self,
        scaling_factor_key: TransformKey = TransformKey.SCALING_FACTOR,
        keys_to_normalize: list[TransformKey] | None = None,
    ) -> None:
        """Inits :class:`NormalizeModule`.

        Args:
            scaling_factor_key: Name of scaling factor key expected in sample. Default is ``'scaling_factor'``.
        """
        super().__init__()
        self.scaling_factor_key = scaling_factor_key

        self.keys_to_normalize = (
            [
                "masked_kspace",
                "target",
                "kspace",
                "body_coil_image",  # sensitivity_map does not require normalization.
                "initial_image",
                "initial_kspace",
                # Dropped reference frame (FROM_KEY / ELASTIC) must match moving k-space scale
                # so registration losses and inference metrics compare like magnitudes.
                "reference_kspace",
            ]
            if keys_to_normalize is None
            else keys_to_normalize
        )

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`NormalizeModule`.

        Args:
            sample: Sample to normalize.

        Returns:
            Sample with normalized values if their respective key is in `keys_to_normalize` and key `scaling_factor_key` exists
                in sample.
        """
        scaling_factor = sample.get(self.scaling_factor_key, None)
        # Normalize data
        if scaling_factor is not None:
            for key, value in sample.items():
                if key not in self.keys_to_normalize:
                    continue
                sample[key] = T.safe_divide(
                    value,
                    scaling_factor.reshape(-1, *[1 for _ in range(value.ndim - 1)]),
                )

            sample["scaling_diff"] = 0.0
        return sample


class WhitenDataModule(DirectModule):
    """Whitens complex data Module."""

    def __init__(self, epsilon: float = 1e-10, key: str = "complex_image") -> None:
        """Inits :class:`WhitenDataModule`.

        Args:
            epsilon: Default is ``1e-10``.
            key: Key to whiten. Default is ``"complex_image"``.
        """
        super().__init__()
        self.epsilon = epsilon
        self.key = key

    def complex_whiten(self, complex_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whiten complex image.

        Args:
            complex_image: Complex image tensor to whiten.

        Returns:
            mean, std, whitened_image: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # From: https://github.com/facebookresearch/fastMRI
        #       blob/da1528585061dfbe2e91ebbe99a5d4841a5c3f43/banding_removal/fastmri/data/transforms.py#L464
        real = complex_image[..., 0]
        imag = complex_image[..., 1]

        # Center around mean.
        mean = complex_image.mean()
        centered_complex_image = complex_image - mean

        # Determine covariance between real and imaginary.
        n_elements = real.nelement()
        real_real = (real.mul(real).sum() - real.mean().mul(real.mean())) / n_elements
        real_imag = (real.mul(imag).sum() - real.mean().mul(imag.mean())) / n_elements
        imag_imag = (imag.mul(imag).sum() - imag.mean().mul(imag.mean())) / n_elements
        eig_input = torch.Tensor([[real_real, real_imag], [real_imag, imag_imag]])

        # Remove correlation by rotating around covariance eigenvectors.
        eig_values, eig_vecs = torch.linalg.eig(eig_input)

        # Scale by eigenvalues for unit variance.
        std = (eig_values.real + self.epsilon).sqrt()
        whitened_image = torch.matmul(centered_complex_image, eig_vecs.real) / std

        return mean, std, whitened_image

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Forward pass of :class:`WhitenDataModule`.

        Args:
            sample: Sample with key `key`.

        Returns:
            Sample with value of `key` whitened.
        """
        _, _, whitened_image = self.complex_whiten(sample[self.key])
        sample[self.key] = whitened_image
        return sample


class AddTargetAcceleration(DirectTransform):
    """This will replace the acceleration factor in the sample with the target acceleration factor."""

    def __init__(self, target_acceleration: float):
        """Initialize the instance.

        Args:
            target_acceleration: Target acceleration.
        """
        super().__init__()
        self.target_acceleration = target_acceleration

    def __call__(self, sample: dict[str, Any]):
        """Call the instance.

        Args:
            sample: Sample.
        """
        sample["acceleration"][:] = self.target_acceleration
        return sample


class ModuleWrapper:
    """ModuleWrapper."""

    class SubWrapper:
        """SubWrapper."""

        def __init__(self, transform: Any, toggle_dims: bool) -> None:
            """Initialize the instance.

            Args:
                transform: Transform.
                toggle_dims: Toggle dims.
            """
            self.toggle_dims = toggle_dims
            self._transform = transform

        def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
            """Call the instance.

            Args:
                sample: Sample.

            Returns:
                The result.
            """
            if self.toggle_dims:
                for k, v in sample.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        sample[k] = v[None]
                    else:
                        sample[k] = [v]

            sample = self._transform.forward(sample)

            if self.toggle_dims:
                for k, v in sample.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        sample[k] = v.squeeze(0)
                    else:
                        sample[k] = v[0]

            return sample

        def __repr__(self) -> str:
            """Return the official string representation.

            Returns:
                The result.
            """
            return self._transform.__repr__()

    def __init__(self, module: Callable, toggle_dims: bool) -> None:
        """Initialize the instance.

        Args:
            module: Module.
            toggle_dims: Toggle dims.
        """
        self._module = module
        self.toggle_dims = toggle_dims

    def __call__(self, *args, **kwargs) -> SubWrapper:
        """Call the instance.

        Args:
            *args: Args.
            **kwargs: Kwargs.

        Returns:
            The result.
        """
        return self.SubWrapper(self._module(*args, **kwargs), toggle_dims=self.toggle_dims)


ApplyMask = ModuleWrapper(ApplyMaskModule, toggle_dims=False)
ComputeImage = ModuleWrapper(ComputeImageModule, toggle_dims=True)
EstimateSensitivityMap = ModuleWrapper(EstimateSensitivityMapModule, toggle_dims=True)
CopyKeys = ModuleWrapper(CopyKeysModule, toggle_dims=False)
DeleteKeys = ModuleWrapper(DeleteKeysModule, toggle_dims=False)
RenameKeys = ModuleWrapper(RenameKeysModule, toggle_dims=False)
IndexSelection = ModuleWrapper(IndexSelectionModule, toggle_dims=False)
DropIndex = ModuleWrapper(DropIndexModule, toggle_dims=False)
SqueezeKey = ModuleWrapper(SqueezeKeyModule, toggle_dims=False)
CompressCoil = ModuleWrapper(CompressCoilModule, toggle_dims=True)
PadCoilDimension = ModuleWrapper(PadCoilDimensionModule, toggle_dims=True)
ComputeScalingFactor = ModuleWrapper(ComputeScalingFactorModule, toggle_dims=True)
Normalize = ModuleWrapper(NormalizeModule, toggle_dims=False)
WhitenData = ModuleWrapper(WhitenDataModule, toggle_dims=False)
GaussianMaskSplitter = ModuleWrapper(GaussianMaskSplitterModule, toggle_dims=True)
UniformMaskSplitter = ModuleWrapper(UniformMaskSplitterModule, toggle_dims=True)
Displacement = ModuleWrapper(DisplacementModule, toggle_dims=True)
RandomElasticDeformation = ModuleWrapper(RandomElasticDeformationModule, toggle_dims=True)


class ToTensor(DirectTransform):
    """Transforms all np.array-like values in sample to torch.tensors."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Calls :class:`ToTensor`.

        Args:
            sample: Contains key 'kspace' with value a np.array of shape (coil, height, width) (2D) or (coil, slice, height,
                width) (3D)

        Returns:
            Contains key 'kspace' with value a torch.Tensor of shape (coil, height, width) (2D) or (coil, slice, height, width)
                (3D)
        """

        ndim = sample["kspace"].ndim - 1

        if ndim not in [2, 3]:
            raise ValueError(f"Can only cast 2D and 3D data (+coil) to tensor. Got {ndim}.")

        # Shape:    2D: (coil, height, width, complex=2), 3D: (coil, slice, height, width, complex=2)
        sample["kspace"] = T.to_tensor(sample["kspace"]).float()
        # Sensitivity maps are not necessarily available in the dataset.
        if "initial_kspace" in sample:
            # Shape:    2D: (coil, height, width, complex=2), 3D: (coil, slice, height, width, complex=2)
            sample["initial_kspace"] = T.to_tensor(sample["initial_kspace"]).float()
        if "initial_image" in sample:
            # Shape:    2D: (height, width), 3D: (slice, height, width)
            sample["initial_image"] = T.to_tensor(sample["initial_image"]).float()

        if "sensitivity_map" in sample:
            # Shape:    2D: (coil, height, width, complex=2), 3D: (coil, slice, height, width, complex=2)
            sample["sensitivity_map"] = T.to_tensor(sample["sensitivity_map"]).float()
        if "target" in sample:
            # Shape:    2D: (coil, height, width), 3D: (coil, slice, height, width)
            sample["target"] = torch.from_numpy(sample["target"]).float()
        if "sampling_mask" in sample:
            sample["sampling_mask"] = torch.from_numpy(sample["sampling_mask"]).bool()
        if "acs_mask" in sample:
            sample["acs_mask"] = torch.from_numpy(sample["acs_mask"]).bool()
        if "scaling_factor" in sample:
            sample["scaling_factor"] = torch.tensor(sample["scaling_factor"]).float()
        if "loglikelihood_scaling" in sample:
            # Shape: (coil, )
            sample["loglikelihood_scaling"] = torch.from_numpy(np.asarray(sample["loglikelihood_scaling"])).float()

        return sample


def build_pre_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Callable | None,
    crop: tuple[int, int] | str | None = None,
    crop_type: str | None = "uniform",
    rescale: tuple[int, int] | list[int] | None = None,
    rescale_mode: RescaleMode | None = RescaleMode.NEAREST,
    rescale_2d_if_3d: bool | None = False,
    pad: tuple[int, int] | list[int] | None = None,
    image_center_crop: bool = True,
    random_rotation_degrees: Sequence[int] | None = (-90, 90),
    random_rotation_probability: float = 0.0,
    random_flip_type: RandomFlipType | None = RandomFlipType.RANDOM,
    random_flip_probability: float = 0.0,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    use_seed: bool = True,
    pad_coils: int | None = None,
) -> DirectTransform:
    """Builds pre (on cpu) supervised MRI transforms.

    More specifically, the following transformations are applied:

    *   Converts input to (complex-valued) tensor.
    *   Applies k-space (center) crop if requested.
    *   Applies random augmentations (rotation, flip, reverse) if requested.
    *   Adds a sampling mask if `mask_func` is defined.
    *   Pads the coil dimension if requested.

    Args:
        forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered).
        backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        mask_func: A function which creates a sampling mask of the appropriate shape.
        crop: If not None, this will transform the "kspace" to an image domain, crop it, and transform it back. If a tuple
            of integers is given then it will crop the backprojected kspace to that size. If "reconstruction_size" is given,
            then it will crop the backprojected kspace according to it, but a key "reconstruction_size" must be present in the
            sample. Default is ``None``.
        crop_type: Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default is
            ``"uniform"``.
        rescale: If not None, this will transform the "kspace" to the image domain, rescale it, and transform it back. Must
            correspond to (height, width). This is ignored if `rescale` is None. Default is None. It is not recommended to be
            used in combination with `crop`.
        rescale_mode: Mode to be used for rescaling. Can be RescaleMode.AREA, RescaleMode.BICUBIC, RescaleMode.BILINEAR,
            RescaleMode.NEAREST, RescaleMode.NEAREST_EXACT, or RescaleMode.TRILINEAR. Note that not all modes are supported for
            2D or 3D data. Default is ``RescaleMode.NEAREST``.
        rescale_2d_if_3d: If True and k-space data is 3D, rescaling will be done only on the height and width dimensions, by
            combining the slice/time dimension with the batch dimension. This is ignored if `rescale` is None. Default is
            ``False``.
        pad: If not None, this will zero-pad the "kspace" to the given size. Must correspond to (height, width) or
            (slice/time, height, width). Default is ``None``.
        image_center_crop: If True the backprojected kspace will be cropped around the center, otherwise randomly. This will
            be ignored if `crop` is None. Default is ``True``.
        random_rotation_degrees: Default is ``(-90, 90)``.
        random_rotation_probability: If greater than 0.0, random rotations will be applied of `random_rotation_degrees`
            degrees, with probability `random_rotation_probability`. Default is ``0.0``.
        random_flip_type: Default is ``RandomFlipType.RANDOM``.
        random_flip_probability: If greater than 0.0, random rotation of `random_flip_type` type, with probability
            `random_flip_probability`. Default is ``0.0``.
        padding_eps: Padding epsilon. Default is ``0.0001``.
        estimate_body_coil_image: Estimate body coil image. Default is ``False``.
        use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default is ``True``.

    Returns:
        An MRI transformation object.
    """
    # pylint: disable=too-many-locals
    logger = logging.getLogger(build_pre_mri_transforms.__name__)

    mri_transforms: list[Callable] = [ToTensor()]
    if crop:
        mri_transforms += [
            CropKspace(
                crop=crop,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                image_space_center_crop=image_center_crop,
                random_crop_sampler_type=crop_type,
                random_crop_sampler_use_seed=use_seed,
            )
        ]
    if rescale:
        if crop:
            logger.warning(
                "Rescale and crop are both given. Rescale will be applied after cropping. This is not recommended."
            )
        mri_transforms += [
            RescaleKspace(
                shape=rescale,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                rescale_mode=rescale_mode,
                rescale_2d_if_3d=rescale_2d_if_3d,
            )
        ]
    if pad:
        mri_transforms += [
            PadKspace(
                pad_shape=pad,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
            )
        ]
    if random_rotation_probability > 0.0:
        mri_transforms += [
            RandomRotation(
                degrees=random_rotation_degrees,
                p=random_rotation_probability,
                keys_to_rotate=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if random_flip_probability > 0.0:
        mri_transforms += [
            RandomFlip(
                flip=random_flip_type,
                p=random_flip_probability,
                keys_to_flip=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if padding_eps > 0.0:
        mri_transforms += [
            ComputeZeroPadding(KspaceKey.KSPACE, "padding", padding_eps),
            ApplyZeroPadding(KspaceKey.KSPACE, "padding"),
        ]
    if mask_func:
        mri_transforms += [
            CreateSamplingMask(
                mask_func,
                shape=(None if (isinstance(crop, str)) else crop),
                use_seed=use_seed,
                return_acs=True,
            ),
        ]
    mri_transforms += [PadCoilDimension(pad_coils=pad_coils, key=KspaceKey.KSPACE)]
    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))

    return Compose(mri_transforms)


def build_post_mri_transforms(
    backward_operator: Callable,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.RSS_ESTIMATE,
    sensitivity_maps_gaussian: float | None = None,
    sensitivity_maps_espirit_threshold: float | None = 0.05,
    sensitivity_maps_espirit_kernel_size: int | None = 6,
    sensitivity_maps_espirit_crop: float | None = 0.95,
    sensitivity_maps_espirit_max_iters: int | None = 30,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.RSS,
    scaling_key: TransformKey = TransformKey.MASKED_KSPACE,
    scale_percentile: float | None = 0.99,
) -> DirectTransform:
    """Builds post (can be put on gpu) supervised MRI transforms.

    More specifically, the following transformations are applied:

    *   Adds coil sensitivities and / or the body coil_image
    *   Masks the fully sampled k-space, if there is a mask function or a mask in the sample.
    *   Computes a scaling factor based on the masked k-space and normalizes data.
    *   Computes a target (image).
    *   Deletes the acs mask and the fully sampled k-space if requested.

    Args:
        backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        estimate_sensitivity_maps: Estimate sensitivity maps using the acs region. Default is ``True``.
        sensitivity_maps_type: Can be SensitivityMapType.RSS_ESTIMATE, SensitivityMapType.UNIT or
            SensitivityMapType.ESPIRIT. Will be ignored if `estimate_sensitivity_maps` is equal to False. Default is
            ``SensitivityMapType.RSS_ESTIMATE``.
        sensitivity_maps_gaussian: Optional sigma for gaussian weighting of sensitivity map.
        sensitivity_maps_espirit_threshold: Threshold for the calibration matrix when `type_of_map` is equal to "espirit".
            Default is ``0.05``.
        sensitivity_maps_espirit_kernel_size: Kernel size for the calibration matrix when `type_of_map` is equal to
            "espirit". Default is ``6``.
        sensitivity_maps_espirit_crop: Output eigenvalue cropping threshold when `type_of_map` is equal to "espirit".
            Default is ``0.95``.
        sensitivity_maps_espirit_max_iters: Power method iterations when `type_of_map` is equal to "espirit". Default is
            ``30``.
        delete_acs_mask: If True will delete key `acs_mask`. Default is ``True``.
        delete_kspace: If True will delete key `kspace` (fully sampled k-space). Default is ``True``.
        image_recon_type: Type to reconstruct target image. Default is ``ReconstructionType.RSS``.
        scaling_key: Key in sample to scale scalable items in sample. Default is ``TransformKey.MASKED_KSPACE``.
        scale_percentile: Data will be rescaled with the given percentile. If None, the division is done by the maximum.
            Default is 0.99 the same mask every time. Default: True.

    Returns:
        An MRI transformation object.
    """
    mri_transforms: list[Callable] = []

    if estimate_sensitivity_maps:
        mri_transforms += [
            EstimateSensitivityMapModule(
                kspace_key=KspaceKey.KSPACE,
                backward_operator=backward_operator,
                type_of_map=sensitivity_maps_type,
                gaussian_sigma=sensitivity_maps_gaussian,
                espirit_threshold=sensitivity_maps_espirit_threshold,
                espirit_kernel_size=sensitivity_maps_espirit_kernel_size,
                espirit_crop=sensitivity_maps_espirit_crop,
                espirit_max_iters=sensitivity_maps_espirit_max_iters,
            )
        ]

    if delete_acs_mask:
        mri_transforms += [DeleteKeysModule(keys=["acs_mask"])]

    mri_transforms += [
        ComputeImageModule(
            kspace_key=KspaceKey.KSPACE,
            target_key="target",
            backward_operator=backward_operator,
            type_reconstruction=image_recon_type,
        ),
        ApplyMaskModule(
            sampling_mask_key="sampling_mask",
            input_kspace_key=KspaceKey.KSPACE,
            target_kspace_key=KspaceKey.MASKED_KSPACE,
        ),
    ]

    mri_transforms += [
        ComputeScalingFactorModule(
            normalize_key=scaling_key,
            percentile=scale_percentile,
            scaling_factor_key=TransformKey.SCALING_FACTOR,
        ),
        NormalizeModule(scaling_factor_key=TransformKey.SCALING_FACTOR),
    ]
    if delete_kspace:
        mri_transforms += [DeleteKeysModule(keys=[KspaceKey.KSPACE])]

    return Compose(mri_transforms)


class RegistrationSimulateReferenceType(DirectEnum):
    """RegistrationSimulateReferenceType."""

    FROM_KEY = "from_key"
    ELASTIC = "elastic"


# pylint: disable=too-many-arguments
def build_supervised_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Callable | None,
    target_acceleration: float | None = None,
    dynamic_mask: bool | None = None,
    crop: tuple[int, int] | str | None = None,
    crop_type: str | None = "uniform",
    rescale: tuple[int, int] | list[int] | None = None,
    rescale_mode: RescaleMode | None = RescaleMode.NEAREST,
    rescale_2d_if_3d: bool | None = False,
    pad: tuple[int, int] | list[int] | None = None,
    image_center_crop: bool = True,
    random_rotation_degrees: Sequence[int] | None = (-90, 90),
    random_rotation_probability: float = 0.0,
    random_flip_type: RandomFlipType | None = RandomFlipType.RANDOM,
    random_flip_probability: float = 0.0,
    random_reverse_probability: float = 0.0,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.RSS_ESTIMATE,
    sensitivity_maps_gaussian: float | None = None,
    sensitivity_maps_espirit_threshold: float | None = 0.05,
    sensitivity_maps_espirit_kernel_size: int | None = 6,
    sensitivity_maps_espirit_crop: float | None = 0.95,
    sensitivity_maps_espirit_max_iters: int | None = 30,
    use_acs_as_mask: bool = False,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.RSS,
    compress_coils: int | None = None,
    pad_coils: int | None = None,
    scaling_key: TransformKey = TransformKey.MASKED_KSPACE,
    scale_percentile: float | None = 0.99,
    registration: bool = False,
    registration_simulate_reference: RegistrationSimulateReferenceType | None = None,
    registration_simulate_elastic_sigma: float = 3.0,
    registration_simulate_elastic_points: int = 3,
    registration_simulate_elastic_rotate: float = 0.0,
    registration_simulate_elastic_zoom: float = 0.0,
    registration_estimate_displacement: bool = True,
    registration_simulate_reference_from_key_index: int = 0,
    registration_moving_key: TransformKey = TransformKey.TARGET,
    demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
    demons_num_iterations: int = 100,
    demons_smooth_displacement_field: bool = True,
    demons_standard_deviations: float = 1.5,
    demons_intensity_difference_threshold: float | None = None,
    demons_maximum_rms_error: float | None = None,
    use_seed: bool = True,
) -> Compose:
    r"""Builds supervised MRI transforms.

    More specifically, the following transformations are applied:

    *   Converts input to (complex-valued) tensor.
    *   Applies k-space (center) crop if requested.
    *   Applies k-space rescaling if requested.
    *   Applies k-space padding if requested.
    *   Applies random augmentations (rotation, flip, reverse) if requested.
    *   Adds a sampling mask if `mask_func` is defined.
    *   Compreses the coil dimension if requested.
    *   Pads the coil dimension if requested.
    *   Adds coil sensitivities and / or the body coil_image
    *   Masks the fully sampled k-space, if there is a mask function or a mask in the sample.
    *   Computes a scaling factor based on the masked k-space and normalizes data.
    *   Computes a target (image).
    *   Deletes the acs mask and the fully sampled k-space if requested.

    Args:
        forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered).
        backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        mask_func: A function which creates a sampling mask of the appropriate shape.
        target_acceleration: Target acceleration factor. Default is ``None``.
        crop: If not None, this will transform the "kspace" to an image domain, crop it, and transform it back. If a tuple
            of integers is given then it will crop the backprojected kspace to that size. If "reconstruction_size" is given,
            then it will crop the backprojected kspace according to it, but a key "reconstruction_size" must be present in the
            sample. Default is ``None``.
        crop_type: Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default is
            ``"uniform"``.
        rescale: If not None, this will transform the "kspace" to the image domain, rescale it, and transform it back. Must
            correspond to (height, width). This is ignored if `rescale` is None. Default is None. It is not recommended to be
            used in combination with `crop`.
        rescale_mode: Mode to be used for rescaling. Can be RescaleMode.AREA, RescaleMode.BICUBIC, RescaleMode.BILINEAR,
            RescaleMode.NEAREST, RescaleMode.NEAREST_EXACT, or RescaleMode.TRILINEAR. Note that not all modes are supported for
            2D or 3D data. Default is ``RescaleMode.NEAREST``.
        rescale_2d_if_3d: If True and k-space data is 3D, rescaling will be done only on the height and width dimensions, by
            combining the slice/time dimension with the batch dimension. This is ignored if `rescale` is None. Default is
            ``False``.
        pad: If not None, this will zero-pad the "kspace" to the given size. Must correspond to (height, width) or
            (slice/time, height, width). Default is ``None``.
        image_center_crop: If True the backprojected kspace will be cropped around the center, otherwise randomly. This will
            be ignored if `crop` is None. Default is ``True``.
        random_rotation_degrees: Default is ``(-90, 90)``.
        random_rotation_probability: If greater than 0.0, random rotations will be applied of `random_rotation_degrees`
            degrees, with probability `random_rotation_probability`. Default is ``0.0``.
        random_flip_type: Default is ``RandomFlipType.RANDOM``.
        random_flip_probability: If greater than 0.0, random rotation of `random_flip_type` type, with probability
            `random_flip_probability`. Default is ``0.0``.
        random_reverse_probability: If greater than 0.0, will perform random reversion along the time or slice dimension (2)
            with probability `random_reverse_probability`. Default is ``0.0``.
        padding_eps: Padding epsilon. Default is ``0.0001``.
        estimate_body_coil_image: Estimate body coil image. Default is ``False``.
        estimate_sensitivity_maps: Estimate sensitivity maps using the acs region. Default is ``True``.
        sensitivity_maps_type: Can be SensitivityMapType.RSS_ESTIMATE, SensitivityMapType.UNIT or
            SensitivityMapType.ESPIRIT. Will be ignored if `estimate_sensitivity_maps` is False. Default is
            ``SensitivityMapType.RSS_ESTIMATE``.
        sensitivity_maps_gaussian: Optional sigma for gaussian weighting of sensitivity map.
        sensitivity_maps_espirit_threshold: Threshold for the calibration matrix when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``0.05``.
        sensitivity_maps_espirit_kernel_size: Kernel size for the calibration matrix when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``6``.
        sensitivity_maps_espirit_crop: Output eigenvalue cropping threshold when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``0.95``.
        sensitivity_maps_espirit_max_iters: Power method iterations when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``30``.
        use_acs_as_mask: If True, will use the acs region as the mask. Default is ``False``.
        delete_acs_mask: If True will delete key `acs_mask`. Default is ``True``.
        delete_kspace: If True will delete key `kspace` (fully sampled k-space). Default is ``True``.
        image_recon_type: Type to reconstruct target image. Default is ``ReconstructionType.RSS``.
        compress_coils: Number of coils to compress input k-space. It is not recommended to be used in combination with
            `pad_coils`. Default is ``None``.
        pad_coils: Number of coils to pad data to.
        scaling_key: Key in sample to scale scalable items in sample. Default is ``TransformKey.MASKED_KSPACE``.
        scale_percentile: Data will be rescaled with the given percentile. If None, the division is done by the maximum.
            Default is ``0.99``.
        registration: If True, will compute a displacement field between the target and the moving image. Default is
            ``False``.
        registration_simulate_reference: If not None, will simulate a reference image for displacement field computation.
            Otherwise, this expects a key in the sample.  Can be RegistrationSimulateReferenceType.FROM_KEY or
            RegistrationSimulateReferenceType.ELASTIC. Default is ``None``.
        registration_simulate_elastic_sigma: Standard deviation for the elastic simulation. Default is ``3.0``.
        registration_simulate_elastic_points: Number of points for the elastic simulation. Default is ``3``.
        registration_simulate_elastic_rotate: Rotation for the elastic simulation. Default is ``0.0``.
        registration_estimate_displacement: If True, will estimate the displacement field between the target and the moving
            image using the demons algorithm. Default is ``True``.
        registration_simulate_elastic_zoom: Zoom for the elastic simulation. Default is ``0.0``.
        registration_simulate_reference_from_key_index: Index to drop from the key to simulate the reference image. Default
            is ``0``.
        demons_filter_type: Type of filter to apply to the displacement field. Default is
            ``DemonsFilterType.SYMMETRIC_FORCES``.
        demons_num_iterations: Number of iterations for the demons algorithm. Default is ``100``.
        demons_smooth_displacement_field: If True, will smooth the displacement field. Default is ``True``.
        demons_standard_deviations: Standard deviation for the smoothing of the displacement field. Default is ``1.5``.
        demons_intensity_difference_threshold: Intensity difference threshold for the demons algorithm. Default is ``None``.
        demons_maximum_rms_error: Maximum RMS error for the demons algorithm. Default is ``None``.
        use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default is ``True``.

    Returns:
        An MRI transformation object.
    """
    mri_transforms: list[Callable] = [ToTensor()]
    if crop:
        mri_transforms += [
            CropKspace(
                crop=crop,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                image_space_center_crop=image_center_crop,
                random_crop_sampler_type=crop_type,
                random_crop_sampler_use_seed=use_seed,
            )
        ]
    if rescale:
        mri_transforms += [
            RescaleKspace(
                shape=rescale,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                rescale_mode=rescale_mode,
                rescale_2d_if_3d=rescale_2d_if_3d,
                kspace_key=KspaceKey.KSPACE,
            )
        ]
    if pad:
        mri_transforms += [
            PadKspace(
                pad_shape=pad,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                kspace_key=KspaceKey.KSPACE,
            )
        ]
    if random_rotation_probability > 0.0:
        mri_transforms += [
            RandomRotation(
                degrees=random_rotation_degrees,
                p=random_rotation_probability,
                keys_to_rotate=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if random_flip_probability > 0.0:
        mri_transforms += [
            RandomFlip(
                flip=random_flip_type,
                p=random_flip_probability,
                keys_to_flip=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if random_reverse_probability > 0.0:
        mri_transforms += [
            RandomReverse(
                p=random_reverse_probability,
                keys_to_reverse=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if padding_eps > 0.0:
        mri_transforms += [
            ComputeZeroPadding(KspaceKey.KSPACE, "padding", padding_eps),
            ApplyZeroPadding(KspaceKey.KSPACE, "padding"),
        ]
    if mask_func:
        mri_transforms += [
            CreateSamplingMask(
                mask_func,
                shape=(None if (isinstance(crop, str)) else crop),
                use_seed=use_seed,
                return_acs=estimate_sensitivity_maps,
                dynamic_mask=dynamic_mask,
            ),
        ]
    if use_acs_as_mask:
        mri_transforms += [CopyKeys(keys=[TransformKey.ACS_MASK], new_keys=[TransformKey.SAMPLING_MASK])]
    if target_acceleration:
        mri_transforms += [AddTargetAcceleration(target_acceleration)]
    if compress_coils:
        mri_transforms += [CompressCoil(num_coils=compress_coils, kspace_key=KspaceKey.KSPACE)]
    if pad_coils:
        mri_transforms += [PadCoilDimension(pad_coils=pad_coils, key=KspaceKey.KSPACE)]

    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))
    # Paper / adpt order: estimate coil sensitivities from full k-space (ACS region via
    # acs_mask inside the module), then apply the sampling mask. ACS k-space is only
    # materialized when needed for scaling (scaling_key=acs_kspace).
    if estimate_sensitivity_maps:
        mri_transforms += [
            EstimateSensitivityMap(
                kspace_key=KspaceKey.KSPACE,
                backward_operator=backward_operator,
                type_of_map=sensitivity_maps_type,
                gaussian_sigma=sensitivity_maps_gaussian,
                espirit_threshold=sensitivity_maps_espirit_threshold,
                espirit_kernel_size=sensitivity_maps_espirit_kernel_size,
                espirit_crop=sensitivity_maps_espirit_crop,
                espirit_max_iters=sensitivity_maps_espirit_max_iters,
            )
        ]
    mri_transforms += [
        ApplyMask(
            sampling_mask_key=TransformKey.SAMPLING_MASK,
            input_kspace_key=KspaceKey.KSPACE,
            target_kspace_key=KspaceKey.MASKED_KSPACE,
        ),
    ]
    if scaling_key == TransformKey.ACS_KSPACE:
        mri_transforms += [
            ApplyMask(
                sampling_mask_key=TransformKey.ACS_MASK,
                input_kspace_key=KspaceKey.KSPACE,
                target_kspace_key=KspaceKey.ACS_KSPACE,
            ),
        ]
    if registration and registration_simulate_reference is not None:
        mri_transforms += [
            DropIndex(
                keys=[
                    TransformKey.KSPACE,
                    TransformKey.ACS_KSPACE,
                    TransformKey.MASKED_KSPACE,
                    TransformKey.ACS_MASK,
                    TransformKey.SAMPLING_MASK,
                    TransformKey.PADDING,
                    TransformKey.SENSITIVITY_MAP,
                    TransformKey.ACCELERATION,
                    TransformKey.CENTER_FRACTION,
                ],
                index=registration_simulate_reference_from_key_index,
                index_dim=1,
                store_deleted_keys=[TransformKey.REFERENCE_KSPACE],
            )
        ]
    mri_transforms += [
        ComputeScalingFactor(
            normalize_key=scaling_key,
            percentile=scale_percentile,
            scaling_factor_key=TransformKey.SCALING_FACTOR,
        ),
        # Match paper Normalize defaults (sensitivity maps are scale-invariant).
        Normalize(scaling_factor_key=None if scale_percentile is None else TransformKey.SCALING_FACTOR),
    ]
    mri_transforms += [
        ComputeImage(
            kspace_key=KspaceKey.KSPACE,
            target_key=TransformKey.TARGET,
            backward_operator=backward_operator,
            type_reconstruction=image_recon_type,
        )
    ]
    if registration:
        if registration_simulate_reference is not None:
            mri_transforms += [
                ComputeImage(
                    kspace_key=KspaceKey.REFERENCE_KSPACE,
                    target_key=TransformKey.REFERENCE_IMAGE,
                    backward_operator=backward_operator,
                    type_reconstruction=image_recon_type,
                ),
                SqueezeKey(keys=[TransformKey.REFERENCE_IMAGE], dim=0),
            ]
            if registration_simulate_reference == RegistrationSimulateReferenceType.ELASTIC:
                mri_transforms += [
                    RandomElasticDeformation(
                        image_key=TransformKey.REFERENCE_IMAGE,
                        target_key=TransformKey.REFERENCE_IMAGE,
                        use_seed=use_seed,
                        sigma=registration_simulate_elastic_sigma,
                        points=registration_simulate_elastic_points,
                        rotate=registration_simulate_elastic_rotate,
                        zoom=registration_simulate_elastic_zoom,
                    )
                ]
        if registration_estimate_displacement:
            mri_transforms += [
                Displacement(
                    transform_type=DisplacementTransformType.MULTISCALE_DEMONS,
                    demons_filter_type=demons_filter_type,
                    demons_num_iterations=demons_num_iterations,
                    demons_smooth_displacement_field=demons_smooth_displacement_field,
                    demons_standard_deviations=demons_standard_deviations,
                    demons_intensity_difference_threshold=demons_intensity_difference_threshold,
                    demons_maximum_rms_error=demons_maximum_rms_error,
                    reference_image_key=TransformKey.REFERENCE_IMAGE,
                    moving_image_key=registration_moving_key,
                )
            ]
    if delete_acs_mask:
        mri_transforms += [DeleteKeys(keys=[TransformKey.ACS_MASK, KspaceKey.ACS_KSPACE])]
    if delete_kspace:
        mri_transforms += [DeleteKeys(keys=[KspaceKey.KSPACE])]

    return Compose(mri_transforms)


class TransformsType(DirectEnum):
    """TransformsType."""

    SUPERVISED = "supervised"
    SSL_SSDU = "ssl_ssdu"


# pylint: disable=too-many-arguments
def build_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Callable | None,
    target_acceleration: float | None = None,
    dynamic_mask: bool | None = None,
    crop: tuple[int, int] | str | None = None,
    crop_type: str | None = "uniform",
    rescale: tuple[int, int] | list[int] | None = None,
    rescale_mode: RescaleMode | None = RescaleMode.NEAREST,
    rescale_2d_if_3d: bool | None = False,
    pad: tuple[int, int] | list[int] | None = None,
    image_center_crop: bool = True,
    random_rotation_degrees: Sequence[int] | None = (-90, 90),
    random_rotation_probability: float = 0.0,
    random_flip_type: RandomFlipType | None = RandomFlipType.RANDOM,
    random_flip_probability: float = 0.0,
    random_reverse_probability: float = 0.0,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.RSS_ESTIMATE,
    sensitivity_maps_gaussian: float | None = None,
    sensitivity_maps_espirit_threshold: float | None = 0.05,
    sensitivity_maps_espirit_kernel_size: int | None = 6,
    sensitivity_maps_espirit_crop: float | None = 0.95,
    sensitivity_maps_espirit_max_iters: int | None = 30,
    use_acs_as_mask: bool = False,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.RSS,
    compress_coils: int | None = None,
    pad_coils: int | None = None,
    scaling_key: TransformKey = TransformKey.MASKED_KSPACE,
    scale_percentile: float | None = 0.99,
    registration: bool = False,
    registration_simulate_reference: RegistrationSimulateReferenceType | None = None,
    registration_simulate_elastic_sigma: float = 3.0,
    registration_simulate_elastic_points: int = 3,
    registration_simulate_elastic_rotate: float = 0.0,
    registration_simulate_elastic_zoom: float = 0.0,
    registration_estimate_displacement: bool = True,
    registration_simulate_reference_from_key_index: int = 0,
    registration_moving_key: TransformKey = TransformKey.TARGET,
    demons_filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
    demons_num_iterations: int = 100,
    demons_smooth_displacement_field: bool = True,
    demons_standard_deviations: float = 1.5,
    demons_intensity_difference_threshold: float | None = None,
    demons_maximum_rms_error: float | None = None,
    use_seed: bool = True,
    transforms_type: TransformsType | None = TransformsType.SUPERVISED,
    mask_split_ratio: float | list[float] | tuple[float, ...] = 0.4,
    mask_split_acs_region: list[int] | tuple[int, int] = (0, 0),
    mask_split_keep_acs: bool | None = False,
    mask_split_type: MaskSplitterType = MaskSplitterType.GAUSSIAN,
    mask_split_gaussian_std: float = 3.0,
    mask_split_half_direction: HalfSplitType = HalfSplitType.VERTICAL,
) -> DirectTransform:
    r"""Build transforms for MRI.

    More specifically, the following transformations are applied:

    *   Converts input to (complex-valued) tensor.
    *   Applies k-space (center) crop if requested.
    *   Applies k-space rescaling if requested.
    *   Applies k-space padding if requested.
    *   Applies random augmentations (rotation, flip, reverse) if requested.
    *   Adds a sampling mask if `mask_func` is defined.
    *   Compreses the coil dimension if requested.
    *   Pads the coil dimension if requested.
    *   Adds coil sensitivities and / or the body coil_image
    *   Masks the fully sampled k-space, if there is a mask function or a mask in the sample.
    *   Computes a scaling factor based on the masked k-space and normalizes data.
    *   Computes a target (image).
    *   Deletes the acs mask and the fully sampled k-space if requested.
    *   Splits the mask if requested for self-supervised learning.

    Args:
        forward_operator: The forward operator, e.g. some form of FFT (centered or uncentered).
        backward_operator: The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        mask_func: A function which creates a sampling mask of the appropriate shape.
        target_acceleration: Target acceleration factor. Default is ``None``.
        crop: If not None, this will transform the "kspace" to an image domain, crop it, and transform it back. If a tuple
            of integers is given then it will crop the backprojected kspace to that size. If "reconstruction_size" is given,
            then it will crop the backprojected kspace according to it, but a key "reconstruction_size" must be present in the
            sample. Default is ``None``.
        crop_type: Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default is
            ``"uniform"``.
        rescale: If not None, this will transform the "kspace" to the image domain, rescale it, and transform it back. Must
            correspond to (height, width). This is ignored if `rescale` is None. Default is None. It is not recommended to be
            used in combination with `crop`.
        rescale_mode: Mode to be used for rescaling. Can be RescaleMode.AREA, RescaleMode.BICUBIC, RescaleMode.BILINEAR,
            RescaleMode.NEAREST, RescaleMode.NEAREST_EXACT, or RescaleMode.TRILINEAR. Note that not all modes are supported for
            2D or 3D data. Default is ``RescaleMode.NEAREST``.
        rescale_2d_if_3d: If True and k-space data is 3D, rescaling will be done only on the height and width dimensions, by
            combining the slice/time dimension with the batch dimension. This is ignored if `rescale` is None. Default is
            ``False``.
        pad: If not None, this will zero-pad the "kspace" to the given size. Must correspond to (height, width) or
            (slice/time, height, width). Default is ``None``.
        image_center_crop: If True the backprojected kspace will be cropped around the center, otherwise randomly. This will
            be ignored if `crop` is None. Default is ``True``.
        random_rotation_degrees: Default is ``(-90, 90)``.
        random_rotation_probability: If greater than 0.0, random rotations will be applied of `random_rotation_degrees`
            degrees, with probability `random_rotation_probability`. Default is ``0.0``.
        random_flip_type: Default is ``RandomFlipType.RANDOM``.
        random_flip_probability: If greater than 0.0, random rotation of `random_flip_type` type, with probability
            `random_flip_probability`. Default is ``0.0``.
        random_reverse_probability: If greater than 0.0, will perform random reversion along the time or slice dimension (2)
            with probability `random_reverse_probability`. Default is ``0.0``.
        padding_eps: Padding epsilon. Default is ``0.0001``.
        estimate_body_coil_image: Estimate body coil image. Default is ``False``.
        estimate_sensitivity_maps: Estimate sensitivity maps using the acs region. Default is ``True``.
        sensitivity_maps_type: Can be SensitivityMapType.RSS_ESTIMATE, SensitivityMapType.UNIT or
            SensitivityMapType.ESPIRIT. Will be ignored if `estimate_sensitivity_maps` is False. Default is
            ``SensitivityMapType.RSS_ESTIMATE``.
        sensitivity_maps_gaussian: Optional sigma for gaussian weighting of sensitivity map.
        sensitivity_maps_espirit_threshold: Threshold for the calibration matrix when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``0.05``.
        sensitivity_maps_espirit_kernel_size: Kernel size for the calibration matrix when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``6``.
        sensitivity_maps_espirit_crop: Output eigenvalue cropping threshold when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``0.95``.
        sensitivity_maps_espirit_max_iters: Power method iterations when `type_of_map` is set to
            `SensitivityMapType.ESPIRIT`. Default is ``30``.
        use_acs_as_mask: If True, will use the acs region as the mask. Default is ``False``.
        delete_acs_mask: If True will delete key `acs_mask`. Default is ``True``.
        delete_kspace: If True will delete key `kspace` (fully sampled k-space). Default is ``True``.
        image_recon_type: Type to reconstruct target image. Default is ``ReconstructionType.RSS``.
        compress_coils: Number of coils to compress input k-space. It is not recommended to be used in combination with
            `pad_coils`. Default is ``None``.
        pad_coils: Number of coils to pad data to.
        scaling_key: Key in sample to scale scalable items in sample. Default is ``TransformKey.MASKED_KSPACE``.
        scale_percentile: Data will be rescaled with the given percentile. If None, the division is done by the maximum.
            Default is ``0.99``.
        registration: If True, will compute a displacement field between the target and the moving image. Default is
            ``False``.
        registration_simulate_reference: If not None, will simulate a reference image for displacement field computation.
            Otherwise, this expects a key in the sample.  Can be RegistrationSimulateReferenceType.FROM_KEY or
            RegistrationSimulateReferenceType.ELASTIC. Default is ``None``.
        registration_simulate_elastic_sigma: Standard deviation for the elastic simulation. Default is ``3.0``.
        registration_simulate_elastic_points: Number of points for the elastic simulation. Default is ``3``.
        registration_simulate_elastic_rotate: Rotation for the elastic simulation. Default is ``0.0``.
        registration_simulate_elastic_zoom: Zoom for the elastic simulation. Default is ``0.0``.
        registration_estimate_displacement: If True, will estimate the displacement field between the target and the moving
            image using the demons algorithm. Default is ``True``.
        registration_simulate_reference_from_key_index: Index to drop from the key to simulate the reference image. Default
            is ``0``.
        registration_moving_key: Key in sample to compute displacement field from. Default is ``TransformKey.TARGET``.
        demons_filter_type: Type of filter to apply to the displacement field. Default is
            ``DemonsFilterType.SYMMETRIC_FORCES``.
        demons_num_iterations: Number of iterations for the demons algorithm. Default is ``100``.
        demons_smooth_displacement_field: If True, will smooth the displacement field. Default is ``True``.
        demons_standard_deviations: Standard deviation for the smoothing of the displacement field. Default is ``1.5``.
        demons_intensity_difference_threshold: Intensity difference threshold for the demons algorithm. Default is ``None``.
        demons_maximum_rms_error: Maximum RMS error for the demons algorithm. Default is ``None``.
        use_seed: If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default is ``True``.
        transforms_type: Can be `TransformsType.SUPERVISED` for supervised learning transforms or `TransformsType.SSL_SSDU`
            for self-supervised learning transforms. Default is ``TransformsType.SUPERVISED``.
        mask_split_ratio: The ratio(s) of the sampling mask splitting. If `transforms_type` is TransformsKey.SUPERVISED,
            this is ignored.
        mask_split_acs_region: A rectangle for the acs region that will be used in the input mask. This applies only if
            `transforms_type` is set to TransformsKey.SSL_SSDU. Default is ``(0, 0)``.
        mask_split_keep_acs: If True, acs region according to the "acs_mask" of the sample will be used in both mask splits.
            This applies only if `transforms_type` is set to TransformsKey.SSL_SSDU. Default is ``False``.
        mask_split_type: How the sampling mask will be split. Can be MaskSplitterType.UNIFORM, MaskSplitterType.GAUSSIAN, or
            MaskSplitterType.HALF. Default is MaskSplitterType.GAUSSIAN. This applies only if `transforms_type` is set to
            TransformsKey.SSL_SSDU. Default: MaskSplitterType.GAUSSIAN.
        mask_split_gaussian_std: Standard deviation of gaussian mask splitting. This applies only if `transforms_type` is
            set to TransformsKey.SSL_SSDU. Ignored if `mask_split_type` is not set to MaskSplitterType.GAUSSIAN. Default is
            ``3.0``.
        mask_split_half_direction: Split type if `mask_split_type` is `MaskSplitterType.HALF`. Can be
            `HalfSplitType.VERTICAL`, `HalfSplitType.HORIZONTAL`, `HalfSplitType.DIAGONAL_LEFT` or
            `HalfSplitType.DIAGONAL_RIGHT`. This applies only if `transforms_type` is set to `TransformsKey.SSL_SSDU`. Ignored
            if `mask_split_type` is not set to `MaskSplitterType.HALF`. Default is ``HalfSplitType.VERTICAL``.

    Returns:
        An MRI transformation object.
    """
    logger = logging.getLogger(build_mri_transforms.__name__)
    logger.info("Creating %s MRI transforms.", transforms_type)

    if crop and rescale:
        logger.warning(
            "Rescale and crop are both given. Rescale will be applied after cropping. This is not recommended."
        )

    if compress_coils and pad_coils:
        logger.warning(
            "Compress coils and pad coils are both given. Compress coils will be applied before padding. "
            "This is not recommended."
        )

    mri_transforms = build_supervised_mri_transforms(
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        mask_func=mask_func,
        target_acceleration=target_acceleration,
        dynamic_mask=dynamic_mask,
        crop=crop,
        crop_type=crop_type,
        rescale=rescale,
        rescale_mode=rescale_mode,
        rescale_2d_if_3d=rescale_2d_if_3d,
        pad=pad,
        image_center_crop=image_center_crop,
        random_rotation_degrees=random_rotation_degrees,
        random_rotation_probability=random_rotation_probability,
        random_flip_type=random_flip_type,
        random_flip_probability=random_flip_probability,
        random_reverse_probability=random_reverse_probability,
        padding_eps=padding_eps,
        estimate_sensitivity_maps=estimate_sensitivity_maps,
        sensitivity_maps_type=sensitivity_maps_type,
        estimate_body_coil_image=estimate_body_coil_image,
        sensitivity_maps_gaussian=sensitivity_maps_gaussian,
        sensitivity_maps_espirit_threshold=sensitivity_maps_espirit_threshold,
        sensitivity_maps_espirit_kernel_size=sensitivity_maps_espirit_kernel_size,
        sensitivity_maps_espirit_crop=sensitivity_maps_espirit_crop,
        sensitivity_maps_espirit_max_iters=sensitivity_maps_espirit_max_iters,
        use_acs_as_mask=use_acs_as_mask,
        delete_acs_mask=(delete_acs_mask if transforms_type == TransformsType.SUPERVISED else False),
        delete_kspace=(delete_kspace if transforms_type == TransformsType.SUPERVISED else False),
        image_recon_type=image_recon_type,
        compress_coils=compress_coils,
        pad_coils=pad_coils,
        scaling_key=scaling_key,
        scale_percentile=scale_percentile,
        registration=registration,
        registration_simulate_reference=registration_simulate_reference,
        registration_simulate_elastic_sigma=registration_simulate_elastic_sigma,
        registration_simulate_elastic_points=registration_simulate_elastic_points,
        registration_simulate_elastic_rotate=registration_simulate_elastic_rotate,
        registration_simulate_elastic_zoom=registration_simulate_elastic_zoom,
        registration_estimate_displacement=registration_estimate_displacement,
        registration_simulate_reference_from_key_index=registration_simulate_reference_from_key_index,
        registration_moving_key=registration_moving_key,
        demons_filter_type=demons_filter_type,
        demons_num_iterations=demons_num_iterations,
        demons_smooth_displacement_field=demons_smooth_displacement_field,
        demons_standard_deviations=demons_standard_deviations,
        demons_intensity_difference_threshold=demons_intensity_difference_threshold,
        demons_maximum_rms_error=demons_maximum_rms_error,
        use_seed=use_seed,
    ).transforms

    mri_transforms = list(mri_transforms) + [
        AddBooleanKeysModule(["is_ssl"], [transforms_type != TransformsType.SUPERVISED])
    ]

    if transforms_type == TransformsType.SUPERVISED:
        return Compose(mri_transforms)

    mask_splitter_kwargs: dict[str, Any] = {
        "ratio": mask_split_ratio,
        "acs_region": mask_split_acs_region,
        "keep_acs": mask_split_keep_acs,
        "use_seed": use_seed,
        "kspace_key": KspaceKey.MASKED_KSPACE,
    }
    mri_transforms += [
        (
            GaussianMaskSplitter(**mask_splitter_kwargs, std_scale=mask_split_gaussian_std)
            if mask_split_type == MaskSplitterType.GAUSSIAN
            else (
                UniformMaskSplitter(**mask_splitter_kwargs)
                if mask_split_type == MaskSplitterType.UNIFORM
                else HalfMaskSplitterModule(
                    **{k: v for k, v in mask_splitter_kwargs.items() if k != "ratio"},
                    direction=mask_split_half_direction,
                )
            )
        ),
        DeleteKeys([TransformKey.ACS_MASK]),
    ]

    mri_transforms += [
        RenameKeys(
            [
                SSLTransformMaskPrefixes.INPUT_ + TransformKey.MASKED_KSPACE,
                SSLTransformMaskPrefixes.TARGET_ + TransformKey.MASKED_KSPACE,
            ],
            ["input_kspace", "kspace"],
        ),
        DeleteKeys(["masked_kspace", "sampling_mask"]),
    ]  # Rename keys for SSL engine

    mri_transforms += [
        ComputeImage(
            kspace_key=KspaceKey.KSPACE,
            target_key=TransformKey.TARGET,
            backward_operator=backward_operator,
            type_reconstruction=image_recon_type,
        )
    ]

    return Compose(mri_transforms)
