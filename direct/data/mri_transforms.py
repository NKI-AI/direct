# Copyright (c) DIRECT Contributors

"""direct.data.mri_transforms module."""

from __future__ import annotations

import copy
import functools
import logging
import random
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from direct.algorithms.mri_algorithms import EspiritCalibration
from direct.data import transforms as T
from direct.data.transforms import RescaleMode
from direct.exceptions import ItemNotFoundException
from direct.types import DirectEnum, IntegerListOrTupleString, KspaceKey, TransformKey
from direct.utils import DirectModule, DirectTransform
from direct.utils.asserts import assert_complex

logger = logging.getLogger(__name__)


class Compose(DirectTransform):
    """Compose several transformations together, for instance ClipAndScale and a flip.

    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
    additional dependencies.
    """

    def __init__(self, transforms: Iterable[Callable]) -> None:
        """Inits :class:`Compose`.

        Parameters
        ----------
        transforms: Iterable[Callable]
            List of transforms.
        """
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`Compose`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
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
        degrees: Sequence[int] = (-90, 90),
        p: float = 0.5,
        keys_to_rotate: Tuple[TransformKey, ...] = (TransformKey.KSPACE,),
    ):
        r"""Inits :class:`RandomRotation`.

        Parameters
        ----------
        degrees: sequence of ints
            Degrees of rotation. Must be a multiple of 90. If len(degrees) > 1, then a degree will be chosen at random.
            Default: (-90, 90).
        p: float
            Probability of rotation. Default: 0.5
        keys_to_rotate : tuple of TransformKeys
            Keys to rotate. Default: "kspace".
        """
        super().__init__()

        assert all(degree % 90 == 0 for degree in degrees)

        self.degrees = degrees
        self.p = p
        self.keys_to_rotate = keys_to_rotate

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`RandomRotation`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
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
        flip: RandomFlipType = RandomFlipType.RANDOM,
        p: float = 0.5,
        keys_to_flip: Tuple[TransformKey, ...] = (TransformKey.KSPACE,),
    ):
        r"""Inits :class:`RandomFlip`.

        Parameters
        ----------
        flip : RandomFlipType
            Horizontal, vertical, or random choice of the two. Default: RandomFlipType.RANDOM.
        p : float
            Probability of flip. Default: 0.5
        keys_to_flip : tuple of TransformKeys
            Keys to flip. Default: "kspace".
        """
        super().__init__()

        self.flip = flip
        self.p = p
        self.keys_to_flip = keys_to_flip

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`RandomFlip`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
            Sample with flipped values of `keys_to_flip`.
        """
        if random.SystemRandom().random() <= self.p:
            dims = (
                (-2,)
                if self.flip == "horizontal"
                else (-1,)
                if self.flip == "vertical"
                else (-2, -1)
                if self.flip == "both"
                else (random.SystemRandom().choice([-2, -1]),)
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
        keys_to_reverse: Tuple[TransformKey, ...] = (TransformKey.KSPACE,),
    ):
        r"""Inits :class:`RandomReverse`.

        Parameters
        ----------
        dim : int
            Dimension along to perform reversion. Typically, this is for time or slice dimension. Default: 2.
        p : float
            Probability of flip. Default: 0.5
        keys_to_reverse : tuple of TransformKeys
            Keys to reverse. Default: "kspace".
        """
        super().__init__()

        self.dim = dim
        self.p = p
        self.keys_to_reverse = keys_to_reverse

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`RandomReverse`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
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


class CreateSamplingMask(DirectTransform):
    """Data Transformer for training MRI reconstruction models.

    Creates sampling mask.
    """

    def __init__(
        self,
        mask_func: Callable,
        shape: Optional[Tuple[int, ...]] = None,
        use_seed: bool = True,
        return_acs: bool = False,
    ) -> None:
        """Inits :class:`CreateSamplingMask`.

        Parameters
        ----------
        mask_func: Callable
            A function which creates a sampling mask of the appropriate shape.
        shape: tuple, optional
            Sampling mask shape. Default: None.
        use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        return_acs: bool
            If True, it will generate an ACS mask. Default: False.
        """
        super().__init__()
        self.mask_func = mask_func
        self.shape = shape
        self.use_seed = use_seed
        self.return_acs = return_acs

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`CreateSamplingMask`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
            Sample with `sampling_mask` key.
        """
        if not self.shape:
            shape = sample["kspace"].shape[-3:]
        elif any(_ is None for _ in self.shape):  # Allow None as values.
            kspace_shape = list(sample["kspace"].shape[1:-1])
            shape = tuple(_ if _ else kspace_shape[idx] for idx, _ in enumerate(self.shape)) + (2,)
        else:
            shape = self.shape + (2,)

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))

        sampling_mask = self.mask_func(shape=shape, seed=seed, return_acs=False).to(sample["kspace"].dtype)

        if sample["kspace"].ndim == 5:
            sampling_mask = sampling_mask.unsqueeze(0)

        # Shape (1, [1], height, width, 1)
        sample["sampling_mask"] = sampling_mask

        if self.return_acs:
            acs_mask = self.mask_func(shape=shape, seed=seed, return_acs=True).to(sample["kspace"].dtype)
            if sample["kspace"].ndim == 5:
                acs_mask = acs_mask.unsqueeze(0)
            sample["acs_mask"] = acs_mask

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

        Parameters
        ----------
        sampling_mask_key: str
            Default: "sampling_mask".
        input_kspace_key: KspaceKey
            Default: KspaceKey.KSPACE.
        target_kspace_key: KspaceKey
            Default KspaceKey.MASKED_KSPACE.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.sampling_mask_key = sampling_mask_key
        self.input_kspace_key = input_kspace_key
        self.target_kspace_key = target_kspace_key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`ApplyMaskModule`.

        Applies mask with key `sampling_mask_key` onto kspace `input_kspace_key`. Result is stored as a tensor with
        key `target_kspace_key`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample containing keys `sampling_mask_key` and `input_kspace_key`.

        Returns
        -------
        Dict[str, Any]
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
        crop: Union[str, Tuple[int, ...], List[int]],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        image_space_center_crop: bool = False,
        random_crop_sampler_type: Optional[str] = "uniform",
        random_crop_sampler_use_seed: Optional[bool] = True,
        random_crop_sampler_gaussian_sigma: Optional[List[float]] = None,
    ) -> None:
        """Inits :class:`CropKspace`.

        Parameters
        ----------
        crop: tuple of ints or str
            Shape to crop the input to or a string pointing to a crop key (e.g. `reconstruction_size`).
        forward_operator: Callable
            The forward operator, e.g. some form of FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.fft2`.
        backward_operator: Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.ifft2`.
        image_space_center_crop: bool
            If set, the crop in the data will be taken in the center
        random_crop_sampler_type: Optional[str]
            If "uniform" the random cropping will be done by uniformly sampling `crop`, as opposed to `gaussian` which
            will sample from a gaussian distribution. If `image_space_center_crop` is True, then this is ignored.
            Default: "uniform".
        random_crop_sampler_use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume
            is cropped the same way. Default: True.
        random_crop_sampler_gaussian_sigma: Optional[List[float]]
            Standard variance of the gaussian when `random_crop_sampler_type` is `gaussian`.
            If `image_space_center_crop` is True, then this is ignored. Default: None.
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
                sampler=random_crop_sampler_type,
                sigma=random_crop_sampler_gaussian_sigma,
            )
            self.random_crop_sampler_use_seed = random_crop_sampler_use_seed

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`CropKspace`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample containing key `kspace`.

        Returns
        -------
        Dict[str, Any]
            Cropped and masked sample.
        """

        kspace = sample["kspace"]  # shape (coil, [slice], height, width, complex=2)

        dim = self.spatial_dims["2D"] if kspace.ndim == 4 else self.spatial_dims["3D"]

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

        cropper_data_list = [backprojected_kspace]
        if "sensitivity_map" in sample:
            cropper_data_list += [sample["sensitivity_map"]]
        cropper_args = {
            "data_list": cropper_data_list,
            "crop_shape": crop_shape,
            "contiguous": False,
        }
        if not self.image_space_center_crop:
            cropper_args["seed"] = (
                None if not self.random_crop_sampler_use_seed else tuple(map(ord, str(sample["filename"])))
            )
        cropped_output = self.crop_func(**cropper_args)
        if "sensitivity_map" in sample:
            cropped_backprojected_kspace, sensitivity_map = cropped_output
            sample["sensitivity_map"] = sensitivity_map
        else:
            cropped_backprojected_kspace = cropped_output

        if "sampling_mask" in sample:
            sample["sampling_mask"] = T.complex_center_crop(
                sample["sampling_mask"], (1,) + tuple(crop_shape)[1:] if kspace.ndim == 5 else crop_shape
            )
            sample["acs_mask"] = T.complex_center_crop(
                sample["acs_mask"], (1,) + tuple(crop_shape)[1:] if kspace.ndim == 5 else crop_shape
            )

        # Compute new k-space for the cropped_backprojected_kspace
        # shape (coil, [slice], new_height, new_width, complex=2)
        sample["kspace"] = self.forward_operator(cropped_backprojected_kspace, dim=dim)  # The cropped kspace

        return sample


class RescaleKspaceModule(DirectModule):
    """Rescale k-space (downsample/upsample) module.

    Rescales the k-space by:
    * It first projects the k-space to the image-domain via the backward operator,
    * It rescales the back-projected k-space to specified shape,
    * It transforms the rescaled back-projected k-space to the k-space domain via the forward operator.
    """

    def __init__(
        self,
        shape: Union[Tuple[int, int], List[int]],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        rescale_mode: RescaleMode = RescaleMode.BILINEAR,
        kspace_key: KspaceKey = KspaceKey.KSPACE,
    ) -> None:
        """Inits :class:`RescaleKspaceModule`.

        Parameters
        ----------
        shape : tuple or list of ints
            Shape to rescale the input. Must be correspond to (height, width).
        forward_operator : Callable
            The forward operator, e.g. some form of FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.fft2`.
        backward_operator : Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.ifft2`.
        rescale_mode : RescaleMode
            Mode to be used for rescaling. Can be RescaleMode.BILINEAR or RescaleMode.BICUBIC.
            Default: RescaleMode.BILINEAR.
        kspace_key : KspaceKey
            K-space key. Default: KspaceKey.KSPACE.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        if len(shape) != 2:
            raise ValueError(f"Shape should be a list or tuple of two integers. Received: {shape}.")
        self.shape = shape
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.rescale_mode = rescale_mode
        self.kspace_key = kspace_key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`RescaleKspaceModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample containing key `kspace`.

        Returns
        -------
        Dict[str, Any]
            Cropped and masked sample.
        """
        kspace = sample[self.kspace_key]  # shape (batch, coil, [slice], height, width, complex=2)

        dim = self.spatial_dims["2D"] if kspace.ndim == 5 else self.spatial_dims["3D"]

        backprojected_kspace = self.backward_operator(kspace, dim=dim)

        if kspace.ndim == 6:
            num_slices = backprojected_kspace.shape[2]
            backprojected_kspace = backprojected_kspace.permute(0, 2, 1, 3, 4, 5)
            backprojected_kspace = backprojected_kspace.reshape(
                backprojected_kspace.shape[0] * num_slices, *backprojected_kspace.shape[2:]
            )

        rescaled_backprojected_kspace = T.complex_image_resize(backprojected_kspace, self.shape, self.rescale_mode)

        if kspace.ndim == 6:
            rescaled_backprojected_kspace = rescaled_backprojected_kspace.reshape(
                backprojected_kspace.shape[0] // num_slices, num_slices, *rescaled_backprojected_kspace.shape[1:]
            )
            rescaled_backprojected_kspace = rescaled_backprojected_kspace.permute(0, 2, 1, 3, 4, 5)

            # Compute new k-space from rescaled_backprojected_kspace
        # shape (coil, [slice], new_height, new_width, complex=2)
        sample[self.kspace_key] = self.forward_operator(rescaled_backprojected_kspace, dim=dim)  # The rescaled kspace

        return sample


class PadKspaceModule(DirectModule):
    """Pad k-space with zeros to desired shape module.

    Rescales the k-space by:
    * It first projects the k-space to the image-domain via the backward operator,
    * It pads the back-projected k-space to specified shape,
    * It transforms the rescaled back-projected k-space to the k-space domain via the forward operator.
    """

    def __init__(
        self,
        pad_shape: Union[Tuple[int, int], List[int]],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        kspace_key: KspaceKey = KspaceKey.KSPACE,
    ) -> None:
        """Inits :class:`RescaleKspaceModule`.

        Parameters
        ----------
        pad_shape : tuple or list of ints
            Shape to rescale the input. Must be correspond to (height, width) or (slice, height, width).
        forward_operator : Callable
            The forward operator, e.g. some form of FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.fft2`.
        backward_operator : Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.ifft2`.
        kspace_key : KspaceKey
            K-space key. Default: KspaceKey.KSPACE.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        if len(pad_shape) not in [2, 3]:
            raise ValueError(f"Shape should be a list or tuple of two or three integers. Received: {pad_shape}.")
        self.shape = pad_shape
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`PadKspaceModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample containing key `kspace`.

        Returns
        -------
        Dict[str, Any]
            Cropped and masked sample.
        """
        kspace = sample[self.kspace_key]  # shape (batch, coil, [slice], height, width, complex=2)
        shape = kspace.shape

        sample["original_size"] = shape[-3:-1]

        dim = self.spatial_dims["2D"] if kspace.ndim == 5 else self.spatial_dims["3D"]

        backprojected_kspace = self.backward_operator(kspace, dim=dim)
        backprojected_kspace = T.view_as_complex(backprojected_kspace)

        padded_backprojected_kspace = T.pad_tensor(backprojected_kspace, self.shape)
        padded_backprojected_kspace = T.view_as_real(padded_backprojected_kspace)

        sample[self.kspace_key] = self.forward_operator(padded_backprojected_kspace, dim=dim)  # The rescaled kspace

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
        eps: Union[float, None] = 0.0001,
    ) -> None:
        """Inits :class:`ComputeZeroPadding`.

        Parameters
        ----------
        kspace_key: KspaceKey
            K-space key. Default: KspaceKey.KSPACE.
        padding_key: str
            Target key. Default: "padding".
        eps: float
            Epsilon to multiply sum of signals. If really high, probably no padding will be produced. Default: 0.0001.
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.padding_key = padding_key
        self.eps = eps

    def __call__(self, sample: Dict[str, Any], coil_dim: int = 0) -> Dict[str, Any]:
        """Updates sample with a key `padding_key` with value a binary tensor.

        Non-zero entries indicate samples in kspace with key `kspace_key` which have minor contribution, i.e. padding.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dict sample containing key `kspace_key`.
        coil_dim : int
            Coil dimension. Default: 0.

        Returns
        -------
        sample : Dict[str, Any]
            Dict sample containing key `padding_key`.
        """
        if self.eps is None:
            return sample
        shape = sample[self.kspace_key].shape

        kspace = T.modulus(sample[self.kspace_key].clone()).sum(coil_dim)

        if len(shape) == 5:  # Check if 3D data
            # Assumes that slice dim is 0
            kspace = kspace.sum(0)

        padding = (kspace < (torch.mean(kspace) * self.eps)).to(kspace.device).to(kspace.dtype)

        if len(shape) == 5:
            padding = padding.unsqueeze(0)

        padding = padding.unsqueeze(coil_dim).unsqueeze(-1)
        sample[self.padding_key] = padding

        return sample


class ApplyZeroPadding(DirectTransform):
    """Applies zero padding present in multi-coil kspace input."""

    def __init__(self, kspace_key: KspaceKey = KspaceKey.KSPACE, padding_key: str = "padding") -> None:
        """Inits :class:`ApplyZeroPadding`.

        Parameters
        ----------
        kspace_key: KspaceKey
            K-space key. Default: KspaceKey.KSPACE.
        padding_key: str
            Target key. Default: "padding".
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.padding_key = padding_key

    def __call__(self, sample: Dict[str, Any], coil_dim: int = 0) -> Dict[str, Any]:
        """Applies zero padding on `kspace_key` with value a binary tensor.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dict sample containing key `kspace_key`.
        coil_dim : int
            Coil dimension. Default: 0.

        Returns
        -------
        sample : Dict[str, Any]
            Dict sample containing key `padding_key`.
        """

        sample[self.kspace_key] = T.apply_padding(sample[self.kspace_key], sample[self.padding_key])

        return sample


class ReconstructionType(str, Enum):
    """Reconstruction method for :class:`ComputeImage` transform."""

    rss = "rss"
    complex = "complex"
    complex_mod = "complex_mod"
    sense = "sense"
    sense_mod = "sense_mod"
    ifft = "ifft"


class CompressCoilModule(DirectModule):
    """Compresses k-space coils using SVD."""

    def __init__(self, kspace_key: KspaceKey, num_coils: int) -> None:
        """Inits :class:`CompressCoilModule`.

        Parameters
        ----------
        kspace_key : KspaceKey
            K-space key.
        num_coils : int
            Number of coils to compress.
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.num_coils = num_coils

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Performs coil compression to input k-space.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dict sample containing key `kspace_key`. Assumes coil dimension is first axis.

        Returns
        -------
        sample : Dict[str, Any]
            Dict sample with `kspace_key` compressed to num_coils.
        """
        k_space = sample[self.kspace_key].clone()

        if k_space.shape[1] <= self.num_coils:
            return sample

        k_space = torch.view_as_complex(k_space)

        ndim = k_space.ndim
        if ndim == 5:  # If 3D sample reshape slice into batch dimension as sensitivities are computed 2D
            num_slices = k_space.shape[2]
            k_space = k_space.permute(0, 2, 1, 3, 4)
            k_space = k_space.reshape(k_space.shape[0] * num_slices, *k_space.shape[2:])

        shape = k_space.shape

        # Reshape the k-space data to combine spatial dimensions
        k_space_reshaped = k_space.reshape(shape[0], shape[1], -1)

        # Compute the coil combination matrix using Singular Value Decomposition (SVD)
        U, S, V = torch.linalg.svd(k_space_reshaped, full_matrices=False)

        # Select the top ncoils_new singular vectors from the decomposition
        U_new = U[:, :, : self.num_coils]

        # Perform coil compression
        compressed_k_space = torch.matmul(U_new.transpose(1, 2), k_space_reshaped)

        # Reshape the compressed k-space back to its original shape
        compressed_k_space = compressed_k_space.reshape(shape[0], self.num_coils, *shape[2:])

        if ndim == 5:
            compressed_k_space = compressed_k_space.reshape(
                shape[0] // num_slices, num_slices, self.num_coils, *shape[2:]
            ).permute(0, 2, 1, 3, 4)

        compressed_k_space = torch.view_as_real(compressed_k_space)
        sample[self.kspace_key] = compressed_k_space

        return sample


class ComputeImageModule(DirectModule):
    """Compute Image transform."""

    def __init__(
        self,
        kspace_key: KspaceKey,
        target_key: str,
        backward_operator: Callable,
        type_reconstruction: ReconstructionType = ReconstructionType.rss,
    ) -> None:
        """Inits :class:`ComputeImageModule`.

        Parameters
        ----------
        kspace_key: KspaceKey
            K-space key.
        target_key: str
            Target key.
        backward_operator: callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        type_reconstruction: ReconstructionType
            Type of reconstruction. Can be "complex", "complex_mod", "sense", "sense_mod", "rss" or "ifft".
            Default: ReconstructionType.rss.
        """
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        self.target_key = target_key
        self.type_reconstruction = type_reconstruction

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`ComputeImageModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Contains key kspace_key with value a torch.Tensor of shape (coil,\*spatial_dims, complex=2).

        Returns
        -------
        sample: dict
            Contains key target_key with value a torch.Tensor of shape (\*spatial_dims) if `type_reconstruction` is
            "rss", "complex_mod" or "sense_mod", and of shape(\*spatial_dims, complex_dim=2) otherwise.
        """
        kspace_data = sample[self.kspace_key]
        dim = self.spatial_dims["2D"] if kspace_data.ndim == 5 else self.spatial_dims["3D"]
        # Get complex-valued data solution
        image = self.backward_operator(kspace_data, dim=dim)
        if self.type_reconstruction == ReconstructionType.ifft:
            sample[self.target_key] = image
        elif self.type_reconstruction in [
            ReconstructionType.complex,
            ReconstructionType.complex_mod,
        ]:
            sample[self.target_key] = image.sum(self.coil_dim)
        elif self.type_reconstruction == ReconstructionType.rss:
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
            ReconstructionType.complex_mod,
            ReconstructionType.sense_mod,
        ]:
            sample[self.target_key] = T.modulus(sample[self.target_key], self.complex_dim)
        return sample


class EstimateBodyCoilImage(DirectTransform):
    """Estimates body coil image."""

    def __init__(self, mask_func: Callable, backward_operator: Callable, use_seed: bool = True) -> None:
        """Inits :class:`EstimateBodyCoilImage'.

        Parameters
        ----------
        mask_func: Callable
            A function which creates a sampling mask of the appropriate shape.
        backward_operator: callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        """
        super().__init__()
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.backward_operator = backward_operator

    def __call__(self, sample: Dict[str, Any], coil_dim: int = 0) -> Dict[str, Any]:
        """Calls :class:`EstimateBodyCoilImage`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Contains key kspace_key with value a torch.Tensor of shape (coil, ..., complex=2).
        coil_dim: int
            Coil dimension. Default: 0.

        Returns
        ----------
        sample: Dict[str, Any]
            Contains key `"body_coil_image`.
        """
        kspace = sample["kspace"]

        # We need to create an ACS mask based on the shape of this kspace, as it can be cropped.
        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        kspace_shape = tuple(sample["kspace"].shape[-3:])
        acs_mask = self.mask_func(shape=kspace_shape, seed=seed, return_acs=True)

        kspace = acs_mask * kspace + 0.0
        dim = self.spatial_dims["2D"] if kspace.ndim == 4 else self.spatial_dims["3D"]
        acs_image = self.backward_operator(kspace, dim=dim)

        sample["body_coil_image"] = T.root_sum_of_squares(acs_image, dim=coil_dim)
        return sample


class SensitivityMapType(DirectEnum):
    espirit = "espirit"
    rss_estimate = "rss_estimate"
    unit = "unit"


class EstimateSensitivityMapModule(DirectModule):
    """Data Transformer for training MRI reconstruction models.

    Estimates sensitivity maps given masked k-space data using one of three methods:

    *   Unit: unit sensitivity map in case of single coil acquisition.
    *   RSS-estimate: sensitivity maps estimated by using the root-sum-of-squares of the autocalibration-signal.
    *   ESPIRIT: sensitivity maps estimated with the ESPIRIT method [1]_.

    References
    ----------

    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT--an eigenvalue
        approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
        doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.
    """

    def __init__(
        self,
        kspace_key: KspaceKey = KspaceKey.KSPACE,
        backward_operator: Callable = T.ifft2,
        type_of_map: Optional[SensitivityMapType] = SensitivityMapType.rss_estimate,
        gaussian_sigma: Optional[float] = None,
        espirit_threshold: Optional[float] = 0.05,
        espirit_kernel_size: Optional[int] = 6,
        espirit_crop: Optional[float] = 0.95,
        espirit_max_iters: Optional[int] = 30,
    ) -> None:
        """Inits :class:`EstimateSensitivityMapModule`.

        Parameters
        ----------
        kspace_key: KspaceKey
            K-space key. Default: KspaceKey.KSPACE.
        backward_operator: callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        type_of_map: SensitivityMapType, optional
            Type of map to estimate. Can be "unit", "rss_estimate" or "espirit". Default: "espirit".
            Note that "espirit" is not supported for 3D data.
        gaussian_sigma: float, optional
            If non-zero, acs_image well be calculated
        espirit_threshold: float, optional
            Threshold for the calibration matrix when `type_of_map`=="espirit". Default: 0.05.
        espirit_kernel_size: int, optional
            Kernel size for the calibration matrix when `type_of_map`=="espirit". Default: 6.
        espirit_crop: float, optional
            Output eigenvalue cropping threshold when `type_of_map`=="espirit". Default: 0.95.
        espirit_max_iters: int, optional
            Power method iterations when `type_of_map`=="espirit". Default: 30.
        """
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        if type_of_map not in ["unit", "rss_estimate", "espirit"]:
            raise ValueError(
                f"Expected type of map to be either `unit`, `rss_estimate`, `espirit`. Got {type_of_map}."
            )
        self.type_of_map = type_of_map

        # RSS estimate attributes
        self.gaussian_sigma = gaussian_sigma
        # Espirit attributes
        if type_of_map == "espirit":
            self.espirit_calibrator = EspiritCalibration(
                backward_operator,
                espirit_threshold,
                espirit_kernel_size,
                espirit_crop,
                espirit_max_iters,
                kspace_key,
            )
        self.espirit_threshold = espirit_threshold
        self.espirit_kernel_size = espirit_kernel_size
        self.espirit_crop = espirit_crop
        self.espirit_max_iters = espirit_max_iters

    def estimate_acs_image(self, sample: Dict[str, Any], width_dim: int = -2) -> torch.Tensor:
        """Estimates the autocalibration (ACS) image by sampling the k-space using the ACS mask.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample dictionary,
        width_dim: int
            Dimension corresponding to width. Default: -2.

        Returns
        -------
        acs_image: torch.Tensor
            Estimate of the ACS image.
        """
        kspace_data = sample[self.kspace_key]  # Shape (coil, height, width, complex=2)

        if kspace_data.shape[self.coil_dim] == 1:
            warnings.warn(
                "Estimation of sensitivity map of Single-coil data. This warning will be displayed only once."
            )

        if "sensitivity_map" in sample:
            warnings.warn(
                "`sensitivity_map` is given, but will be overwritten. This warning will be displayed only once."
            )

        if self.gaussian_sigma == 0 or not self.gaussian_sigma:
            kspace_acs = kspace_data * sample["acs_mask"] + 0.0  # + 0.0 removes the sign of zeros.
        else:
            gaussian_mask = torch.linspace(-1, 1, kspace_data.size(width_dim), dtype=kspace_data.dtype)
            gaussian_mask = torch.exp(-((gaussian_mask / self.gaussian_sigma) ** 2))
            gaussian_mask_shape = torch.ones(len(kspace_data.shape)).int()
            gaussian_mask_shape[width_dim] = kspace_data.size(width_dim)
            gaussian_mask = gaussian_mask.reshape(tuple(gaussian_mask_shape))
            kspace_acs = kspace_data * sample["acs_mask"] * gaussian_mask + 0.0

        # Get complex-valued data solution
        # Shape (batch, [slice], coil, height, width, complex=2)
        dim = self.spatial_dims["2D"] if kspace_data.ndim == 5 else self.spatial_dims["3D"]
        acs_image = self.backward_operator(kspace_acs, dim=dim)

        return acs_image

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates sensitivity maps for the input sample.

        Parameters
        ----------
        sample: Dict[str, Any]
            Must contain key matching kspace_key with value a (complex) torch.Tensor
            of shape (coil, height, width, complex=2).

        Returns
        -------
        sample: Dict[str, Any]
            Sample with key "sensitivity_map" with value the estimated sensitivity map.
        """
        if self.type_of_map == "unit":
            kspace = sample[self.kspace_key]
            sensitivity_map = torch.zeros(kspace.shape).float()
            # Assumes complex channel is last
            assert_complex(kspace, complex_last=True)
            sensitivity_map[..., 0] = 1.0
            # Shape (coil, height, width, complex=2)
            sensitivity_map = sensitivity_map.to(kspace.device)

        elif self.type_of_map == "rss_estimate":
            # Shape (batch, coil, height, width, complex=2)
            acs_image = self.estimate_acs_image(sample)
            # Shape (batch, height, width)
            acs_image_rss = T.root_sum_of_squares(acs_image, dim=self.coil_dim)
            # Shape (batch, 1, height, width, 1)
            acs_image_rss = acs_image_rss.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)
            # Shape (batch, coil, height, width, complex=2)
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
        )  # shape (height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)

        sample["sensitivity_map"] = T.safe_divide(sensitivity_map, sensitivity_map_norm)
        return sample


class AddBooleanKeysModule(DirectModule):
    """Adds keys with boolean values to sample."""

    def __init__(self, keys: List[str], values: List[bool]):
        """Inits :class:`AddBooleanKeysModule`.

        Parameters
        ----------
        keys : List[str]
            A list of keys to be added.
        values : List[bool]
            A list of values corresponding to the keys.
        """
        super().__init__()
        self.keys = keys
        self.values = values

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Adds boolean keys to the input sample dictionary.

        Parameters
        ----------
        sample : Dict[str, Any]
            The input sample dictionary.

        Returns
        -------
        Dict[str, Any]
            The modified sample with added boolean keys.
        """
        for key, value in zip(self.keys, self.values):
            sample[key] = value

        return sample


class CopyKeysModule(DirectModule):
    """Copy keys to a new name from the sample if present."""

    def __init__(self, keys: List[str], new_keys: List[str]):
        """Inits :class:`CopyKeysModule`.

        Parameters
        ----------
        keys: List[str]
            Key(s) to copy.
        new_keys: List[str]
            Key(s) to create.
        """
        super().__init__()
        self.keys = keys
        self.new_keys = new_keys

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`CopyKeysModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary to look for keys and copy them with a new name.

        Returns
        -------
        Dict[str, Any]
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


class DeleteKeysModule(DirectModule):
    """Remove keys from the sample if present."""

    def __init__(self, keys: List[str]):
        """Inits :class:`DeleteKeysModule`.

        Parameters
        ----------
        keys: List[str]
            Key(s) to delete.
        """
        super().__init__()
        self.keys = keys

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`DeleteKeysModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary to look for keys and remove them.

        Returns
        -------
        Dict[str, Any]
            Dictionary with deleted specified keys.
        """
        for key in self.keys:
            if key in sample:
                del sample[key]

        return sample


class RenameKeysModule(DirectModule):
    """Rename keys from the sample if present."""

    def __init__(self, old_keys: List[str], new_keys: List[str]):
        """Inits :class:`RenameKeysModule`.

        Parameters
        ----------
        old_keys: List[str]
            Key(s) to rename.
        new_keys: List[str]
            Key(s) to replace old keys.
        """
        super().__init__()
        self.old_keys = old_keys
        self.new_keys = new_keys

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`RenameKeysModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary to look for keys and rename them.

        Returns
        -------
        Dict[str, Any]
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
        pad_coils: Optional[int] = None,
        key: str = "masked_kspace",
        coil_dim: int = 1,
    ):
        """Inits :class:`PadCoilDimensionModule`.

        Parameters
        ----------
        pad_coils: int, optional
            Number of coils to pad to. Default: None.
        key: str
            Key to pad in sample. Default: "masked_kspace".
        coil_dim: int
            Coil dimension along which the pad will be done. Default: 0.
        """
        super().__init__()
        self.num_coils = pad_coils
        self.key = key
        self.coil_dim = coil_dim

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`PadCoilDimensionModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary with key `self.key`.

        Returns
        -------
        sample: Dict[str, Any]
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
                f"Tried to pad to {self.num_coils} coils, but already have {curr_num_coils} for "
                f"{sample['filename']}."
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
        normalize_key: Union[None, TransformKey] = TransformKey.MASKED_KSPACE,
        percentile: Union[None, float] = 0.99,
        scaling_factor_key: TransformKey = TransformKey.SCALING_FACTOR,
    ):
        """Inits :class:`ComputeScalingFactorModule`.

        Parameters
        ----------
        normalize_key : TransformKey or None
            Key name to compute the data for. If the maximum has to be computed on the ACS, ensure the reconstruction
            on the ACS is available (typically `body_coil_image`). Default: "masked_kspace".
        percentile : float or None
            Rescale data with the given percentile. If None, the division is done by the maximum. Default: 0.99.
        scaling_factor_key : TransformKey
            Name of how the scaling factor will be stored. Default: "scaling_factor".
        """
        super().__init__()
        self.normalize_key = normalize_key
        self.percentile = percentile
        self.scaling_factor_key = scaling_factor_key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`ComputeScalingFactorModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample with key `normalize_key` to compute scaling_factor.

        Returns
        -------
        sample: Dict[str, Any]
            Sample with key `scaling_factor_key`.
        """
        if self.normalize_key == "scaling_factor":  # This is a real-valued given number
            scaling_factor = sample["scaling_factor"]
        elif not self.normalize_key:
            kspace = sample["masked_kspace"]
            scaling_factor = torch.tensor([1.0] * kspace.size(0), device=kspace.device, dtype=kspace.dtype)
        else:
            data = sample[self.normalize_key]
            scaling_factor: Union[List, torch.Tensor] = []
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
        keys_to_normalize: Optional[List[TransformKey]] = None,
    ):
        """Inits :class:`NormalizeModule`.

        Parameters
        ----------
        scaling_factor_key : TransformKey
            Name of scaling factor key expected in sample. Default: 'scaling_factor'.
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
            ]
            if keys_to_normalize is None
            else keys_to_normalize
        )

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`NormalizeModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample to normalize.

        Returns
        -------
        sample: Dict[str, Any]
            Sample with normalized values if their respective key is in `keys_to_normalize` and key
            `scaling_factor_key` exists in sample.
        """
        scaling_factor = sample.get(self.scaling_factor_key, None)
        # Normalize data
        if scaling_factor is not None:
            for key in sample.keys():
                if key not in self.keys_to_normalize:
                    continue
                sample[key] = T.safe_divide(
                    sample[key],
                    scaling_factor.reshape(-1, *[1 for _ in range(sample[key].ndim - 1)]),
                )

            sample["scaling_diff"] = 0.0
        return sample


class WhitenDataModule(DirectModule):
    """Whitens complex data Module."""

    def __init__(self, epsilon: float = 1e-10, key: str = "complex_image"):
        """Inits :class:`WhitenDataModule`.

        Parameters
        ----------
        epsilon: float
            Default: 1e-10.
        key: str
            Key to whiten. Default: "complex_image".
        """
        super().__init__()
        self.epsilon = epsilon
        self.key = key

    def complex_whiten(self, complex_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whiten complex image.

        Parameters
        ----------
        complex_image: torch.Tensor
            Complex image tensor to whiten.

        Returns
        -------
        mean, std, whitened_image: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # From: https://github.com/facebookresearch/fastMRI
        #       blob/da1528585061dfbe2e91ebbe99a5d4841a5c3f43/banding_removal/fastmri/data/transforms.py#L464  # noqa
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

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`WhitenDataModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample with key `key`.

        Returns
        -------
        sample: Dict[str, Any]
            Sample with value of `key` whitened.
        """
        _, _, whitened_image = self.complex_whiten(sample[self.key])
        sample[self.key] = whitened_image
        return sample


class ModuleWrapper:
    class SubWrapper:
        def __init__(self, transform, toggle_dims):
            self.toggle_dims = toggle_dims
            self._transform = transform

        def __call__(self, sample):
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

        def __repr__(self):
            return self._transform.__repr__()

    def __init__(self, module: Callable, toggle_dims: bool):
        self._module = module
        self.toggle_dims = toggle_dims

    def __call__(self, *args, **kwargs):
        return self.SubWrapper(self._module(*args, **kwargs), toggle_dims=self.toggle_dims)


ApplyMask = ModuleWrapper(ApplyMaskModule, toggle_dims=False)
ComputeImage = ModuleWrapper(ComputeImageModule, toggle_dims=True)
EstimateSensitivityMap = ModuleWrapper(EstimateSensitivityMapModule, toggle_dims=True)
RescaleKspace = ModuleWrapper(RescaleKspaceModule, toggle_dims=True)
PadKspace = ModuleWrapper(PadKspaceModule, toggle_dims=False)
CopyKeys = ModuleWrapper(CopyKeysModule, toggle_dims=False)
DeleteKeys = ModuleWrapper(DeleteKeysModule, toggle_dims=False)
RenameKeys = ModuleWrapper(RenameKeysModule, toggle_dims=False)
CompressCoil = ModuleWrapper(CompressCoilModule, toggle_dims=True)
PadCoilDimension = ModuleWrapper(PadCoilDimensionModule, toggle_dims=True)
ComputeScalingFactor = ModuleWrapper(ComputeScalingFactorModule, toggle_dims=True)
Normalize = ModuleWrapper(NormalizeModule, toggle_dims=False)
WhitenData = ModuleWrapper(WhitenDataModule, toggle_dims=False)


class ToTensor(DirectTransform):
    """Transforms all np.array-like values in sample to torch.tensors."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`ToTensor`.

        Parameters
        ----------
        sample: Dict[str, Any]
             Contains key 'kspace' with value a np.array of shape (coil, height, width) (2D)
             or (coil, slice, height, width) (3D)

        Returns
        -------
        sample: Dict[str, Any]
             Contains key 'kspace' with value a torch.Tensor of shape (coil, height, width) (2D)
             or (coil, slice, height, width) (3D)
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
    mask_func: Optional[Callable],
    crop: Optional[Union[Tuple[int, int], str]] = None,
    crop_type: Optional[str] = "uniform",
    rescale: Optional[tuple[int, int]] = None,
    rescale_mode: RescaleMode = RescaleMode.BILINEAR,
    image_center_crop: bool = True,
    random_rotation: bool = False,
    random_rotation_degrees: Optional[Sequence[int]] = (-90, 90),
    random_rotation_probability: Optional[float] = 0.5,
    random_flip: bool = False,
    random_flip_type: Optional[RandomFlipType] = RandomFlipType.RANDOM,
    random_flip_probability: Optional[float] = 0.5,
    compute_and_apply_padding: bool = True,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    use_seed: bool = True,
    compress_coils: Optional[int] = None,
    pad_coils: Optional[int] = None,
    use_acs_as_mask: bool = False,
) -> object:
    """Build transforms for MRI.

    - Converts input to (complex-valued) tensor.
    - Adds a sampling mask if `mask_func` is defined.
    - Adds coil sensitivities and / or the body coil_image
    - Crops the input data if needed and masks the fully sampled k-space.
    - Add a target.
    - Normalize input data.
    - Pads the coil dimension.

    Parameters
    ----------
    forward_operator : Callable
        The forward operator, e.g. some form of FFT (centered or uncentered).
    backward_operator : Callable
        The backward operator, e.g. some form of inverse FFT (centered or uncentered).
    mask_func : Callable or None
        A function which creates a sampling mask of the appropriate shape.
    crop : Tuple[int, int] or str, Optional
        If not None, this will transform the "kspace" to an image domain, crop it, and transform it back.
        If a tuple of integers is given then it will crop the backprojected kspace to that size. If
        "reconstruction_size" is given, then it will crop the backprojected kspace according to it, but
        a key "reconstruction_size" must be present in the sample. Default: None.
    crop_type : Optional[str]
        Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default: "uniform".
    rescale : tuple of ints
        If not None, this will transform the "kspace" to an image domain, scale it, and transform it back.
        Must correspond to (height, width). Default: None.
    rescale_mode : RescaleMode
        Mode to be used for rescaling. If `rescale` is None, this will be ignored.
        Can be RescaleMode.BILINEAR or RescaleMode.BICUBIC. Default: RescaleMode.BILINEAR.
    image_center_crop : bool
        If True the backprojected kspace will be cropped around the center, otherwise randomly.
        This will be ignored if `crop` is None. Default: True.
    random_rotation : bool
        If True, random rotations will be applied of `random_rotation_degrees` degrees, with probability
        `random_rotation_probability`. Default: False.
    random_rotation_degrees : Sequence[int], optional
        Default: (-90, 90).
    random_rotation_probability : float, optional
        Default: 0.5.
    random_flip : bool
        If True, random rotation of `random_flip_type` type, with probability `random_flip_probability`. Default: False.
    random_flip_type : RandomFlipType, optional
        Default: RandomFlipType.RANDOM.
    random_flip_probability : float, optional
        Default: 0.5.
    compute_and_apply_padding : bool
        Default: True.
    padding_eps: float
        Padding epsilon. Default: 0.0001.
    estimate_body_coil_image : bool
        Estimate body coil image. Default: False.
    use_seed : bool
        If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
        the same mask every time. Default: True.
    compress_coils : int, optional
        Number of coils to compress input k-space. Default: None.
    pad_coils : int, optional
        Number of coils to pad data to. Default: None.
    use_acs_as_mask : bool
        If True, this will replace `sampling_mask` with `acs_mask`. Might be useful for downstream tasks
        such as adaptive sampling. Default: False.

    Returns
    -------
    object: Callable
        An MRI transformation object.
    """
    # pylint: disable=too-many-locals
    mri_transforms: List[Callable] = [ToTensor()]
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
                kspace_key=KspaceKey.KSPACE,
            )
        ]
    if random_rotation:
        mri_transforms += [
            RandomRotation(
                degrees=random_rotation_degrees,
                p=random_rotation_probability,
                keys_to_rotate=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if random_flip:
        mri_transforms += [
            RandomFlip(
                flip=random_flip_type,
                p=random_flip_probability,
                keys_to_flip=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if compute_and_apply_padding:
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
    if use_acs_as_mask:
        mri_transforms += [CopyKeys(["acs_mask"], ["sampling_mask"])]
    if compute_and_apply_padding:
        mri_transforms += [ApplyZeroPadding("sampling_mask", "padding")]
    if compress_coils:
        mri_transforms += [CompressCoil(kspace_key=KspaceKey.KSPACE)]
    if pad_coils:
        mri_transforms += [PadCoilDimension(pad_coils=pad_coils, key=KspaceKey.KSPACE)]
    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))

    return Compose(mri_transforms)


def build_post_mri_transforms(
    backward_operator: Callable,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.rss_estimate,
    sensitivity_maps_gaussian: Optional[float] = None,
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05,
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6,
    sensitivity_maps_espirit_crop: Optional[float] = 0.95,
    sensitivity_maps_espirit_max_iters: Optional[int] = 30,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.rss,
    scaling_key: TransformKey = TransformKey.MASKED_KSPACE,
    scale_percentile: Optional[float] = 0.99,
) -> object:
    """Build transforms for MRI.

    * Converts input to (complex-valued) tensor.
    * Adds a sampling mask if `mask_func` is defined.
    * Adds coil sensitivities and / or the body coil_image
    * Crops the input data if needed and masks the fully sampled k-space.
    * Add a target.
    * Normalize input data.
    * Pads the coil dimension.

    Parameters
    ----------
    backward_operator : Callable
        The backward operator, e.g. some form of inverse FFT (centered or uncentered).
    estimate_sensitivity_maps : bool
        Estimate sensitivity maps using the acs region. Default: True.
    sensitivity_maps_type: sensitivity_maps_type
        Can be SensitivityMapType.rss_estimate, SensitivityMapType.unit or SensitivityMapType.espirit.
        Will be ignored if `estimate_sensitivity_maps` is equal to False. Default: SensitivityMapType.rss_estimate.
    sensitivity_maps_gaussian : float
        Optional sigma for gaussian weighting of sensitivity map.
    sensitivity_maps_espirit_threshold: float, optional
            Threshold for the calibration matrix when `type_of_map` is equal to "espirit". Default: 0.05.
    sensitivity_maps_espirit_kernel_size: int, optional
        Kernel size for the calibration matrix when `type_of_map` is equal to "espirit". Default: 6.
    sensitivity_maps_espirit_crop: float, optional
        Output eigenvalue cropping threshold when `type_of_map` is equal to "espirit". Default: 0.95.
    sensitivity_maps_espirit_max_iters: int, optional
        Power method iterations when `type_of_map` is equal to "espirit". Default: 30.
    delete_acs_mask : bool
        If True will delete key `acs_mask`. Default: True.
    delete_kspace : bool
        If True will delete key `kspace` (fully sampled k-space). Default: True.
    image_recon_type : ReconstructionType
        Type to reconstruct target image. Default: ReconstructionType.rss.
    scaling_key : TransformKey
        Key in sample to scale scalable items in sample. Default: TransformKey.MASKED_KSPACE.
    scale_percentile : float, optional
        Data will be rescaled with the given percentile. If None, the division is done by the maximum. Default: 0.99
        the same mask every time. Default: True.

    Returns
    -------
    object: Callable
        An MRI transformation object.
    """
    mri_transforms: List[Callable] = []

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


def build_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Optional[Callable],
    crop: Optional[Union[Tuple[int, int], str]] = None,
    crop_type: Optional[str] = "uniform",
    rescale: Optional[tuple[int, int]] = None,
    rescale_mode: RescaleMode = RescaleMode.BILINEAR,
    pad: Optional[Union[Tuple[int, int], str]] = None,
    image_center_crop: bool = True,
    random_rotation: bool = False,
    random_rotation_degrees: Optional[Sequence[int]] = (-90, 90),
    random_rotation_probability: Optional[float] = 0.5,
    random_flip: bool = False,
    random_flip_type: Optional[RandomFlipType] = RandomFlipType.RANDOM,
    random_flip_probability: Optional[float] = 0.5,
    random_reverse: bool = False,
    random_reverse_probability: float = 0.5,
    compute_and_apply_padding: bool = True,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.rss_estimate,
    sensitivity_maps_gaussian: Optional[float] = None,
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05,
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6,
    sensitivity_maps_espirit_crop: Optional[float] = 0.95,
    sensitivity_maps_espirit_max_iters: Optional[int] = 30,
    use_acs_as_mask: bool = False,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.rss,
    compress_coils: Optional[int] = None,
    pad_coils: Optional[int] = None,
    scaling_key: TransformKey = TransformKey.MASKED_KSPACE,
    scale_percentile: Optional[float] = 0.99,
    use_seed: bool = True,
) -> object:
    """Build transforms for MRI.

    * Converts input to (complex-valued) tensor.
    * Adds a sampling mask if `mask_func` is defined.
    * Adds coil sensitivities and / or the body coil_image
    * Crops the input data if needed and masks the fully sampled k-space.
    * Add a target.
    * Normalize input data.
    * Pads the coil dimension.

    Parameters
    ----------
    forward_operator : Callable
        The forward operator, e.g. some form of FFT (centered or uncentered).
    backward_operator : Callable
        The backward operator, e.g. some form of inverse FFT (centered or uncentered).
    mask_func : Callable or None
        A function which creates a sampling mask of the appropriate shape.
    crop : Tuple[int, int] or str, Optional
        If not None, this will transform the "kspace" to an image domain, crop it, and transform it back.
        If a tuple of integers is given then it will crop the backprojected kspace to that size. If
        "reconstruction_size" is given, then it will crop the backprojected kspace according to it, but
        a key "reconstruction_size" must be present in the sample. Default: None.
    crop_type : Optional[str]
        Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default: "uniform".
    rescale : tuple of ints
        If not None, this will transform the "kspace" to the image domain, scale it, and transform it back.
        Must correspond to (height, width). Default: None.
    rescale_mode : RescaleMode
        Mode to be used for rescaling. If `rescale` is None, this will be ignored.
        Can be RescaleMode.BILINEAR or RescaleMode.BICUBIC. Default: RescaleMode.BILINEAR.
    pad : tuple of ints
        If not None, this will transform the "kspace" to the image domain, pad it if necessary, and transform it back.
        Must correspond to (height, width). Default: None.
    image_center_crop : bool
        If True the backprojected kspace will be cropped around the center, otherwise randomly.
        This will be ignored if `crop` is None. Default: True.
    random_rotation : bool
        If True, random rotations will be applied of `random_rotation_degrees` degrees, with probability
        `random_rotation_probability`. Default: False.
    random_rotation_degrees : Sequence[int], optional
        Default: (-90, 90).
    random_rotation_probability : float, optional
        Default: 0.5.
    random_flip : bool
        If True, random rotation of `random_flip_type` type, with probability `random_flip_probability`. Default: False.
    random_flip_type : RandomFlipType, optional
        Default: RandomFlipType.RANDOM.
    random_flip_probability : float, optional
        Default: 0.5.
    random_reverse : bool
        If True will perform random reversion along the time or slice dimension (2). Default: False.
    random_reverse_probability : float
        Default: 0.5.
    compute_and_apply_padding : bool
        Default: True.
    padding_eps: float
        Padding epsilon. Default: 0.0001.
    estimate_body_coil_image : bool
        Estimate body coil image. Default: False.
    estimate_sensitivity_maps : bool
        Estimate sensitivity maps using the acs region. Default: True.
    sensitivity_maps_type: sensitivity_maps_type
        Can be SensitivityMapType.rss_estimate, SensitivityMapType.unit or SensitivityMapType.espirit.
        Will be ignored if `estimate_sensitivity_maps` is False. Default: SensitivityMapType.rss_estimate.
    sensitivity_maps_gaussian : float
        Optional sigma for gaussian weighting of sensitivity map.
    sensitivity_maps_espirit_threshold : float, optional
            Threshold for the calibration matrix when `type_of_map` is equal to "espirit". Default: 0.05.
    sensitivity_maps_espirit_kernel_size : int, optional
        Kernel size for the calibration matrix when `type_of_map` is equal to "espirit". Default: 6.
    sensitivity_maps_espirit_crop : float, optional
        Output eigenvalue cropping threshold when `type_of_map` is equal to "espirit". Default: 0.95.
    sensitivity_maps_espirit_max_iters : int, optional
        Power method iterations when `type_of_map` is equal to "espirit". Default: 30.
    use_acs_as_mask : bool
        If True, this will replace `sampling_mask` with `acs_mask`. Might be useful for downstream tasks
        such as adaptive sampling. Default: False.
    delete_acs_mask : bool
        If True will delete key `acs_mask`. Default: True.
    delete_kspace : bool
        If True will delete key `kspace` (fully sampled k-space). Default: True.
    image_recon_type : ReconstructionType
        Type to reconstruct target image. Default: ReconstructionType.rss.
    compress_coils : int, optional
        Number of coils to compress input k-space. Default: None.
    pad_coils : int, optional
        Number of coils to pad data to. Default: None.
    scaling_key : TransformKey
        Key in sample to scale scalable items in sample. Default: TransformKey.MASKED_KSPACE.
    scale_percentile : float, optional
        Data will be rescaled with the given percentile. If None, the division is done by the maximum. Default: 0.99
    use_seed : bool
        If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
        the same mask every time. Default: True.

    Returns
    -------
    object: Callable
        An MRI transformation object.
    """
    # TODO: Use seed

    mri_transforms: List[Callable] = [ToTensor()]
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
                kspace_key=KspaceKey.KSPACE,
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
    if random_rotation:
        mri_transforms += [
            RandomRotation(
                degrees=random_rotation_degrees,
                p=random_rotation_probability,
                keys_to_rotate=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if random_flip:
        mri_transforms += [
            RandomFlip(
                flip=random_flip_type,
                p=random_flip_probability,
                keys_to_flip=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if random_reverse:
        mri_transforms += [
            RandomReverse(
                p=random_reverse_probability,
                keys_to_reverse=(TransformKey.KSPACE, TransformKey.SENSITIVITY_MAP),
            )
        ]
    if compute_and_apply_padding:
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
            ),
        ]
    if compute_and_apply_padding:
        mri_transforms += [ApplyZeroPadding("sampling_mask", "padding")]
    if use_acs_as_mask:  # Might be needed for adaptive sampling, as sampling happens downstream
        mri_transforms += [CopyKeys(["acs_mask"], ["sampling_mask"])]

    if compress_coils:
        mri_transforms += [CompressCoil(num_coils=compress_coils, kspace_key=KspaceKey.KSPACE)]
    if pad_coils:
        mri_transforms += [PadCoilDimension(pad_coils=pad_coils, key=KspaceKey.KSPACE)]

    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))

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

    if delete_acs_mask:
        mri_transforms += [DeleteKeys(keys=["acs_mask"])]

    mri_transforms += [
        ApplyMask(
            sampling_mask_key="sampling_mask",
            input_kspace_key=KspaceKey.KSPACE,
            target_kspace_key=KspaceKey.MASKED_KSPACE,
        ),
    ]

    mri_transforms += [
        ComputeScalingFactor(
            normalize_key=scaling_key, percentile=scale_percentile, scaling_factor_key=TransformKey.SCALING_FACTOR
        ),
        Normalize(scaling_factor_key=TransformKey.SCALING_FACTOR),
    ]

    mri_transforms += [
        ComputeImage(
            kspace_key=KspaceKey.KSPACE,
            target_key=TransformKey.TARGET,
            backward_operator=backward_operator,
            type_reconstruction=image_recon_type,
        )
    ]

    if delete_kspace:
        mri_transforms += [DeleteKeys(keys=[KspaceKey.KSPACE])]

    return Compose(mri_transforms)
