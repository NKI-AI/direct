# Copyright (c) DIRECT Contributors

"""DIRECT samplers module. 

This module contains classes for creating sub-sampling masks. The masks are used to sample k-space data in MRI
reconstruction. The masks are created by selecting a subset of samples from the input k-space data."""

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

from __future__ import annotations

import contextlib
import inspect
import logging
from abc import abstractmethod
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
from scipy.linalg import toeplitz
from scipy.ndimage import rotate

import direct.data.transforms as T
from direct.common._gaussian import gaussian_mask_1d, gaussian_mask_2d  # pylint: disable=no-name-in-module
from direct.common._poisson import poisson as _poisson  # pylint: disable=no-name-in-module
from direct.environment import DIRECT_CACHE_DIR
from direct.types import DirectEnum, MaskFuncMode, Number, TensorOrNdarray
from direct.utils import str_to_class
from direct.utils.io import download_url

# pylint: disable=arguments-differ


__all__ = [
    "BaseMaskFunc",
    "CIRCUSMaskFunc",
    "CIRCUSSamplingMode",
    "CalgaryCampinasMaskFunc",
    "CartesianEquispacedMaskFunc",
    "CartesianMagicMaskFunc",
    "CartesianRandomMaskFunc",
    "CartesianVerticalMaskFunc",
    "EquispacedMaskFunc",
    "FastMRIEquispacedMaskFunc",
    "FastMRIMagicMaskFunc",
    "FastMRIRandomMaskFunc",
    "Gaussian1DMaskFunc",
    "Gaussian2DMaskFunc",
    "KtBaseMaskFunc",
    "KtGaussian1DMaskFunc",
    "KtRadialMaskFunc",
    "KtUniformMaskFunc",
    "MagicMaskFunc",
    "MaskFuncMode",
    "RadialMaskFunc",
    "RandomMaskFunc",
    "SpiralMaskFunc",
    "VariableDensityPoissonMaskFunc",
    "build_masking_function",
    "centered_disk_mask",
    "integerize_seed",
]

logger = logging.getLogger(__name__)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class BaseMaskFunc:
    """:class:`BaseMaskFunc` is the base class to create a sub-sampling mask of a given shape.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.
    center_fractions : Union[list[float], tuple[float, ...]], optional
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        If multiple values are provided, then one of these numbers is chosen uniformly each time. If uniform_range
        is True, then two values should be given. Default: None.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Optional[Union[list[float], tuple[float, ...]]] = None,
        uniform_range: bool = True,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`BaseMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions : Union[list[float], tuple[float, ...]], optional
            Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
            If multiple values are provided, then one of these numbers is chosen uniformly each time. If uniform_range
            is True, then two values should be given. Default: None.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if center_fractions is not None:
            if len([center_fractions]) != len([accelerations]):
                raise ValueError(
                    f"Number of center fractions should match number of accelerations. "
                    f"Got {len([center_fractions])} {len([accelerations])}."
                )

        self.center_fractions = center_fractions
        self.accelerations = accelerations

        self.uniform_range = uniform_range
        self.mode = mode

        self.rng = np.random.RandomState()

    def choose_acceleration(self) -> Union[Number, tuple[Number, Number]]:
        """Chooses an acceleration and center fraction.

        Returns
        -------
        Union[Number, tuple[Number, Number]]
            Acceleration and center fraction.

        Raises
        ------
        NotImplementedError
            If uniform range is not yet implemented.
        """
        if not self.accelerations:
            return None

        if not self.uniform_range:
            choice = self.rng.randint(0, len(self.accelerations))
            acceleration = self.accelerations[choice]
            if self.center_fractions is None:
                return acceleration

            center_fraction = self.center_fractions[choice]
            return center_fraction, acceleration
        raise NotImplementedError("Uniform range is not yet implemented.")

    @abstractmethod
    def mask_func(self, *args, **kwargs) -> torch.Tensor:
        """Abstract methot to create a sub-sampling mask.

        Returns
        -------
        torch.Tensor
            Sampling mask.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("This method should be implemented by a child class.")

    def _reshape_and_add_coil_axis(self, mask: TensorOrNdarray, shape: tuple[int, ...]) -> torch.Tensor:
        """Reshape the mask with ones to match shape and add a coil axis.

        Parameters
        ----------
        mask : np.ndarray or torch.Tensor
            Input mask of shape (num_rows, num_cols) if mode is MaskFuncMode.STATIC, and
            (nt or num_slices, num_rows, num_cols) if mode is MaskFuncMode.DYNAMIC or
            MaskFuncMode.MULTISLICE to be reshaped.
        shape : tuple of ints
            Shape of the output array after reshaping. Expects shape to be (\*, num_rows, num_cols, channels) for
            mode MaskFuncMode.STATIC, and (\*, nt or num_slices, num_rows, num_cols, channels) for mode
            MaskFuncMode.DYNAMIC where \* is any number of dimensions.

        Returns
        -------
        toch.Tensor
            Reshaped mask tensor with ones with an added coil axis.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]

        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask_shape[-3] = num_rows

        # If mode is dynamic or multislice dim should not be 1
        if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
            mask_shape[-4] = shape[-4]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Reshape the mask to match the shape and make boolean
        mask = mask.reshape(*mask_shape).bool()
        # Add coil axis
        mask = mask[None, ...]

        return mask

    def __call__(self, shape: tuple[int, ...], *args, **kwargs) -> torch.Tensor:
        """Calls the mask function.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the mask to be created. Needs to be at least 3 dimensions. If mode is MaskFuncMode.DYNAMIC,
            or MaskFuncMode.MULTISLICE, then the shape should have at least 4 dimensions.
        args : tuple
            Additional arguments to be passed to the mask function.
        kwargs : dict
            Additional keyword arguments to be passed to the mask function.

        Returns
        -------
        torch.Tensor
            Sampling mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions.")
        if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] and len(shape) < 4:
            raise ValueError("Shape should have 4 or more dimensions for dynamic or multislice mode.")

        mask = self.mask_func(shape, *args, **kwargs)
        return mask


class CartesianVerticalMaskFunc(BaseMaskFunc):
    """:class:`CartesianVerticalMaskFunc` is the base class to create vertical Cartesian sub-sampling masks.

    This is the base class of Cartesian and FastMRI mask functions. The difference between Cartesian and FastMRI
    mask functions is that the former uses the number of center lines to be retained, while the latter uses the
    fraction of center lines to be retained.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`CartesianVerticalMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )

    @staticmethod
    def center_mask_func(num_cols: int, num_low_freqs: int) -> np.ndarray:
        """Creates the ACS (center) mask.

        Parameters
        ----------
        num_cols : int
            Number of center columns/lines.
        num_low_freqs : int
            Number of low-frequency columns/lines.

        Returns
        -------
        np.ndarray
           ACS (center) mask.
        """
        # create the mask
        mask = np.zeros(num_cols, dtype=bool)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        return mask

    @staticmethod
    def _broadcast_mask(mask: np.ndarray, num_rows: int) -> np.ndarray:
        """Broadcast the input mask array to match a specified number of rows.

        Useful for line masks that need to be broadcasted to match the number of rows in the k-space.

        Parameters
        ----------
        mask : np.ndarray
            Input mask array to be broadcasted.
        num_rows : int
            Number of rows to which the mask array will be broadcasted.

        Returns
        -------
        np.ndarray
            Broadcasted mask array.

        Raises
        ------
        ValueError
            If the input mask array has an unsupported number of dimensions.
        """
        if mask.ndim == 1:
            broadcast_mask = np.tile(mask, (num_rows, 1))
        elif mask.ndim == 2:
            broadcast_mask = np.tile(mask[:, np.newaxis, :], (1, num_rows, 1))
        else:
            raise ValueError(
                f"Mask should have 1 dimension for mode STATIC "
                f"and 2 dimensions for mode DYNAMIC or MULTISLICE. Got mask of shape {mask.shape}."
            )

        return broadcast_mask


class RandomMaskFunc(CartesianVerticalMaskFunc):
    r"""Random vertical line mask function.

    The mask selects a subset of columns from the input k-space data. If the k-space data has :math:`N` columns,
    the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})` columns in the center corresponding
            to low-frequencies if center_fraction < 1.0, or :math:`N_{\text{low freqs}} = \text{center_fraction}`
            if center_fraction >= 1 and is integer.
        #.  The other columns are selected uniformly at random with a probability equal to:
            :math:`\text{prob} = (N / \text{acceleration} - N_{\text{low freqs}}) / (N - N_{\text{low freqs}})`.
            This ensures that the expected number of columns selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[Number], tuple[Number, ...]]
        If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
        If >= 1 (integer) this corresponds to the exact number of low-frequency columns to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[Number], tuple[Number, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`RandomMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[Number], tuple[Number, ...]]
            If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
            If >= 1 (integer) this corresponds to the exact number of low-frequency columns to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates vertical line mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.


        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]
        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):

            center_fraction, acceleration = self.choose_acceleration()

            if center_fraction < 1.0:
                num_low_freqs = int(round(num_cols * center_fraction))
            else:
                num_low_freqs = int(center_fraction)

            mask = self.center_mask_func(num_cols, num_low_freqs)

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                mask = mask[np.newaxis].repeat(num_slc_or_time, axis=0)

            if return_acs:
                return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)

            # Create the mask
            mask = mask.reshape(num_slc_or_time, -1)  # In case mode != MaskFuncMode.STATIC:
            for i in range(num_slc_or_time):
                prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
                mask[i] = mask[i] | (self.rng.uniform(size=num_cols) < prob)

        return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)


class FastMRIRandomMaskFunc(RandomMaskFunc):
    r"""Random vertical line mask function.

    The mask selects a subset of columns from the input k-space data. If the k-space data has :math:`N` columns,
    the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})`  columns in the center corresponding
            to low-frequencies.
        #.  The other columns are selected uniformly at random with a probability equal to:
            :math:`\text{prob} = (N / \text{acceleration} - N_{\text{low freqs}}) / (N - N_{\text{low freqs}})`.
            This ensures that the expected number of columns selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`FastMRIRandomMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if not all(0 < center_fraction < 1 for center_fraction in center_fractions):
            raise ValueError(
                f"Center fraction values should be between 0 and 1. Received {center_fractions}. "
                f"For exact number of center lines, use `CartesianMagicMaskFunc`."
            )
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )


class CartesianRandomMaskFunc(RandomMaskFunc):
    r"""Cartesian random vertical line mask function.

    Similar to :class:`FastMRIRandomMaskFunc`, but instead of center fraction (`center_fractions`) representing
    the fraction of center lines to the original size, here, it represents the actual number of center lines.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.
    center_fractions : Union[list[int], tuple[int, ...]]
        Number of low-frequence (center) columns to be retained.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[int], tuple[int, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`CartesianRandomMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions : Union[list[int], tuple[int, ...]]
            Number of low-frequence (center) columns to be retained.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if not all((1 < center_fraction) and isinstance(center_fraction, int) for center_fraction in center_fractions):
            raise ValueError(
                f"Center fraction values should be integers greater then or equal to 1 corresponding to the number of "
                f"center lines. Received {center_fractions}. For fractions, use `FastMRIMagicMaskFunc`."
            )
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )


class EquispacedMaskFunc(CartesianVerticalMaskFunc):
    r"""Equispaced vertical line mask function.

    :class:`EquispacedMaskFunc` creates a sub-sampling mask of given shape. The mask selects a subset of columns
    from the input k-space data. If the k-space data has N columns, the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})` columns in the center corresponding
            to low-frequencies if center_fraction < 1.0, or :math:`N_{\text{low freqs}} = \text{center_fraction}`
            if center_fraction >= 1 and is integer.
        #.  The other columns are selected with equal spacing at a proportion that reaches the desired acceleration
            rate taking into consideration the number of low frequencies. This ensures that the expected number of
            columns selected is equal to :math:`\frac{N}{\text{acceleration}}`.

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require modifications to standard GRAPPA
    approaches. Nonetheless, this aspect of the function has been preserved to match the public multicoil data.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[Number], tuple[Number, ...]]
        If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
        If >= 1 (integer) this corresponds to the exact number of low-frequency columns to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[Number], tuple[Number, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`EquispacedMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[Number], tuple[Number, ...]]
            If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
            If >= 1 (integer) this corresponds to the exact number of low-frequency columns to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates an vertical equispaced vertical line mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]
        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):

            center_fraction, acceleration = self.choose_acceleration()

            if center_fraction < 1.0:
                num_low_freqs = int(round(num_cols * center_fraction))
            else:
                num_low_freqs = int(center_fraction)

            mask = self.center_mask_func(num_cols, num_low_freqs)

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                mask = mask[np.newaxis].repeat(num_slc_or_time, axis=0)

            if return_acs:
                return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)

            mask = mask.reshape(num_slc_or_time, -1)  # In case mode != MaskFuncMode.STATIC:
            for i in range(num_slc_or_time):
                offset = self.rng.randint(0, round(adjusted_accel))
                accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
                accel_samples = np.around(accel_samples).astype(np.uint)
                mask[i, accel_samples] = True

        return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)


class FastMRIEquispacedMaskFunc(EquispacedMaskFunc):
    r"""Equispaced vertical line mask function.

    :class:`FastMRIEquispacedMaskFunc` creates a sub-sampling mask of given shape. The mask selects a subset of columns
    from the input k-space data. If the k-space data has N columns, the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})` columns in the center corresponding
            to low-frequencies.
        #.  The other columns are selected with equal spacing at a proportion that reaches the desired acceleration
            rate taking into consideration the number of low frequencies. This ensures that the expected number of
            columns selected is equal to :math:`\frac{N}{\text{acceleration}}`.

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require modifications to standard GRAPPA
    approaches. Nonetheless, this aspect of the function has been preserved to match the public multicoil data.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction (< 1.0) of low-frequency columns to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`FastMRIEquispacedMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction (< 1.0) of low-frequency columns to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if not all(0 < center_fraction < 1 for center_fraction in center_fractions):
            raise ValueError(
                f"Center fraction values should be between 0 and 1. Received {center_fractions}. "
                f"For exact number of center lines, use `CartesianMagicMaskFunc`."
            )
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )


class CartesianEquispacedMaskFunc(EquispacedMaskFunc):
    r"""Cartesian equispaced vertical line mask function.

    Similar to :class:`FastMRIEquispacedMaskFunc`, but instead of center fraction (`center_fractions`) representing
    the fraction of center lines to the original size, here, it represents the actual number of center lines.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.
    center_fractions : Union[list[int], tuple[int, ...]]
        Number of low-frequence (center) columns to be retained.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[int], tuple[int, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`CartesianEquispacedMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions : Union[list[int], tuple[int, ...]]
            Number of low-frequence (center) columns to be retained.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if not all((1 < center_fraction) and isinstance(center_fraction, int) for center_fraction in center_fractions):
            raise ValueError(
                f"Center fraction values should be integers greater then or equal to 1 corresponding to the number of "
                f"center lines. Received {center_fractions}. For fractions, use `FastMRIMagicMaskFunc`."
            )
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )


class MagicMaskFunc(CartesianVerticalMaskFunc):
    """Vertical line mask function as implemented in [1]_.

    :class:`MagicMaskFunc` exploits the conjugate symmetry via offset-sampling. It is essentially an
    equispaced mask with an offset for the opposite site of the k-space. Since MRI images often exhibit approximate
    conjugate k-space symmetry, this mask is generally more efficient than :class:`EquispacedMaskFunc`.

    References
    ----------
    .. [1] Defazio, Aaron. “Offset Sampling Improves Deep Learning Based Accelerated MRI Reconstructions by
        Exploiting Symmetry.” ArXiv:1912.01101 [Cs, Eess], Feb. 2020. arXiv.org, http://arxiv.org/abs/1912.01101.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.
    center_fractions : Union[list[Number], tuple[Number, ...]]
        If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
        If >= 1 (integer) this corresponds to the exact number of low-frequency columns to be retained.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[Number], tuple[Number, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`MagicMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions : Union[list[Number], tuple[Number, ...]]
            If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
            If >= 1.0 this corresponds to the exact number of low-frequency columns to be retained.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        r"""Creates a vertical equispaced mask that exploits conjugate symmetry.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]
        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):

            center_fraction, acceleration = self.choose_acceleration()

            # This is essentially for CartesianMagicMaskFunc, indicating the excact number of low frequency lines
            # to be retained.
            if center_fraction > 1:
                num_low_freqs = center_fraction
            # Otherwise, if < 1, it is the fraction of low frequency lines to be retained, for FastMRIMagicMaskFunc.
            else:
                num_low_freqs = int(round(num_cols * center_fraction))

            # bound the number of low frequencies between 1 and target columns
            target_cols_to_sample = int(round(num_cols / acceleration))
            num_low_freqs = max(min(num_low_freqs, target_cols_to_sample), 1)

            acs_mask = self.center_mask_func(num_cols, num_low_freqs)

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                acs_mask = acs_mask[np.newaxis].repeat(num_slc_or_time, axis=0)

            if return_acs:
                return self._reshape_and_add_coil_axis(self._broadcast_mask(acs_mask, num_rows), shape)

            # adjust acceleration rate based on target acceleration.
            adjusted_target_cols_to_sample = target_cols_to_sample - num_low_freqs
            adjusted_acceleration = 0
            if adjusted_target_cols_to_sample > 0:
                adjusted_acceleration = int(round(num_cols / adjusted_target_cols_to_sample))

            acs_mask = acs_mask.reshape(num_slc_or_time, -1)  # In case mode != MaskFuncMode.STATIC:
            mask = []
            for i in range(num_slc_or_time):
                offset = self.rng.randint(0, high=adjusted_acceleration)

                if offset % 2 == 0:
                    offset_pos = offset + 1
                    offset_neg = offset + 2
                else:
                    offset_pos = offset - 1 + 3
                    offset_neg = offset - 1 + 0

                poslen = (num_cols + 1) // 2
                neglen = num_cols - (num_cols + 1) // 2
                mask_positive = np.zeros(poslen, dtype=bool)
                mask_negative = np.zeros(neglen, dtype=bool)

                mask_positive[offset_pos::adjusted_acceleration] = True
                mask_negative[offset_neg::adjusted_acceleration] = True
                mask_negative = np.flip(mask_negative)

                mask.append(np.fft.fftshift(np.concatenate((mask_positive, mask_negative))))
                mask[i] = np.logical_or(mask[i], acs_mask[i])
            mask = np.stack(mask, axis=0).squeeze()

        return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)


class FastMRIMagicMaskFunc(MagicMaskFunc):
    """Vertical line mask function as implemented in [1]_.

    :class:`FastMRIMagicMaskFunc` exploits the conjugate symmetry via offset-sampling. It is essentially an
    equispaced mask with an offset for the opposite site of the k-space. Since MRI images often exhibit approximate
    conjugate k-space symmetry, this mask is generally more efficient than :class:`FastMRIEquispacedMaskFunc`.

    References
    ----------
    .. [1] Defazio, Aaron. “Offset Sampling Improves Deep Learning Based Accelerated MRI Reconstructions by
        Exploiting Symmetry.” ArXiv:1912.01101 [Cs, Eess], Feb. 2020. arXiv.org, http://arxiv.org/abs/1912.01101.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction (< 1.0) of low-frequency columns to be retained.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`FastMRIMagicMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction (< 1.0) of low-frequency columns to be retained.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if not all(0 < center_fraction < 1 for center_fraction in center_fractions):
            raise ValueError(
                f"Center fraction values should be between 0 and 1. Received {center_fractions}. "
                f"For exact number of center lines, use `CartesianMagicMaskFunc`."
            )

        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )


class CartesianMagicMaskFunc(MagicMaskFunc):
    r"""Cartesian equispaced mask function as implemented in [1]_.

    Similar to :class:`FastMRIMagicMaskFunc`, but instead of center fraction (`center_fractions`) representing
    the fraction of center lines to the original size, here, it represents the actual number of center lines.

    References
    ----------
    .. [1] Defazio, Aaron. “Offset Sampling Improves Deep Learning Based Accelerated MRI Reconstructions by
       Exploiting Symmetry.” ArXiv:1912.01101 [Cs, Eess], Feb. 2020. arXiv.org, http://arxiv.org/abs/1912.01101.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.
    center_fractions : Union[list[int], tuple[int, ...]]
        Number of low-frequence (center) columns to be retained.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[int], tuple[int, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`CartesianMagicMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions : Union[list[int], tuple[int, ...]]
            Number of low-frequence (center) columns to be retained.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        if not all((1 < center_fraction) and isinstance(center_fraction, int) for center_fraction in center_fractions):
            raise ValueError(
                f"Center fraction values should be integers greater then or equal to 1 corresponding to the number of "
                f"center lines. Received {center_fractions}. For fractions, use `FastMRIMagicMaskFunc`."
            )
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )


class CalgaryCampinasMaskFunc(BaseMaskFunc):
    r"""Implementation of Calgary-Campinas sampling mask function.

    Samples are drawn from pre-computed masks provided by the challenge. The masks are available for accelerations of
    5 and 10. The masks are available for shapes of :math:`218 \times 170/174/180`. The masks are downloaded from the
    challenge website and cached locally. The masks are loaded based on the shape of the input k-space data. The masks
    are randomly selected from the available masks for the given shape. The masks are broadcasted to the shape of the
    input k-space data. The autocalibration signal region is generated as a
    circular mask centered in the k-space with a radius of 18.

    For more information, please refer to the challenge website [1]_.

    Parameters
    ----------
    accelerations : Union[list[int], tuple[int, ...]]
        Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
        mask_type. Has to be the same length as center_fractions if uniform_range is not True.

    Raises
    ------
    ValueError
        If the acceleration is not 5 or 10.

    References
    ----------
    .. [1] https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge?authuser=0
    """

    BASE_URL = "https://s3.aiforoncology.nl/direct-project/calgary_campinas_masks/"
    MASK_MD5S = {
        "R10_218x170.npy": "6e1511c33dcfc4a960f526252676f7c3",
        "R10_218x174.npy": "78fe23ae5eed2d3a8ff3ec128388dcc9",
        "R10_218x180.npy": "5039a6c19ac2aa3472a94e4b015e5228",
        "R5_218x170.npy": "6599715103cf3d71d6e87d09f865e7da",
        "R5_218x174.npy": "5bd27d2da3bf1e78ad1b65c9b5e4b621",
        "R5_218x180.npy": "717b51f3155c3a64cfaaddadbe90791d",
    }

    # TODO: Configuration improvements, so no **kwargs needed.
    # pylint: disable=unused-argument
    def __init__(self, accelerations: Union[list[int], tuple[int, ...]], **kwargs) -> None:  # noqa
        """Inits :class:`CalgaryCampinasMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[int], tuple[int, ...]]
            Amount of under-sampling. Only 5 and 10 are supported.

        Raises
        ------
        ValueError
            If the acceleration is not 5 or 10.

        """
        super().__init__(accelerations=accelerations, uniform_range=False)

        if not all(_ in [5, 10] for _ in accelerations):
            raise ValueError("CalgaryCampinas only provide 5x and 10x acceleration masks.")

        self.masks = {}
        self.shapes: list[Number] = []

        for acceleration in accelerations:
            self.masks[acceleration] = self.__load_masks(acceleration)

    @staticmethod
    def circular_centered_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
        """Creates a circular centered mask.

        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the mask as a two-dim tuple.
        radius : int
            Radius of the circular mask.

        Returns
        -------
        np.ndarray
            The circular mask.
        """
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
        return mask[np.newaxis, ..., np.newaxis]

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        r"""Downloads and loads pre-computed Poisson masks.

        Currently supports shapes of :math:`218 \times 170/174/180` and acceleration factors of `5` or `10`.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """
        shape = tuple(shape)[:-1]
        if return_acs:
            return torch.from_numpy(self.circular_centered_mask(shape, 18))

        if shape not in self.shapes:
            raise ValueError(f"No mask of shape {shape} is available in the CalgaryCampinas dataset.")

        with temp_seed(self.rng, seed):
            acceleration = self.choose_acceleration()
            masks = self.masks[acceleration]

            mask, num_masks = masks[shape]
            # Randomly pick one example
            choice = self.rng.randint(0, num_masks)

        return torch.from_numpy(mask[choice][np.newaxis, ..., np.newaxis])

    def __load_masks(self, acceleration: int) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
        """Loads Calgary-Campinas masks for a given acceleration.

        Parameters
        ----------
        acceleration : int
            Acceleration factor. Can be 5 or 10.

        Returns
        -------
        dict[tuple[int, int], tuple[np.ndarray, int]]
            Dictionary of masks with the shape as key and the mask and number of masks as value.

        Raises
        ------
        RuntimeError
            If the download fails.
        """
        masks_path = DIRECT_CACHE_DIR / "calgary_campinas_masks"
        paths = [
            f"R{acceleration}_218x170.npy",
            f"R{acceleration}_218x174.npy",
            f"R{acceleration}_218x180.npy",
        ]

        downloaded = [download_url(self.BASE_URL + _, masks_path, md5=self.MASK_MD5S[_]) is None for _ in paths]
        if not all(downloaded):
            raise RuntimeError(f"Failed to download all Calgary-Campinas masks from {self.BASE_URL}.")

        output = {}
        for path in paths:
            shape = [int(_) for _ in path.split("_")[-1][:-4].split("x")]
            self.shapes.append(tuple(shape))
            mask_array = np.load(masks_path / path)
            output[tuple(shape)] = mask_array, mask_array.shape[0]

        return output


class CIRCUSSamplingMode(DirectEnum):
    """Enumeration of CIRCUS sampling modes.

    CIRCUS sampling modes are radial and spiral.
    """

    CIRCUS_RADIAL = "circus-radial"
    CIRCUS_SPIRAL = "circus-spiral"


class CIRCUSMaskFunc(BaseMaskFunc):
    """Implementation of Cartesian undersampling (radial or spiral) using CIRCUS as shown in [1]_.

    It creates radial or spiral masks for Cartesian acquired data on a grid.

    Parameters
    ----------
    subsampling_scheme : CIRCUSSamplingMode
        The subsampling scheme to use. Can be either `CIRCUSSamplingMode.CIRCUS_RADIAL` or
        `CIRCUSSamplingMode.CIRCUS_SPIRAL`.
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]], optional
        Fraction (< 1.0) of low-frequency samples to be retained. If None, it will calculate the acs mask based on the
        maximum masked disk in the generated mask (with a tolerance).Default: None.
    uniform_range : bool
        If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.

    References
    ----------

    .. [1] Liu J, Saloner D. Accelerated MRI with CIRcular Cartesian UnderSampling (CIRCUS): a variable density
        Cartesian sampling strategy for compressed sensing and parallel imaging. Quant Imaging Med Surg.
        2014 Feb;4(1):57-67. doi: 10.3978/j.issn.2223-4292.2014.02.01. PMID: 24649436; PMCID: PMC3947985.
    """

    def __init__(
        self,
        subsampling_scheme: CIRCUSSamplingMode,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Optional[Union[list[float], tuple[float, ...]]] = None,
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`CIRCUSMaskFunc`.

        Parameters
        ----------
        subsampling_scheme : CIRCUSSamplingMode
            The subsampling scheme to use. Can be either `CIRCUSSamplingMode.CIRCUS_RADIAL` or
            `CIRCUSSamplingMode.CIRCUS_SPIRAL`.
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]], optional
            Fraction (< 1.0) of low-frequency samples to be retained. If None, it will calculate the acs mask
            based on the maximum masked disk in the generated mask (with a tolerance).Default: None.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.

        Raises
        ------
        NotImplementedError
            If the subsampling_scheme is not `CIRCUSSamplingMode.CIRCUS_RADIAL` or `CIRCUSSamplingMode.CIRCUS_SPIRAL`.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions if center_fractions else tuple(0 for _ in range(len(accelerations))),
            uniform_range=uniform_range,
            mode=mode,
        )
        if subsampling_scheme not in [CIRCUSSamplingMode.CIRCUS_RADIAL, CIRCUSSamplingMode.CIRCUS_SPIRAL]:
            raise NotImplementedError(
                f"Currently CIRCUSMaskFunc is only implemented for 'circus-radial' or 'circus-spiral' "
                f"as a subsampling_scheme. Got subsampling_scheme={subsampling_scheme}."
            )

        self.subsampling_scheme = "circus-radial" if subsampling_scheme is None else subsampling_scheme

    @staticmethod
    def get_square_ordered_idxs(square_side_size: int, square_id: int) -> tuple[tuple, ...]:
        """Returns ordered (clockwise) indices of a sub-square of a square matrix.

        Parameters
        ----------
        square_side_size : int
            Square side size. Dim of array.
        square_id : int
            Number of sub-square. Can be 0, ..., square_side_size // 2.

        Returns
        -------
        ordered_idxs : list of tuples.
            Indices of each point that belongs to the square_id-th sub-square
            starting from top-left point clockwise.
        """
        assert square_id in range(square_side_size // 2)

        ordered_idxs = list()

        for col in range(square_id, square_side_size - square_id):
            ordered_idxs.append((square_id, col))

        for row in range(square_id + 1, square_side_size - (square_id + 1)):
            ordered_idxs.append((row, square_side_size - (square_id + 1)))

        for col in range(square_side_size - (square_id + 1), square_id, -1):
            ordered_idxs.append((square_side_size - (square_id + 1), col))

        for row in range(square_side_size - (square_id + 1), square_id, -1):
            ordered_idxs.append((row, square_id))

        return tuple(ordered_idxs)

    def circus_radial_mask(self, shape: tuple[int, int], acceleration: Number) -> torch.Tensor:
        # pylint: disable=too-many-locals
        """Implements CIRCUS radial undersampling.

        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the mask.
        acceleration : Number
            The acceleration factor.

        Returns
        -------
        torch.Tensor
            The radial mask.
        """
        max_dim = max(shape) - max(shape) % 2
        min_dim = min(shape) - min(shape) % 2
        num_nested_squares = max_dim // 2
        M = int(np.prod(shape) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))

        mask = np.zeros((max_dim, max_dim), dtype=np.float32)

        t = self.rng.randint(low=0, high=1e4, size=1, dtype=int).item()

        for square_id in range(num_nested_squares):
            ordered_indices = self.get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )
            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad, constant_values=0)
        mask = T.center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    def circus_spiral_mask(self, shape: tuple[int, int], acceleration: Number) -> torch.Tensor:
        # pylint: disable=too-many-locals
        """Implements CIRCUS spiral undersampling.

        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the mask.
        acceleration : Number
            The acceleration factor.

        Returns
        -------
        torch.Tensor
            The spiral mask.
        """
        max_dim = max(shape) - max(shape) % 2
        min_dim = min(shape) - min(shape) % 2

        num_nested_squares = max_dim // 2

        M = int(np.prod(shape) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))

        mask = np.zeros((max_dim, max_dim), dtype=np.float32)

        c = self.rng.uniform(low=1.1, high=1.3, size=1).item()

        for square_id in range(num_nested_squares):
            ordered_indices = self.get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )

            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                i = np.floor(np.mod(m / GOLDEN_RATIO, 1) * K)
                indices_idx = int(np.mod((i + np.ceil(J**c) - 1), K))

                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad)
        mask = T.center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    @staticmethod
    def circular_centered_mask(mask: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """Finds the maximum (masked) disk in mask given a tolerance.

        Parameters
        ----------
        mask : torch.Tensor
            The mask.
        eps : float, optional
            Tolerance for the disk radius. The disk radius is increased until the ratio of the disk area to the
            intersection area is greater than 1 + eps. Default: 0.1

        Returns
        -------
        torch.Tensor
            _description_
        """

        shape = mask.shape
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        Y, X = torch.tensor(Y), torch.tensor(X)
        radius = 1

        while True:
            # Creates a disk with R=radius and finds intersection with mask
            disk = (Y - center[0]) ** 2 + (X - center[1]) ** 2 <= radius**2
            intersection = disk & mask
            ratio = disk.sum() / intersection.sum()
            if ratio > 1.0 + eps:
                return intersection
            radius += eps

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Produces :class:`CIRCUSMaskFunc` sampling masks.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """

        num_rows = shape[-3]
        num_cols = shape[-2]

        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()

            if center_fraction == 0:
                acs_mask = []
                mask = []
                for _ in range(num_slc_or_time):
                    if self.subsampling_scheme == "circus-radial":
                        mask.append(
                            self.circus_radial_mask(
                                shape=(num_rows, num_cols),
                                acceleration=acceleration,
                            )
                        )
                    elif self.subsampling_scheme == "circus-spiral":
                        mask.append(
                            self.circus_spiral_mask(
                                shape=(num_rows, num_cols),
                                acceleration=acceleration,
                            )
                        )
                    acs_mask.append(self.circular_centered_mask(mask[-1]))
                mask = torch.stack(mask, dim=0).squeeze()
                acs_mask = torch.stack(acs_mask, dim=0).squeeze()
                acs_mask = self._reshape_and_add_coil_axis(acs_mask, shape)
                if return_acs:
                    return acs_mask

            else:
                acs_mask = centered_disk_mask((num_rows, num_cols), center_fraction)
                num_low_freqs = acs_mask.sum()
                adjusted_accel = (acceleration * (num_low_freqs - num_rows * num_cols)) / (
                    num_low_freqs * acceleration - num_rows * num_cols
                )

                if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                    acs_mask = acs_mask[np.newaxis].repeat(num_slc_or_time, axis=0)

                acs_mask = self._reshape_and_add_coil_axis(acs_mask, shape)

                if return_acs:
                    return acs_mask

                mask = []
                for _ in range(num_slc_or_time):
                    if self.subsampling_scheme == "circus-radial":
                        mask.append(
                            self.circus_radial_mask(
                                shape=(num_rows, num_cols),
                                acceleration=adjusted_accel,
                            )
                        )
                    elif self.subsampling_scheme == "circus-spiral":
                        mask.append(
                            self.circus_spiral_mask(
                                shape=(num_rows, num_cols),
                                acceleration=adjusted_accel,
                            )
                        )

                mask = torch.stack(mask, dim=0).squeeze()
            return self._reshape_and_add_coil_axis(mask, shape) | acs_mask


class RadialMaskFunc(CIRCUSMaskFunc):
    """Computes CIRCUS radial masks for Cartesian data.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]], optional
        Fraction (< 1.0) of low-frequency samples to be retained. If None, it will calculate the acs mask
        based on the maximum masked disk in the generated mask (with a tolerance).Default: None.
    uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Optional[Union[list[float], tuple[float, ...]]] = None,
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`RadialMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]], optional
            Fraction (< 1.0) of low-frequency samples to be retained. If None, it will calculate the acs mask
            based on the maximum masked disk in the generated mask (with a tolerance).Default: None.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            subsampling_scheme=CIRCUSSamplingMode.CIRCUS_RADIAL,
            uniform_range=uniform_range,
            mode=mode,
        )


class SpiralMaskFunc(CIRCUSMaskFunc):
    """Computes CIRCUS spiral masks for Cartesian data.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]], optional
        Fraction (< 1.0) of low-frequency samples to be retained. If None, it will calculate the acs mask
        based on the maximum masked disk in the generated mask (with a tolerance).Default: None.
    uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Optional[Union[list[float], tuple[float, ...]]] = None,
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`SpiralMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]], optional
            Fraction (< 1.0) of low-frequency samples to be retained. If None, it will calculate the acs mask
            based on the maximum masked disk in the generated mask (with a tolerance).Default: None.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values. Default: False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            subsampling_scheme=CIRCUSSamplingMode.CIRCUS_SPIRAL,
            uniform_range=uniform_range,
            mode=mode,
        )


class VariableDensityPoissonMaskFunc(BaseMaskFunc):
    r"""Variable Density Poisson sampling mask function. Based on [1]_.

    Parameters
    ----------
    accelerations : list or tuple of positive numbers
        Amount of under-sampling.
    center_fractions : list or tuple of floats
        Must have the same length as `accelerations`. Amount of center fully-sampling.
        For center_scale='r', then a centered disk area with radius equal to
        :math:`R = \sqrt{{n_r}^2 + {n_c}^2} \\times r` will be fully sampled, where :math:`n_r` and :math:`n_c`
        denote the input shape.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    crop_corner : bool, optional
        If True mask will be disk. Default: False.
    max_attempts : int, optional
        Maximum rejection samples. Default: 10.
    tol : float, optional
        Maximum deviation between the generated mask acceleration and the desired acceleration. Default: 0.2.
    slopes : Optional[Union[list[float], tuple[float, ...]]]
        An increasing sequence of non-negative floats (of length 2) to be used
        for the generation of the sampling radius. Default: None.

    Notes
    -----

    * Code inspired and modified from [2]_ with BSD-3 licence, Copyright (c) 2016, Frank Ong, Copyright (c) 2016,
        The Regents of the University of California [3]_.

    References
    ----------

    .. [1] Bridson, Robert. “Fast Poisson Disk Sampling in Arbitrary Dimensions.” ACM SIGGRAPH 2007
        Sketches on - SIGGRAPH ’07, ACM Press, 2007, pp. 22-es. DOI.org (Crossref),
        https://doi.org/10.1145/1278780.1278807.
    .. [2] https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/samp.py#L11
    .. [3] https://github.com/mikgroup/sigpy/blob/master/LICENSE

    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        mode: MaskFuncMode = MaskFuncMode.STATIC,
        crop_corner: Optional[bool] = False,
        max_attempts: Optional[int] = 10,
        tol: Optional[float] = 0.2,
        slopes: Optional[Union[list[float], tuple[float, ...]]] = None,
    ) -> None:
        r"""Inits :class:`VariableDensityPoissonMaskFunc`.

        Parameters
        ----------
        accelerations : list or tuple of positive numbers
            Amount of under-sampling.
        center_fractions : list or tuple of floats
            Must have the same length as `accelerations`. Amount of center fully-sampling.
            For center_scale='r', then a centered disk area with radius equal to
            :math:`R = \sqrt{{n_r}^2 + {n_c}^2} \times r` will be fully sampled, where :math:`n_r` and :math:`n_c`
            denote the input shape.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        crop_corner : bool, optional
            If True mask will be disk. Default: False.
        max_attempts : int, optional
            Maximum rejection samples. Default: 10.
        tol : float, optional
            Maximum deviation between the generated mask acceleration and the desired acceleration. Default: 0.2.
        slopes : Optional[Union[list[float], tuple[float, ...]]]
            An increasing sequence of non-negative floats (of length 2) to be used
            for the generation of the sampling radius. Default: None.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=False,
            mode=mode,
        )
        self.crop_corner = crop_corner
        self.max_attempts = max_attempts
        self.tol = tol
        if slopes is not None:
            assert (
                slopes[0] >= 0 and slopes[0] < slopes[1] and len(slopes) == 2
            ), f"`slopes` must be an increasing sequence of two non-negative floats. Received {slopes}."
        self.slopes = slopes

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Produces variable Density Poisson sampling masks.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask of shape (1, shape[0], shape[1], 1).
        """
        num_rows, num_cols = shape[-3:-1]
        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):
            self.rng.seed(integerize_seed(seed))

            center_fraction, acceleration = self.choose_acceleration()

            if return_acs:
                acs_mask = centered_disk_mask((num_rows, num_cols), center_fraction)
                if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                    acs_mask = acs_mask[np.newaxis].repeat(num_slc_or_time, axis=0)
                return self._reshape_and_add_coil_axis(acs_mask, shape)

            mask = []
            for _ in range(num_slc_or_time):
                mask.append(self.poisson(num_rows, num_cols, center_fraction, acceleration, self.rng.randint(1e5)))
            mask = np.stack(mask, axis=0).squeeze()

        return self._reshape_and_add_coil_axis(mask, shape)

    def poisson(
        self,
        num_rows: int,
        num_cols: int,
        center_fraction: float,
        acceleration: float,
        seed: int = 0,
    ) -> torch.Tensor:
        """Calculates mask by calling the cython `_poisson` method.

        Parameters
        ----------
        num_rows : int
            Number of rows - x-axis size.
        num_cols : int
            Number of columns - y-axis size.
        center_fraction : float
            Amount of center fully-sampling.
        acceleration : float
            Acceleration factor.
        seed : int
            Seed to be used by cython function. Default: 0.

        Returns
        -------
        mask : torch.Tensor
            Sampling mask of shape (`num_rows`, `num_cols`).
        """
        # pylint: disable=too-many-locals
        x, y = np.mgrid[:num_rows, :num_cols]

        x = np.maximum(abs(x - num_rows / 2), 0)
        x /= x.max()
        y = np.maximum(abs(y - num_cols / 2), 0)
        y /= y.max()
        r = np.sqrt(x**2 + y**2)

        if self.slopes is not None:
            slope_min, slope_max = self.slopes
        else:
            slope_min, slope_max = 0, max(num_rows, num_cols)

        while slope_min < slope_max:
            slope = (slope_max + slope_min) / 2
            radius_x = np.clip((1 + r * slope) * num_rows / max(num_rows, num_cols), 1, None)

            radius_y = np.clip((1 + r * slope) * num_cols / max(num_rows, num_cols), 1, None)

            mask = np.zeros((num_rows, num_cols), dtype=int)

            _poisson(num_rows, num_cols, self.max_attempts, mask, radius_x, radius_y, seed)

            mask = mask | centered_disk_mask((num_rows, num_cols), center_fraction)

            if self.crop_corner:
                mask *= r < 1

            actual_acceleration = num_rows * num_cols / mask.sum()

            if abs(actual_acceleration - acceleration) < self.tol:
                break
            if actual_acceleration < acceleration:
                slope_min = slope
            else:
                slope_max = slope

        if abs(actual_acceleration - acceleration) >= self.tol:
            raise ValueError(f"Cannot generate mask to satisfy accel={acceleration}.")

        return mask


class Gaussian1DMaskFunc(CartesianVerticalMaskFunc):
    """Gaussian 1D vertical line mask function.

    This method uses Cython under the hood to generate a 1D Gaussian mask, employing rejection sampling.


    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`Gaussian1DMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a vertical gaussian mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.


        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """

        num_cols = shape[-2]
        num_rows = shape[-3]
        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):
            self.rng.seed(integerize_seed(seed))

            center_fraction, acceleration = self.choose_acceleration()
            num_low_freqs = int(round(num_cols * center_fraction))

            mask = self.center_mask_func(num_cols, num_low_freqs).astype(int)

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                mask = mask[np.newaxis].repeat(num_slc_or_time, axis=0)

            if return_acs:
                return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)

            # Calls cython function
            nonzero_count = int(np.round(num_cols / acceleration - num_low_freqs - 1))

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                for i in range(num_slc_or_time):
                    gaussian_mask_1d(
                        nonzero_count,
                        num_cols,
                        num_cols // 2,
                        6 * np.sqrt(num_cols // 2),
                        mask[i],
                        self.rng.randint(1e5),
                    )
                mask = mask.squeeze()
            else:
                gaussian_mask_1d(
                    nonzero_count, num_cols, num_cols // 2, 6 * np.sqrt(num_cols // 2), mask, self.rng.randint(1e5)
                )

        return self._reshape_and_add_coil_axis(self._broadcast_mask(mask, num_rows), shape)


class Gaussian2DMaskFunc(BaseMaskFunc):
    """Gaussian 2D mask function.

    This method uses Cython under the hood to generate a 2D Gaussian mask, employing rejection sampling.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency samples (float < 1.0) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        mode: MaskFuncMode = MaskFuncMode.STATIC,
    ) -> None:
        """Inits :class:`Gaussian2DMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency samples (float < 1.0) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        mode : MaskFuncMode
            Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
            If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
            broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
            this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
            along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
            slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=mode,
        )

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a 2D gaussian mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.


        Returns
        -------
        mask : torch.Tensor
            The sampling mask.
        """
        num_rows, num_cols = shape[-3:-1]
        num_slc_or_time = shape[-4] if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE] else 1

        with temp_seed(self.rng, seed):
            self.rng.seed(integerize_seed(seed))

            center_fraction, acceleration = self.choose_acceleration()

            mask = centered_disk_mask((num_rows, num_cols), center_fraction)

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                mask = mask[np.newaxis].repeat(num_slc_or_time, axis=0)

            if return_acs:
                return self._reshape_and_add_coil_axis(mask, shape)

            std = 6 * np.array([np.sqrt(num_rows // 2), np.sqrt(num_cols // 2)], dtype=float)

            if self.mode in [MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE]:
                for i in range(num_slc_or_time):
                    # Calls cython function
                    gaussian_mask_2d(
                        int(np.round(num_cols * num_rows / acceleration - mask[i].sum() - 1)),
                        num_rows,
                        num_cols,
                        num_rows // 2,
                        num_cols // 2,
                        std,
                        mask[i],
                        self.rng.randint(1e5),
                    )
                mask = mask.squeeze()
            else:
                nonzero_count = int(np.round(num_cols * num_rows / acceleration - mask.sum() - 1))
                # Calls cython function
                gaussian_mask_2d(
                    nonzero_count, num_rows, num_cols, num_rows // 2, num_cols // 2, std, mask, self.rng.randint(1e5)
                )

        return self._reshape_and_add_coil_axis(mask, shape)


class KtBaseMaskFunc(BaseMaskFunc):
    """Base class for kt mask functions.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
    ) -> None:
        """Inits :class:`KtBaseMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
            mode=MaskFuncMode.DYNAMIC,
        )

    @staticmethod
    def zero_pad_to_center(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        """Zero pads an array to the target shape around its center.

        Parameters
        ----------
        array : ndarray
            The input array.
        target_shape : tuple of int
            The target shape for each dimension.

        Returns
        -------
        ndarray
            The zero-padded array.
        """
        current_shape = list(array.shape)

        # Extend current_shape if it has fewer dimensions than target_shape
        if len(current_shape) < len(target_shape):
            current_shape.extend([1] * (len(target_shape) - len(current_shape)))

        # If the shapes are already the same, return the original array
        if all(current_shape[i] == target_shape[i] for i in range(len(target_shape))):
            return array

        # Create an array of zeros with the target shape
        padded_array = np.zeros(target_shape, dtype=array.dtype)

        # Calculate the slices for inserting the original array into the padded array
        insert_slices = tuple(
            slice((target_dim - current_dim) // 2, (target_dim - current_dim) // 2 + current_dim)
            for target_dim, current_dim in zip(target_shape, current_shape)
        )

        # Insert the original array into the padded array
        padded_array[insert_slices] = array

        return padded_array

    @staticmethod
    def linear_indices_to_2d_coordinates(indices: np.ndarray, row_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Converts linear indices to 2D coordinates.

        Parameters
        ----------
        indices : ndarray
            The linear indices to convert.
        row_length : int
            The length of the rows in the 2D array.

        Returns
        -------
        tuple of ndarray
            The x and y coordinates.
        """
        x_coords = indices - np.floor((indices - 1) / row_length) * row_length
        y_coords = np.ceil(indices / row_length)
        return x_coords.astype(int), y_coords.astype(int)

    @staticmethod
    def find_nearest_empty_location(target_index: int, empty_indices: np.ndarray, row_length: int) -> int:
        """Finds the nearest empty index to the target index in 2D space.

        Parameters
        ----------
        target_index : int
            The index of the target point.
        empty_indices : ndarray
            The indices of empty locations.
        row_length : int
            The length of the rows in the 2D array.

        Returns
        -------
        int
            The nearest empty index.
        """
        x0, y0 = KtBaseMaskFunc.linear_indices_to_2d_coordinates(target_index, row_length)
        x, y = KtBaseMaskFunc.linear_indices_to_2d_coordinates(empty_indices, row_length)

        distance_x = (x - x0) ** 2
        distance_y = (y - y0) ** 2
        distance_y = distance_y.astype(float)
        distance_y[distance_y > np.finfo(float).eps] = np.inf  # Preventing zero distance consideration
        distance = np.sqrt(distance_x + distance_y)

        nearest_index = np.argmin(distance)
        return empty_indices[nearest_index]

    @staticmethod
    def resolve_duplicates_on_kt_grid(
        phase: np.ndarray, time: np.ndarray, ny: int, nt: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corrects overlapping trajectories in k-space by shifting points to the nearest vacant locations.

        Parameters
        ----------
        phase : ndarray
            The phase coordinates of the trajectories.
        time : ndarray
            The time coordinates of the trajectories.
        ny : int
            The number of phase encoding steps.
        nt : int
            The number of time encoding steps.

        Returns
        -------
        tuple of ndarray
            Corrected phase and time coordinates.
        """
        # pylint: disable=too-many-locals
        phase_corrected = phase + np.ceil((ny + 1) / 2)
        time_corrected = time + np.ceil((nt + 1) / 2)
        trajectory_indices = (time_corrected - 1) * ny + phase_corrected

        unique_indices, counts = np.unique(trajectory_indices, return_counts=True)
        repeated_values = unique_indices[counts != 1]
        duplicate_indices = []

        for value in repeated_values:
            duplicates = np.where(trajectory_indices == value)[0]
            duplicate_indices.extend(duplicates[1:])

        empty_indices = np.setdiff1d(np.arange(1, ny * nt + 1), trajectory_indices)

        for _, duplicate_index in enumerate(duplicate_indices):
            new_index = KtBaseMaskFunc.find_nearest_empty_location(
                trajectory_indices[duplicate_index], empty_indices, ny
            )
            trajectory_indices[duplicate_index] = new_index
            empty_indices = np.setdiff1d(empty_indices, new_index)

        phase_corrected, time_corrected = KtBaseMaskFunc.linear_indices_to_2d_coordinates(trajectory_indices, ny)
        phase_corrected = phase_corrected - np.ceil((ny + 1) / 2)
        time_corrected = time_corrected - np.ceil((nt + 1) / 2)

        return phase_corrected, time_corrected

    @staticmethod
    def crop_center(array, target_height, target_width):
        """Crops the center of an array to the target height and width.

        Parameters
        ----------
        array : ndarray
            The input array.
        target_height : int
            The target height.
        target_width : int
            The target width.

        Returns
        -------
        ndarray
            The cropped array.
        """
        target_shape = [target_height, target_width]
        current_shape = list(array.shape)

        # Extend target_shape if it has fewer dimensions than current_shape
        if len(target_shape) < len(current_shape):
            target_shape.extend([1] * (len(current_shape) - len(target_shape)))

        # If the shapes are already the same, return the original array
        if current_shape == target_shape:
            return array

        # Calculate the slices for cropping the array
        crop_slices = tuple(
            slice((current_dim - target_dim) // 2, (current_dim - target_dim) // 2 + target_dim)
            for current_dim, target_dim in zip(current_shape, target_shape)
        )

        # Crop the array
        cropped_array = array[crop_slices]
        return cropped_array


class KtRadialMaskFunc(KtBaseMaskFunc):
    """Kt radial mask function.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    crop_corner : bool, optional
        If True, the mask is cropped to the corners. Default: False.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        crop_corner: bool = False,
    ) -> None:
        """Inits :class:`KtRadialMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        crop_corner : bool, optional
            If True, the mask is cropped to the corners. Default: False.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )
        self.crop_corner = crop_corner

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a kt radial mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        torch.Tensor
            The sampling mask.
        """
        if len(shape) not in [4, 5]:
            raise ValueError("Shape should have 4 or 5 dimensions.")

        (nt, num_rows, num_cols) = shape[-4:-1]

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            offset_angle = self.rng.uniform(0, 360)

        acs_mask = centered_disk_mask((num_rows, num_cols), center_fraction)
        num_low_freqs = acs_mask.sum()
        acs_mask = np.tile(acs_mask, (nt, 1, 1))

        if return_acs:
            return torch.from_numpy(acs_mask.astype(bool)[np.newaxis, ..., np.newaxis])

        adjusted_acceleration = (acceleration * (num_low_freqs - num_rows * num_cols)) / (
            num_low_freqs * acceleration - num_rows * num_cols
        )

        rate = 1 / adjusted_acceleration
        num_beams = int(rate * np.mean([num_rows, num_cols]))  # num_beams is the number of angles

        if self.crop_corner:
            temp_size = max(num_rows, num_cols)
        else:
            temp_size = int(np.sqrt(2) * max(num_rows, num_cols))

        aux = np.zeros((temp_size, temp_size))
        aux[int(temp_size / 2), :] = 1

        base_mask = np.sum(
            [rotate(aux, angle, reshape=False, order=0) for angle in np.linspace(0, 180, num_beams)], axis=0
        )
        mask = [self.crop_center(base_mask, num_rows, num_cols)]

        nt_angles = np.linspace(offset_angle, offset_angle + 180, nt)
        for angle in nt_angles[:-1]:
            mask.append(self.crop_center(rotate(base_mask, angle, reshape=False, order=0), num_rows, num_cols))
        mask = np.stack(mask, 0)

        mask = mask + acs_mask
        mask = mask > 0

        return self._reshape_and_add_coil_axis(mask, shape)


class KtUniformMaskFunc(KtBaseMaskFunc):
    """Kt uniform mask function.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
    ) -> None:
        """Inits :class:`KtUniformMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a kt uniform mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        torch.Tensor
            The sampling mask.
        """
        if len(shape) not in [4, 5]:
            raise ValueError("Shape should have 4 or 5 dimensions.")

        (nt, num_rows, num_cols) = shape[-4:-1]

        with temp_seed(self.rng, seed):

            center_fraction, acceleration = self.choose_acceleration()
            num_low_freqs = int(round(num_cols * center_fraction))

            # Fully sampled rectangle region
            acs_mask = self.zero_pad_to_center(np.ones((nt, num_rows, num_low_freqs)), [nt, num_rows, num_cols])

            if return_acs:
                return torch.from_numpy(acs_mask.astype(bool)[np.newaxis, ..., np.newaxis])

            adjusted_acceleration = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )

            ptmp = np.zeros(num_cols)
            ttmp = np.zeros(nt)

            ptmp[
                np.round(
                    np.arange(self.rng.randint(0, adjusted_acceleration), num_cols, adjusted_acceleration)
                ).astype(int)
            ] = 1
            ttmp[
                np.round(np.arange(self.rng.randint(0, adjusted_acceleration), nt, adjusted_acceleration)).astype(int)
            ] = 1

        top_mat = toeplitz(ptmp, ttmp)
        ind = np.where(top_mat.ravel())[0]

        ph = (ind % num_cols) - (num_cols // 2)
        ti = (ind // num_cols) - (nt // 2)

        ph, ti = self.resolve_duplicates_on_kt_grid(ph, ti, num_cols, nt)
        samp = np.zeros((num_cols, nt), dtype=int)
        inds = np.round(num_cols * (ti + nt // 2) + (ph + num_cols // 2)).astype(int)
        inds[inds <= 0] = 1  # Ensure indices are within bounds
        samp.ravel()[inds] = 1

        mask = np.tile(samp, (num_rows, 1, 1)).transpose(2, 0, 1)
        mask = mask + acs_mask
        mask = mask > 0

        return self._reshape_and_add_coil_axis(mask, shape)


class KtGaussian1DMaskFunc(KtBaseMaskFunc):
    """Kt Gaussian 1D mask function.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[float], tuple[float, ...]]
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    alpha : float, optional
        0 < alpha < 1 controls sampling density; 0: uniform density, 1: maximally non-uniform density.
        Default: 0.28.
    std_scale : float, optional
        The standard deviation scaling of the Gaussian envelope for sampling density. Default: 5.0.
    """

    def __init__(
        self,
        accelerations: Union[list[Number], tuple[Number, ...]],
        center_fractions: Union[list[float], tuple[float, ...]],
        uniform_range: bool = False,
        alpha: float = 0.28,
        std_scale: float = 5.0,
    ) -> None:
        """Inits :class:`KtGaussian1DMaskFunc`.

        Parameters
        ----------
        accelerations : Union[list[Number], tuple[Number, ...]]
            Amount of under-sampling.
        center_fractions : Union[list[float], tuple[float, ...]]
            Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        uniform_range : bool, optional
            If True then an acceleration will be uniformly sampled between the two values, by default False.
        alpha : float, optional
            0 < alpha < 1 controls sampling density; 0: uniform density, 1: maximally non-uniform density.
            Default: 0.28.
        std_scale : float, optional
            The standard deviation scaling of the Gaussian envelope for sampling density. Default: 5.0.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )
        self.alpha = alpha
        self.std_scale = std_scale

    def mask_func(
        self,
        shape: Union[list[int], tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a kt Gaussian 1D mask.

        Parameters
        ----------
        shape : list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs : bool
            Return the autocalibration signal region as a mask.
        seed : int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        torch.Tensor
            The sampling mask.
        """
        if len(shape) not in [4, 5]:
            raise ValueError("Shape should have 4 or 5 dimensions.")

        (nt, num_rows, num_cols) = shape[-4:-1]

        with temp_seed(self.rng, seed):

            center_fraction, acceleration = self.choose_acceleration()
            num_low_freqs = int(round(num_cols * center_fraction))

            # Fully sampled rectangle region
            acs_mask = self.zero_pad_to_center(np.ones((nt, num_rows, num_low_freqs)), [nt, num_rows, num_cols])

            if return_acs:
                return torch.from_numpy(acs_mask.astype(bool)[np.newaxis, ..., np.newaxis])

            adjusted_acceleration = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )

            p1 = np.arange(-num_cols // 2, num_cols // 2)
            t1 = []

            tr = round(num_cols / adjusted_acceleration)  # Number of readout lines per frame (temporal resolution)
            ti = np.zeros(tr * nt, dtype=int)
            ph = np.zeros(tr * nt, dtype=int)

            sigma = num_cols / self.std_scale  # Std of the Gaussian envelope for sampling density

            prob = 0.1 + self.alpha / (1 - self.alpha + 1e-10) * np.exp(-(p1**2) / (sigma**2))

            ind = 0
            for i in range(-nt // 2, nt // 2):
                a = np.where(np.array(t1) == i)[0]
                n_tmp = tr - len(a)
                prob_tmp = prob.copy()
                prob_tmp[a] = 0
                p_tmp = self.rng.choice(np.arange(-num_cols // 2, num_cols // 2), n_tmp, p=prob_tmp / prob_tmp.sum())
                ti[ind : ind + n_tmp] = i
                ph[ind : ind + n_tmp] = p_tmp
                ind += n_tmp

            ph, ti = self.resolve_duplicates_on_kt_grid(ph, ti, num_cols, nt)
            samp = np.zeros((nt, num_cols), dtype=int)
            inds = np.round(num_cols * (ti + nt // 2) + (ph + num_cols // 2)).astype(int)
            samp.ravel()[inds] = 1
            samp = samp.T

            mask = np.tile(samp, (num_rows, 1, 1)).transpose(2, 0, 1)
            mask = mask + acs_mask
            mask = mask > 0

            return self._reshape_and_add_coil_axis(mask, shape)


def integerize_seed(seed: Union[None, tuple[int, ...], list[int]]) -> int:
    """Returns an integer seed.

    If input is integer, will return the input. If input is None, will return a random integer seed.
    If input is a tuple or list, will return a random integer seed based on the input.

    Can be useful for functions that take as input only integer seeds (e.g. cython functions).

    Parameters
    ----------
    seed : int, tuple or list of ints, None
         Input seed to integerize.

    Returns
    -------
    out_seed: int
        Integer seed.
    """
    if isinstance(seed, int):
        return seed
    rng = np.random.RandomState()
    if seed is None:
        return rng.randint(0, 1e6)
    if isinstance(seed, (tuple, list)):
        with temp_seed(rng, seed):
            return rng.randint(0, 1e6)
    raise ValueError(f"Seed should be an integer, a tuple or a list of integers, or None. Got {type(seed)}.")


def centered_disk_mask(shape: Union[list[int], tuple[int, ...]], center_scale: float) -> np.ndarray:
    r"""Creates a mask with a centered disk of radius :math:`R=\sqrt{c_x \cdot c_y \cdot r / \pi}`.

    Parameters
    ----------
    shape : list or tuple of ints
        The shape of the (2D) mask to be created.
    center_scale : float
        Center scale.

    Returns
    -------
    mask : np.ndarray
    """
    center_x = shape[0] // 2
    center_y = shape[1] // 2

    X, Y = np.indices(shape)
    radius = int(np.sqrt(np.prod(shape) * center_scale / np.pi))

    mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius**2
    return mask.astype(int)


def build_masking_function(
    name: str,
    accelerations: Union[list[Number], tuple[Number, ...]],
    center_fractions: Optional[Union[list[Number], tuple[Number, ...]]] = None,
    uniform_range: bool = False,
    mode: MaskFuncMode = MaskFuncMode.STATIC,
    **kwargs: dict[str, Any],
) -> BaseMaskFunc:
    """Builds a mask function.

    Parameters
    ----------
    name : str
        Name of the mask function.
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[Number], tuple[Number, ...]], optional
        Fraction of low-frequency columns (float < 1.0) or number of low-frequence columns (integer) to be retained.
        If multiple values are provided, then one of these numbers is chosen uniformly each time. If uniform_range
        is True, then two values should be given, by default None.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode, optional
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    **kwargs : dict[str, Any], optional
        Additional keyword arguments to be passed to the mask function. These will be passed as keyword arguments
        to the mask function constructor. If the mask function constructor does not accept these arguments, they will
        be ignored.

    Returns
    -------
    BaseMaskFunc
        The mask function.
    """
    MaskFunc: BaseMaskFunc = str_to_class("direct.common.subsample", name + "MaskFunc")  # noqa

    # Inspect the constructor of the MaskFunc class to get its parameters
    constructor_params = inspect.signature(MaskFunc.__init__).parameters

    # Prepare the arguments to be passed, starting with those we know we want to pass
    init_args = {
        "accelerations": accelerations,
    }

    kwargs.update(
        {
            "center_fractions": center_fractions,
            "uniform_range": uniform_range,
            "mode": mode,
        }
    )
    # Now, iterate over the kwargs
    for key, value in kwargs.items():
        # If the class constructor accepts a **kwargs argument, or the key is explicitly defined in the
        # constructor parameters, include it in the init_args
        if "kwargs" in constructor_params or key in constructor_params:
            init_args[key] = value

    # Create the MaskFunc instance with the prepared arguments
    mask_func = MaskFunc(**init_args)

    return mask_func
