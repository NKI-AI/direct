# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

import numpy as np
import torch
import pathlib

from typing import Tuple, Optional
from abc import abstractmethod

from direct.utils import str_to_class
from direct.types import Number

import logging
import contextlib

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class BaseMaskFunc:
    """
    BaseMaskFunc is the base class to create a sub-sampling mask of a given shape.
    """

    def __init__(
        self,
        accelerations: Optional[Tuple[Number, ...]],
        center_fractions: Optional[Tuple[float, ...]] = None,
        uniform_range: bool = True,
    ):
        """
        Parameters
        ----------
        center_fractions : List([float])
            Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is chosen uniformly each time. If uniform_range
            is True, then two values should be given.
        accelerations : List([int])
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is True.
        uniform_range : bool
            If True then an acceleration will be uniformly sampled between the two values.
        """
        if center_fractions is not None:
            if len(center_fractions) != len(accelerations):
                raise ValueError(
                    f"Number of center fractions should match number of accelerations. "
                    f"Got {len(center_fractions)} {len(accelerations)}."
                )

        self.center_fractions = center_fractions
        self.accelerations = accelerations

        self.uniform_range = uniform_range

        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        if not self.accelerations:
            return None

        if not self.uniform_range:
            choice = self.rng.randint(0, len(self.accelerations))
            acceleration = self.accelerations[choice]
            if self.center_fractions is None:
                return acceleration

            center_fraction = self.center_fractions[choice]
            return center_fraction, acceleration

        else:
            # if self.uniform_range and (len(self.center_fractions) != 2 or len(self.accelerations) != 2):
            #     raise ValueError(f'Center fractions and accelerations have to'
            #                      f' have length two when uniform range is set. '
            #                      f'Got {self.center_fractions} and {self.accelerations}.')
            raise NotImplementedError("Uniform range is not yet implemented.")

    @abstractmethod
    def mask_func(self, shape):
        raise NotImplementedError(
            f"This method should be implemented by a child class."
        )

    def __call__(self, data, seed=None, return_acs=False):
        """
        Parameters
        ----------
        data : object
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.
        return_acs : bool
            Return the autocalibration signal region as a mask.

        Returns
        -------
        ndarray
        """
        self.rng.seed(seed)
        mask = self.mask_func(data, return_acs=return_acs)
        return mask


class FastMRIRandomMaskFunc(BaseMaskFunc):
    def __init__(
        self,
        accelerations: Tuple[Number, ...],
        center_fractions: Optional[Tuple[float, ...]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(self, shape, return_acs=False, seed=None):
        """
        Create vertical line mask.
        Code from: https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py

        The mask selects a subset of columns from the input k-space data. If the k-space data has N
        columns, the mask picks out:
            1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
               low-frequencies
            2. The other columns are selected uniformly at random with a probability equal to:
               prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
        This ensures that the expected number of columns selected is equal to (N / acceleration)

        It is possible to use multiple center_fractions and accelerations, in which case one possible
        (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
        called.

        For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
        is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
        probability that 8-fold acceleration with 4% center fraction is selected.

        Parameters
        ----------

        data : iterable[int]
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.
        return_acs : bool
            Return the autocalibration signal region as a mask.

        Returns
        -------
        torch.Tensor : the sampling mask

        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_rows = shape[-3]
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # Create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # Reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.int32)
            mask_shape[-3] = num_rows

            mask = np.broadcast_to(mask, mask_shape)[
                np.newaxis, ...
            ].copy()  # Add coil axis, make array writable.

            # TODO: Think about making this more efficient.
            if return_acs:
                acs_mask = np.zeros_like(mask)
                acs_mask[:, :, pad : pad + num_low_freqs, ...] = 1
                return torch.from_numpy(acs_mask)

        return torch.from_numpy(mask)


class FastMRIEquispacedMaskFunc(BaseMaskFunc):
    def __init__(
        self,
        accelerations: Tuple[Number, ...],
        center_fractions: Optional[Tuple[float, ...]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(self, shape, return_acs=False, seed=None):
        """
        Create equispaced vertical line mask.
        Code from: https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py

        FastMRIEquispacedMaskFunc creates a sub-sampling mask of a given shape.
        The mask selects a subset of columns from the input k-space data. If the
        k-space data has N columns, the mask picks out:
            1. N_low_freqs = (N * center_fraction) columns in the center
               corresponding tovlow-frequencies.
            2. The other columns are selected with equal spacing at a proportion
               that reaches the desired acceleration rate taking into consideration
               the number of low frequencies. This ensures that the expected number
               of columns selected is equal to (N / acceleration)
        It is possible to use multiple center_fractions and accelerations, in which
        case one possible (center_fraction, acceleration) is chosen uniformly at
        random each time the EquispacedMaskFunc object is called.

        Note that this function may not give equispaced samples (documented in
        https://github.com/facebookresearch/fastMRI/issues/54), which will require
        modifications to standard GRAPPA approaches. Nonetheless, this aspect of
        the function has been preserved to match the public multicoil data.

        Parameters
        ----------

        data : iterable[int]
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.
        return_acs : bool
            Return the autocalibration signal region as a mask.

        Returns
        -------
        torch.Tensor : the sampling mask

        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_rows = shape[-3]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # Reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.int32)
            mask_shape[-3] = num_rows

            mask = np.broadcast_to(mask, mask_shape)[
                np.newaxis, ...
            ].copy()  # Add coil axis, make array writable.

            if return_acs:
                acs_mask = np.zeros_like(mask)
                acs_mask[:, :, pad : pad + num_low_freqs, ...] = 1
                return torch.from_numpy(acs_mask)

        return torch.from_numpy(mask)


class CalgaryCampinasMaskFunc(BaseMaskFunc):
    # TODO: Configuration improvements, so no **kwargs needed.
    def __init__(self, accelerations: Tuple[int, ...], **kwargs):  # noqa
        super().__init__(accelerations=accelerations, uniform_range=False)

        if not all([_ in [5, 10] for _ in accelerations]):
            raise ValueError(
                f"CalgaryCampinas only provide 5x and 10x acceleration masks."
            )

        self.masks = {}
        self.shapes = []

        for acceleration in accelerations:
            self.masks[acceleration] = self.__load_masks(acceleration)

    @staticmethod
    def circular_centered_mask(shape, radius):
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
        return mask[np.newaxis, ..., np.newaxis]

    def mask_func(self, shape, return_acs=False):
        shape = tuple(shape)[:-1]
        if return_acs:
            return torch.from_numpy(self.circular_centered_mask(shape, 18))

        if shape not in self.shapes:
            raise ValueError(
                f"No mask of shape {shape} is available in the CalgaryCampinas dataset."
            )

        acceleration = self.choose_acceleration()
        masks = self.masks[acceleration]

        mask, num_masks = masks[shape]
        # Randomly pick one example
        choice = self.rng.randint(0, num_masks)
        return torch.from_numpy(mask[choice][np.newaxis, ..., np.newaxis])

    def __load_masks(self, acceleration):
        masks_path = pathlib.Path(
            pathlib.Path(__file__).resolve().parent / "calgary_campinas_masks"
        )
        paths = [
            f"R{acceleration}_218x170.npy",
            f"R{acceleration}_218x174.npy",
            f"R{acceleration}_218x180.npy",
        ]
        output = {}
        for path in paths:
            shape = [int(_) for _ in path.split("_")[-1][:-4].split("x")]
            self.shapes.append(tuple(shape))
            mask_array = np.load(masks_path / path)
            output[tuple(shape)] = mask_array, mask_array.shape[0]

        return output


class DictionaryMaskFunc(BaseMaskFunc):
    def __init__(self, data_dictionary, **kwargs):  # noqa
        super().__init__(accelerations=None)

        self.data_dictionary = data_dictionary

    def mask_func(self, data, return_acs=False):
        return self.data_dictionary[data]


def build_masking_function(
    name, accelerations, center_fractions=None, uniform_range=False, **kwargs
):

    MaskFunc: BaseMaskFunc = str_to_class(
        "direct.common.subsample", name + "MaskFunc"
    )  # noqa
    mask_func = MaskFunc(
        accelerations=accelerations,
        center_fractions=center_fractions,
        uniform_range=uniform_range,
    )

    return mask_func
