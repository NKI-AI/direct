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
logger = logging.getLogger(__name__)


class BaseMaskFunc:
    """
    BaseMaskFunc is the base class to create a sub-sampling mask of a given shape.
    """

    def __init__(self, accelerations: Tuple[Number, ...],
                 center_fractions: Optional[Tuple[float, ...]] = None,
                 uniform_range: bool = True, return_parameters: bool = False):
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
        return_parameters : bool
            If True, then the accelerations and center fractions will be returned.
        """
        if center_fractions is not None:
            if len(center_fractions) != len(accelerations):
                raise ValueError(f'Number of center fractions should match number of accelerations. '
                                 f'Got {len(center_fractions)} {len(accelerations)}.')

        self.center_fractions = center_fractions
        self.accelerations = accelerations

        self.uniform_range = uniform_range

        self.rng = np.random.RandomState()
        self.return_parameters = return_parameters

    def choose_acceleration(self):
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
            raise NotImplementedError('Uniform range is not yet implemented.')

    @abstractmethod
    def mask_func(self, shape):
        raise NotImplementedError(f'This method should be implemented by a child class.')

    def __call__(self, shape, seed=None):
        """
        Parameters
        ----------
        shape : iterable[int]
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.

        Returns
        -------
        ndarray
        """
        self.rng.seed(seed)
        mask = self.mask_func(shape)
        return mask


class FastMRIMaskFunc(BaseMaskFunc):
    def __init__(self, accelerations: Tuple[Number, ...], center_fractions: Optional[Tuple[float, ...]] = None,
                 uniform_range: bool = False):
        super().__init__(accelerations=accelerations, center_fractions=center_fractions, uniform_range=uniform_range)

    def mask_func(self, shape):
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
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        num_rows = shape[-3]
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = mask.reshape(*mask_shape).astype(np.int32)
        mask_shape[-3] = num_rows

        mask = np.broadcast_to(mask, mask_shape)[np.newaxis, ...].copy()  # Add coil axis, make array writable.

        if self.return_parameters:
            return torch.from_numpy(mask), acceleration, center_fraction

        return torch.from_numpy(mask)


def build_masking_functions(
        name,
        center_fractions,
        accelerations,
        uniform_range=False,
        val_center_fractions=None,
        val_accelerations=None):

    MaskFunc: BaseMaskFunc = str_to_class('direct.common.subsample', name + 'MaskFunc') # noqa

    train_mask_func = MaskFunc(accelerations, center_fractions, uniform_range=uniform_range)

    val_center_fractions = center_fractions if not val_center_fractions else val_center_fractions
    val_accelerations = accelerations if not val_accelerations else val_accelerations
    val_mask_func = MaskFunc(val_accelerations, val_center_fractions, uniform_range=False)

    return train_mask_func, val_mask_func
