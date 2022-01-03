# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

import contextlib
import logging
from abc import abstractmethod
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch

import direct.data.transforms as T
from direct.environment import DIRECT_CACHE_DIR
from direct.types import Number
from direct.utils import str_to_class
from direct.utils.io import download_url

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
            if len([center_fractions]) != len([accelerations]):
                raise ValueError(
                    f"Number of center fractions should match number of accelerations. "
                    f"Got {len([center_fractions])} {len([accelerations])}."
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
        raise NotImplementedError("Uniform range is not yet implemented.")

    @abstractmethod
    def mask_func(self, shape):
        raise NotImplementedError("This method should be implemented by a child class.")

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
        mask = self.mask_func(data, seed=seed, return_acs=return_acs)  # pylint: disable = E1123
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
        Creates vertical line mask.

        The mask selects a subset of columns from the input k-space data. If the k-space data has N
        columns, the mask picks out:

            #.  N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.
            #.  The other columns are selected uniformly at random with a probability equal to: prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs). This ensures that the expected number of columns selected is equal to (N / acceleration)

        It is possible to use multiple center_fractions and accelerations, in which case one possible
        (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
        called.

        For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
        is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
        probability that 8-fold acceleration with 4% center fraction is selected.

        Parameters
        ----------
        shape : iterable[int]
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.
        return_acs : bool
            Return the autocalibration signal region as a mask.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask.

        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_rows = shape[-3]
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # Create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # Reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.int32)
            mask_shape[-3] = num_rows

            mask = np.broadcast_to(mask, mask_shape)[np.newaxis, ...].copy()  # Add coil axis, make array writable.

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
        Creates equispaced vertical line mask.

        FastMRIEquispacedMaskFunc creates a sub-sampling mask of a given shape. The mask selects a subset of columns
        from the input k-space data. If the k-space data has N columns, the mask picks out:

            #.  N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.
            #.  The other columns are selected with equal spacing at a proportion that reaches the desired acceleration rate taking into consideration the number of low frequencies. This ensures that the expected number of columns selected is equal to (N / acceleration).

        It is possible to use multiple center_fractions and accelerations, in which case one possible
        (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc object is called.

        Note that this function may not give equispaced samples (documented in https://github.com/facebookresearch/fastMRI/issues/54),
        which will require modifications to standard GRAPPA approaches. Nonetheless, this aspect of the function has
        been preserved to match the public multicoil data.

        Parameters
        ----------
        shape : iterable[int]
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.
        return_acs : bool
            Return the autocalibration signal region as a mask.

        Returns
        -------
        mask : torch.Tensor
            The sampling mask.

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
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # Reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.int32)
            mask_shape[-3] = num_rows

            mask = np.broadcast_to(mask, mask_shape)[np.newaxis, ...].copy()  # Add coil axis, make array writable.

            if return_acs:
                acs_mask = np.zeros_like(mask)
                acs_mask[:, :, pad : pad + num_low_freqs, ...] = 1
                return torch.from_numpy(acs_mask)

        return torch.from_numpy(mask)


class CalgaryCampinasMaskFunc(BaseMaskFunc):
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
    def __init__(self, accelerations: Tuple[int, ...], **kwargs):  # noqa
        super().__init__(accelerations=accelerations, uniform_range=False)

        if not all(_ in [5, 10] for _ in accelerations):
            raise ValueError("CalgaryCampinas only provide 5x and 10x acceleration masks.")

        self.masks = {}
        self.shapes: List[Number] = []

        for acceleration in accelerations:
            self.masks[acceleration] = self.__load_masks(acceleration)

    @staticmethod
    def circular_centered_mask(shape, radius):
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
        return mask[np.newaxis, ..., np.newaxis]

    def mask_func(self, shape, return_acs=False, seed=None):
        """
        Downloads and loads pre=computed Poisson masks.
        Currently supports shapes of 218 x 170/ 218/ 174 and acceleration factors of 5 or 10.

        Parameters
        ----------

        shape : iterable[int]
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        seed : int (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape.
        return_acs : bool
            Return the autocalibration signal region as a mask.

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

    def __load_masks(self, acceleration):
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


class CIRCUSSamplingMode(str, Enum):

    circus_radial = "circus-radial"
    circus_spiral = "circus-spiral"


class CIRCUSMaskFunc(BaseMaskFunc):
    """
    Implementation of Cartesian undersampling (radial or spiral) using CIRCUS as shown in [1]_. It creates
    radial or spiral masks for Cartesian acquired data on a grid.

    References
    ----------

    .. [1] Liu J, Saloner D. Accelerated MRI with CIRcular Cartesian UnderSampling (CIRCUS): a variable density Cartesian sampling strategy for compressed sensing and parallel imaging. Quant Imaging Med Surg. 2014 Feb;4(1):57-67. doi: 10.3978/j.issn.2223-4292.2014.02.01. PMID: 24649436; PMCID: PMC3947985.

    """

    def __init__(
        self,
        accelerations,
        subsampling_scheme: CIRCUSSamplingMode,
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=tuple(0 for _ in range(len(accelerations))),
            uniform_range=False,
        )
        if subsampling_scheme not in ["circus-spiral", "circus-radial"]:
            raise NotImplementedError(
                f"Currently CIRCUSMaskFunc is only implemented for 'circus-radial' or 'circus-spiral' "
                f"as a subsampling_scheme. Got subsampling_scheme={subsampling_scheme}."
            )

        self.subsampling_scheme = "circus-radial" if subsampling_scheme is None else subsampling_scheme

    @staticmethod
    def get_square_ordered_idxs(square_side_size: int, square_id: int) -> Tuple[Tuple, ...]:
        """
        Returns ordered (clockwise) indices of a sub-square of a square matrix.

        Parameters
        ----------
        square_side_size: int
            Square side size. Dim of array.
        square_id: int
            Number of sub-square. Can be 0, ..., square_side_size // 2.

        Returns
        -------
        ordered_idxs: List of tuples.
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

    def circus_radial_mask(self, shape, acceleration):
        """
        Implements CIRCUS radial undersampling.
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

    def circus_spiral_mask(self, shape, acceleration):
        """
        Implements CIRCUS spiral undersampling.
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
                indices_idx = int(np.mod((i + np.ceil(J ** c) - 1), K))

                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad)
        mask = T.center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    @staticmethod
    def circular_centered_mask(mask, eps=0.1):
        shape = mask.shape
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        Y, X = torch.tensor(Y), torch.tensor(X)
        radius = 1

        # Finds the maximum (unmasked) disk in mask given a tolerance.
        while True:
            # Creates a disk with R=radius and finds intersection with mask
            disk = (Y - center[0]) ** 2 + (X - center[1]) ** 2 <= radius ** 2
            intersection = disk & mask
            ratio = disk.sum() / intersection.sum()
            if ratio > 1.0 + eps:
                return intersection
            radius += eps

    def mask_func(self, shape, return_acs=False, seed=None):

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_rows = shape[-3]
            num_cols = shape[-2]
            acceleration = self.choose_acceleration()[1]

            if self.subsampling_scheme == "circus-radial":
                mask = self.circus_radial_mask(
                    shape=(num_rows, num_cols),
                    acceleration=acceleration,
                )
            elif self.subsampling_scheme == "circus-spiral":
                mask = self.circus_spiral_mask(
                    shape=(num_rows, num_cols),
                    acceleration=acceleration,
                )

            if return_acs:
                return self.circular_centered_mask(mask).unsqueeze(0).unsqueeze(-1)

            return mask.unsqueeze(0).unsqueeze(-1)


class RadialMaskFunc(CIRCUSMaskFunc):
    """
    Computes radial masks for Cartesian data.

    """

    def __init__(
        self,
        accelerations,
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            subsampling_scheme=CIRCUSSamplingMode.circus_radial,
            **kwargs,
        )


class SpiralMaskFunc(CIRCUSMaskFunc):
    """
    Computes spiral masks for Cartesian data.

    """

    def __init__(
        self,
        accelerations,
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            subsampling_scheme=CIRCUSSamplingMode.circus_spiral,
            **kwargs,
        )


class DictionaryMaskFunc(BaseMaskFunc):
    def __init__(self, data_dictionary, **kwargs):  # noqa
        super().__init__(accelerations=None)

        self.data_dictionary = data_dictionary

    def mask_func(self, data, return_acs=False):
        return self.data_dictionary[data]


def build_masking_function(name, accelerations, center_fractions=None, uniform_range=False, **kwargs):
    MaskFunc: BaseMaskFunc = str_to_class("direct.common.subsample", name + "MaskFunc")  # noqa
    mask_func = MaskFunc(
        accelerations=accelerations,
        center_fractions=center_fractions,
        uniform_range=uniform_range,
    )

    return mask_func
