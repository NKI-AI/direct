# Copyright (c) DIRECT Contributors

"""Direct SSL mask splitters module. 

This module contains classes for splitting masks for self-supervised learning tasks for MRI reconstruction.
For example, the :class:`GaussianMaskSplitterModule` splits the input mask into two disjoint masks using a Gaussian 
split scheme, while the :class:`UniformMaskSplitterModule` splits the input mask into two disjoint masks using a 
uniform split scheme. The :class:`HalfMaskSplitterModule` splits the input mask into two disjoint masks in a half 
line direction.
"""

from __future__ import annotations

import contextlib
from abc import abstractmethod
from math import ceil
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch

from direct.data.transforms import apply_mask
from direct.ssl.mask_fillers import gaussian_fill, uniform_fill
from direct.types import DirectEnum, KspaceKey, TransformKey
from direct.utils import DirectModule

__all__ = [
    "GaussianMaskSplitterModule",
    "HalfMaskSplitterModule",
    "HalfSplitType",
    "MaskSplitterType",
    "UniformMaskSplitterModule",
    "SSLTransformMaskPrefixes",
]


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class SSLTransformMaskPrefixes(DirectEnum):
    """SSL Transform mask prefixes.

    These are used to prefix the input and target masks in the sample.
    """

    INPUT_ = "input_"
    TARGET_ = "target_"


class MaskSplitterType(DirectEnum):
    """SSL mask splitter types.

    These are used to define the type of mask splitting.
    """

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    HALF = "half"


class HalfSplitType(DirectEnum):
    """SSL half mask splitter types.

    These are used to define the type of half mask splitting.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL_LEFT = "diagonal_left"
    DIAGONAL_RIGHT = "diagonal_right"


class MaskSplitter(DirectModule):
    r"""Splits input mask :math:`\Omega` into two disjoint masks :math:`\Theta`, :math:`\Lambda` such that

    .. math::
        \Omega = \Theta \cup \Lambda, \Theta = \Omega \backslash \Lambda.

    Inspired and adapted from code implementation of _[1], _[2].

    References
    ----------

    .. [1] Yaman, Burhaneddin, et al. “Self‐supervised Learning of Physics‐guided Reconstruction Neural Networks
        without Fully Sampled Reference Data.” Magnetic Resonance in Medicine, vol. 84, no. 6, Dec. 2020,
        pp. 3172–91. DOI.org (Crossref), https://doi.org/10.1002/mrm.28378.
    .. [2] Yaman, Burhaneddin, et al. “Self-Supervised Physics-Based Deep Learning MRI Reconstruction Without
        Fully-Sampled Data.” 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), 2020,
        pp. 921–25. IEEE Xplore, https://doi.org/10.1109/ISBI45749.2020.9098514.
    """

    def __init__(
        self,
        split_type: MaskSplitterType,
        ratio: Union[float, list[float], tuple[float, ...]] = 0.5,
        acs_region: Union[list[int], tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        kspace_key: KspaceKey = KspaceKey.MASKED_KSPACE,
    ):
        r"""Inits :class:`MaskSplitter`.

        Parameters
        ----------
        split_type: MaskSplitterType
            Type of mask splitting. Can be `gaussian` or `uniform`.
        ratio: list of tuple of floats
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}. Default: 0.5.
        acs_region: list or tuple of ints
            Size of ACS region to include in training (input) mask. Default: (0, 0).
        keep_acs: bool
            If True, both input and target masks will keep the acs region and ratio will be applied on the rest of
            the mask. Assumes `acs_mask` is present in the sample.
        use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        kspace_key: str
            K-space key. Default "masked_kspace".
        """
        super().__init__()
        assert split_type in ["gaussian", "uniform"]
        self.split_type = split_type
        if isinstance(ratio, float):
            ratio = [ratio]
        if not all([0 < r < 1 for r in ratio]):
            raise ValueError(f"Ratios should be floats between 0 and 1. Received: {ratio}.")

        self.ratio = ratio
        self.acs_region = acs_region
        self.keep_acs = keep_acs

        self.kspace_key = kspace_key

        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def _choose_ratio(self) -> float:
        """Chooses a ratio from the list of ratios randomly.

        Returns
        -------
        float
            The chosen ratio.
        """
        choice = self.rng.randint(0, len(self.ratio))
        return self.ratio[choice]

    def _gaussian_split(
        self,
        mask: torch.Tensor,
        std_scale: float = 3.0,
        seed: Union[tuple[int, ...], list[int], int, None] = None,
        acs_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits `mask` into an input and target disjoint masks using a bivariate Gaussian sampling.

        Parameters
        ----------
        mask: torch.Tensor
            Masking tensor to split.
        std_scale: float = 3.0
            This is used to calculate the standard deviation of the Gaussian distribution. Default: 3.0.
        seed: int, list or tuple of ints or None
            Default: None.
        acs_mask: torch.Tensor, optional
            ACS mask. Needs to be passed if `keep_acs` is True. If `keep_acs` is False but this is passed, it will be
            ignored. Default: None.

        Returns
        -------
        (input_mask, target_mask): Tuple(torch.Tensor, torch.Tensor)
            Two (disjoint) masks using a uniform split scheme from the input mask. For SSDU these will be used as
            input and target masks.
        """
        nrow, ncol = mask.shape

        center_x = nrow // 2
        center_y = ncol // 2

        if self.keep_acs and acs_mask is None:
            raise ValueError(f"`keep_acs` is set to True but not received an input for `acs_mask`.")
        mask = mask.clone() if not self.keep_acs else mask.clone() & (~acs_mask)

        with temp_seed(self.rng, seed):
            if seed is None:
                seed = np.random.randint(0, 1e5)
            elif isinstance(seed, (tuple, list)):
                seed = int(np.mean(seed))
            elif isinstance(seed, int):
                seed = seed

            nonzero_mask_count = int(ceil(mask.sum() * self._choose_ratio()))

        temp_mask = mask.clone()
        if not self.keep_acs:
            temp_mask[
                center_x - self.acs_region[0] // 2 : center_x + self.acs_region[0] // 2,
                center_y - self.acs_region[1] // 2 : center_y + self.acs_region[1] // 2,
            ] = False

        target_mask = torch.zeros_like(mask, dtype=mask.dtype, device=mask.device)

        target_mask = torch.tensor(
            gaussian_fill(
                nonzero_mask_count,
                nrow,
                ncol,
                center_x,
                center_y,
                std_scale,
                temp_mask.cpu().numpy().astype(int),
                target_mask.cpu().numpy().astype(int),
                seed,
            ),
            dtype=mask.dtype,
        ).to(mask.device)
        input_mask = mask & (~target_mask)

        if self.keep_acs:
            input_mask, target_mask = input_mask | acs_mask, target_mask | acs_mask

        return input_mask, target_mask

    def _uniform_split(
        self,
        mask: torch.Tensor,
        seed: Union[tuple[int, ...], list[int], int, None] = None,
        acs_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits `mask` into an input and target disjoint masks using a uniform sampling.

        Parameters
        ----------
        mask: torch.Tensor
            Masking tensor to split.
        seed: int, list or tuple of ints or None
            Default: None.
        acs_mask: torch.Tensor, optional
            ACS mask. Needs to be passed if `keep_acs` is True. If `keep_acs` is False but this is passed, it will be
            ignored. Default: None.

        Returns
        -------
        (input_mask, target_mask): Tuple(torch.Tensor, torch.Tensor)
            Two (disjoint) masks using a gaussian split scheme from the input mask. For SSDU these will be used as
            input and target masks.
        """
        nrow, ncol = mask.shape

        center_x = nrow // 2
        center_y = ncol // 2

        if self.keep_acs and acs_mask is None:
            raise ValueError(f"`keep_acs` is set to True but not received an input for `acs_mask`.")
        mask = mask.clone() if not self.keep_acs else mask.clone() & (~acs_mask)
        temp_mask = mask.cpu().clone()

        if not self.keep_acs:
            temp_mask[
                center_x - self.acs_region[0] // 2 : center_x + self.acs_region[0] // 2,
                center_y - self.acs_region[1] // 2 : center_y + self.acs_region[1] // 2,
            ] = False

        with temp_seed(self.rng, seed):
            target_mask = uniform_fill(
                int(torch.count_nonzero(temp_mask.flatten()) * self._choose_ratio()),
                nrow,
                ncol,
                temp_mask.cpu(),
                self.rng,
            ).to(mask.device)

        input_mask = mask & (~target_mask)

        if self.keep_acs:
            input_mask, target_mask = input_mask | acs_mask, target_mask | acs_mask

        return input_mask, target_mask

    def _half_split(self, mask: torch.Tensor, direction: HalfSplitType, acs_mask: Optional[torch.Tensor] = None):
        nrow, ncol = mask.shape

        center_x = nrow // 2
        center_y = ncol // 2

        if direction in ["horizontal", "vertical"]:
            input_mask, target_mask = [torch.zeros_like(mask, dtype=mask.dtype, device=mask.device) for _ in range(2)]
            if direction == "horizontal":
                input_mask[:center_x] = mask[:center_x]
                target_mask[center_x:] = mask[center_x:]
            else:
                input_mask[:, :center_y] = mask[:, :center_y]
                target_mask[:, center_y:] = mask[:, center_y:]
        else:
            x = torch.linspace(-1, 1, nrow)
            y = torch.linspace(-1, 1, ncol)
            xv, yv = torch.meshgrid(x, y, indexing="ij")
            if direction == "diagonal_right":
                input_mask = mask * (xv + yv <= 0)
                target_mask = mask * (xv + yv > 0)
            else:
                input_mask = mask * (xv - yv <= 0)
                target_mask = mask * (xv - yv > 0)
        if self.keep_acs:
            input_mask, target_mask = input_mask | acs_mask, target_mask | acs_mask

        return input_mask, target_mask

    @staticmethod
    def _unsqueeze_mask(masks: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        """Unsqueeze coil and complex dimensions of mask tensors.

        Parameters
        ----------
        masks : iterable of torch.Tensor
            Input masks.

        Returns
        -------
        list of torch.Tensor
            Listr of tensors with unsqueezed dimensions.
        """
        return [mask[None, ..., None] for mask in masks]

    @abstractmethod
    def split_method(
        self, sampling_mask: torch.Tensor, acs_mask: Union[torch.Tensor, None], seed: Union[int, Iterable[int], None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the `sampling_mask` into two disjoint masks based on the class' split method.

        Parameters
        ----------
        sampling_mask : torch.Tensor
            The input mask tensor to be split.
        acs_mask : torch.Tensor or None
            The ACS mask. Needs to be passed if `keep_acs` is True. If `keep_acs` is False but this is passed, it will be
            ignored. Default: None.
        seed : int, iterable of ints or None
            Seed to generate split.


        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, input_mask and target_mask.
        """

        raise NotImplementedError(f"Must be implemented by inheriting class.")

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Splits the mask tensor in the sample into two disjoint masks and applied them to the k-space.

        Parameters
        ----------
        sample: dict[str, Any]
            The input sample.

        Returns
        -------
        dict[str, Any]:
            The updated sample with the two disjoint masks, input_mask and target_mask,
            as well as the two disjoint k-spaces.
        """
        sampling_mask = sample["sampling_mask"].clone()
        kspace = sample[self.kspace_key].clone()
        acs_mask = sample["acs_mask"].clone() if self.keep_acs else None

        input_mask, target_mask = zip(
            *[
                self._unsqueeze_mask(
                    self.split_method(
                        sampling_mask[_],
                        acs_mask[_] if self.keep_acs else None,
                        (
                            None
                            if not self.use_seed
                            else tuple(map(ord, str(sample["filename"][_]) + str(sample["slice_no"][_])))
                        ),
                    )
                )
                for _ in range(kspace.shape[0])
            ]
        )
        input_mask, target_mask = torch.stack(input_mask, dim=0), torch.stack(target_mask, dim=0)

        del sampling_mask, acs_mask

        sample[SSLTransformMaskPrefixes.INPUT_ + self.kspace_key], _ = apply_mask(kspace, input_mask)
        sample[SSLTransformMaskPrefixes.TARGET_ + self.kspace_key], _ = apply_mask(kspace, target_mask)

        sample[SSLTransformMaskPrefixes.INPUT_ + TransformKey.SAMPLING_MASK] = input_mask
        sample[SSLTransformMaskPrefixes.TARGET_ + TransformKey.SAMPLING_MASK] = target_mask

        return sample


class UniformMaskSplitterModule(MaskSplitter):
    """Uses Uniform splitting method to split the input mask into two disjoint masks."""

    def __init__(
        self,
        ratio: Union[float, list[float], tuple[float, ...]] = 0.5,
        acs_region: Union[list[int], tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        kspace_key: KspaceKey = KspaceKey.MASKED_KSPACE,
    ):
        """Inits :class:`UniformMaskSplitterModule`.

        Parameters
        ----------
        ratio: float, list[float] or tuple[float, ...], optional
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}`. Default: 0.5.
        acs_region: list[int] or tuple[int, int], optional
            Size of ACS region to include in training (input) mask. Default: (0, 0).
        keep_acs: bool, optional
            If True, both input and target masks will keep the acs region and ratio will be applied on the rest of the mask.
            Assumes `acs_mask` is present in the sample. Default: False.
        use_seed: bool, optional
            If True, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        kspace_key: str, optional
            K-space key. Default "masked_kspace".
        """
        super().__init__(
            split_type=MaskSplitterType.UNIFORM,
            ratio=ratio,
            acs_region=acs_region,
            keep_acs=keep_acs,
            use_seed=use_seed,
            kspace_key=kspace_key,
        )

    def split_method(
        self, sampling_mask: torch.Tensor, acs_mask: Union[torch.Tensor, None], seed: Union[int, Iterable[int], None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the `sampling_mask` into two disjoint masks based on the uniform split method.

        Parameters
        ----------
        sampling_mask : torch.Tensor
            The input mask tensor to be split.
        acs_mask : torch.Tensor or None
            The ACS mask. Needs to be passed if `keep_acs` is True. If `keep_acs` is False but this is passed, it will be
            ignored. Default: None.
        seed : int, iterable of ints or None
            Seed to generate split.


        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, input_mask and target_mask.
        """
        input_mask, target_mask = self._uniform_split(
            mask=sampling_mask.squeeze(),
            seed=seed,
            acs_mask=acs_mask.squeeze() if self.keep_acs else None,
        )
        return input_mask, target_mask


class GaussianMaskSplitterModule(MaskSplitter):
    """Uses Gaussian splitting method to split the input mask into two disjoint masks."""

    def __init__(
        self,
        ratio: Union[float, list[float], tuple[float, ...]] = 0.5,
        acs_region: Union[list[int], tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        kspace_key: KspaceKey = KspaceKey.MASKED_KSPACE,
        std_scale: float = 3.0,
    ):
        """Inits :class:`GaussianMaskSplitterModule`.

        Parameters
        ----------
        ratio: float, list[float] or tuple[float, ...], optional
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}`. Default: 0.5.
        acs_region: list[int] or tuple[int, int], optional
            Size of ACS region to include in training (input) mask. Default: (0, 0).
        keep_acs: bool, optional
            If True, both input and target masks will keep the acs region and ratio will be applied on the rest of the mask.
            Assumes `acs_mask` is present in the sample. Default: False.
        use_seed: bool, optional
            If True, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        kspace_key: str, optional
            K-space key. Default "masked_kspace".
        std_scale: float, optional
            This is used to calculate the standard deviation of the Gaussian distribution. Default: 3.0.
        """
        super().__init__(
            split_type=MaskSplitterType.GAUSSIAN,
            ratio=ratio,
            acs_region=acs_region,
            keep_acs=keep_acs,
            use_seed=use_seed,
            kspace_key=kspace_key,
        )
        self.std_scale = std_scale

    def split_method(
        self, sampling_mask: torch.Tensor, acs_mask: Union[torch.Tensor, None], seed: Union[int, Iterable[int], None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the `sampling_mask` into two disjoint masks based on gaussian split method.

        Parameters
        ----------
        sampling_mask : torch.Tensor
            The input mask tensor to be split.
        acs_mask : torch.Tensor or None
            The ACS mask. Needs to be passed if `keep_acs` is True. If `keep_acs` is False but this is passed, it will be
            ignored. Default: None.
        seed : int, iterable of ints or None
            Seed to generate split.


        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, input_mask and target_mask.
        """

        input_mask, target_mask = self._gaussian_split(
            mask=sampling_mask.squeeze(),
            seed=seed,
            std_scale=self.std_scale,
            acs_mask=acs_mask.squeeze() if self.keep_acs else None,
        )
        return input_mask, target_mask


class HalfMaskSplitterModule(MaskSplitter):
    """Splits the input mask into two disjoint masks in a half line direction."""

    def __init__(
        self,
        acs_region: Union[list[int], tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        direction: HalfSplitType = HalfSplitType.VERTICAL,
        kspace_key: KspaceKey = KspaceKey.MASKED_KSPACE,
    ):
        """Inits :class:`GaussianMaskSplitterModule`.

        Parameters
        ----------
        acs_region: list[int] or tuple[int, int], optional
            Size of ACS region to include in training (input) mask. Default: (0, 0).
        keep_acs: bool, optional
            If True, both input and target masks will keep the acs region and ratio will be applied on the rest of the mask.
            Assumes `acs_mask` is present in the sample. Default: False.
        use_seed: bool, optional
            If True, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        kspace_key: str, optional
            K-space key. Default "masked_kspace".
        """
        super().__init__(
            split_type=MaskSplitterType.HALF,
            ratio=[0.5],
            acs_region=acs_region,
            keep_acs=keep_acs,
            use_seed=use_seed,
            kspace_key=kspace_key,
        )
        self.direction = direction

    def split_method(
        self,
        sampling_mask: torch.Tensor,
        acs_mask: Union[torch.Tensor, None],
        seed: Union[int, Iterable[int], None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the `sampling_mask` into two disjoint masks based on gaussian split method.

        Parameters
        ----------
        sampling_mask : torch.Tensor
            The input mask tensor to be split.
        acs_mask : torch.Tensor or None
            The ACS mask. Needs to be passed if `keep_acs` is True. If `keep_acs` is False but this is passed, it will be
            ignored. Default: None.
        seed : int, iterable of ints or None
            Seed to generate split.


        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, input_mask and target_mask.
        """

        input_mask, target_mask = self._half_split(
            mask=sampling_mask.squeeze(),
            direction=self.direction,
            acs_mask=acs_mask.squeeze() if self.keep_acs else None,
        )
        return input_mask, target_mask
