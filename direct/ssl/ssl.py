# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Direct SSL mask splitters module."""

import contextlib
from abc import abstractmethod
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from direct.data.transforms import apply_mask
from direct.ssl.mask_fillers import gaussian_fill, uniform_fill
from direct.types import DirectEnum, KspaceKey
from direct.utils import DirectModule

__all__ = [
    "GaussianMaskSplitterModule",
    "HalfMaskSplitterModule",
    "HalfSplitType",
    "MaskSplitterType",
    "UniformMaskSplitterModule",
]


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class MaskSplitterType(DirectEnum):
    uniform = "uniform"
    gaussian = "gaussian"
    half = "half"


class HalfSplitType(DirectEnum):
    horizontal = "horizontal"
    vertical = "vertical"
    diagonal_left = "diagonal_left"
    diagonal_right = "diagonal_right"


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
        ratio: Union[float, List[float], Tuple[float, ...]] = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        kspace_key: KspaceKey = KspaceKey.masked_kspace,
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

    def _choose_ratio(self):
        choice = self.rng.randint(0, len(self.ratio))
        return self.ratio[choice]

    def _gaussian_split(
        self,
        mask: torch.Tensor,
        std_scale: float = 3.0,
        seed: Union[Tuple[int, ...], List[int], int, None] = None,
        acs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        (theta_mask, lambda_mask): Tuple(torch.Tensor, torch.Tensor)
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

        lambda_mask = torch.zeros_like(mask, dtype=mask.dtype, device=mask.device)

        lambda_mask = torch.tensor(
            gaussian_fill(
                nonzero_mask_count,
                nrow,
                ncol,
                center_x,
                center_y,
                std_scale,
                temp_mask.cpu().numpy().astype(int),
                lambda_mask.cpu().numpy().astype(int),
                seed,
            ),
            dtype=mask.dtype,
        ).to(mask.device)
        theta_mask = mask & (~lambda_mask)

        if self.keep_acs:
            theta_mask, lambda_mask = theta_mask | acs_mask, lambda_mask | acs_mask

        return theta_mask, lambda_mask

    def _uniform_split(
        self,
        mask: torch.Tensor,
        seed: Union[Tuple[int, ...], List[int], int, None] = None,
        acs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        (theta_mask, lambda_mask): Tuple(torch.Tensor, torch.Tensor)
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
            lambda_mask = uniform_fill(
                int(torch.count_nonzero(temp_mask.flatten()) * self._choose_ratio()),
                nrow,
                ncol,
                temp_mask.cpu(),
                self.rng,
            ).to(mask.device)

        theta_mask = mask & (~lambda_mask)

        if self.keep_acs:
            theta_mask, lambda_mask = theta_mask | acs_mask, lambda_mask | acs_mask

        return theta_mask, lambda_mask

    def _half_split(self, mask: torch.Tensor, direction: HalfSplitType, acs_mask: Optional[torch.Tensor] = None):
        nrow, ncol = mask.shape

        center_x = nrow // 2
        center_y = ncol // 2

        if direction in ["horizontal", "vertical"]:
            theta_mask, lambda_mask = [torch.zeros_like(mask, dtype=mask.dtype, device=mask.device) for _ in range(2)]
            if direction == "horizontal":
                theta_mask[:center_x] = mask[:center_x]
                lambda_mask[center_x:] = mask[center_x:]
            else:
                theta_mask[:, :center_y] = mask[:, :center_y]
                lambda_mask[:, center_y:] = mask[:, center_y:]
        else:
            x = torch.linspace(-1, 1, nrow)
            y = torch.linspace(-1, 1, ncol)
            xv, yv = torch.meshgrid(x, y, indexing="ij")
            if direction == "diagonal_right":
                theta_mask = mask * (xv + yv <= 0)
                lambda_mask = mask * (xv + yv > 0)
            else:
                theta_mask = mask * (xv - yv <= 0)
                lambda_mask = mask * (xv - yv > 0)
        if self.keep_acs:
            theta_mask, lambda_mask = theta_mask | acs_mask, lambda_mask | acs_mask

        return theta_mask, lambda_mask

    @staticmethod
    def _unsqueeze_mask(masks: Iterable[torch.Tensor]) -> List[torch.Tensor]:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, theta_mask and lambda_mask.
        """

        raise NotImplementedError(f"Must be implemented by inheriting class.")

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Splits the mask tensor in the sample into two disjoint masks and applied them to the k-space.

        Parameters
        ----------
        sample: Dict[str, Any]
            The input sample.

        Returns
        -------
        Dict[str, Any]:
            The updated sample with the two disjoint masks, theta_mask and lambda_mask,
            as well as the two disjoint k-spaces.
        """
        sampling_mask = sample["sampling_mask"].clone()
        kspace = sample[self.kspace_key].clone()
        acs_mask = sample["acs_mask"].clone() if self.keep_acs else None

        theta_mask, lambda_mask = zip(
            *[
                self._unsqueeze_mask(
                    self.split_method(
                        sampling_mask[_],
                        acs_mask[_] if self.keep_acs else None,
                        None
                        if not self.use_seed
                        else tuple(map(ord, str(sample["filename"][_]) + str(sample["slice_no"][_]))),
                    )
                )
                for _ in range(kspace.shape[0])
            ]
        )
        theta_mask, lambda_mask = torch.stack(theta_mask, dim=0), torch.stack(lambda_mask, dim=0)

        del sampling_mask, acs_mask

        sample["theta_" + self.kspace_key], _ = apply_mask(kspace, theta_mask)
        sample["lambda_" + self.kspace_key], _ = apply_mask(kspace, lambda_mask)

        sample["theta_sampling_mask"] = theta_mask
        sample["lambda_sampling_mask"] = lambda_mask

        return sample


class UniformMaskSplitterModule(MaskSplitter):
    """Uses Uniform splitting method to split the input mask into two disjoint masks."""

    def __init__(
        self,
        ratio: Union[float, List[float], Tuple[float, ...]] = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        kspace_key: KspaceKey = KspaceKey.masked_kspace,
    ):
        """Inits :class:`UniformMaskSplitterModule`.

        Parameters
        ----------
        ratio: float, List[float] or Tuple[float, ...], optional
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}`. Default: 0.5.
        acs_region: List[int] or Tuple[int, int], optional
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
            split_type=MaskSplitterType.uniform,
            ratio=ratio,
            acs_region=acs_region,
            keep_acs=keep_acs,
            use_seed=use_seed,
            kspace_key=kspace_key,
        )

    def split_method(
        self, sampling_mask: torch.Tensor, acs_mask: Union[torch.Tensor, None], seed: Union[int, Iterable[int], None]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, theta_mask and lambda_mask.
        """
        theta_mask, lambda_mask = self._uniform_split(
            mask=sampling_mask.squeeze(),
            seed=seed,
            acs_mask=acs_mask.squeeze() if self.keep_acs else None,
        )
        return theta_mask, lambda_mask


class GaussianMaskSplitterModule(MaskSplitter):
    """Uses Gaussian splitting method to split the input mask into two disjoint masks."""

    def __init__(
        self,
        ratio: Union[float, List[float], Tuple[float, ...]] = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        kspace_key: KspaceKey = KspaceKey.masked_kspace,
        std_scale: float = 3.0,
    ):
        """Inits :class:`GaussianMaskSplitterModule`.

        Parameters
        ----------
        ratio: float, List[float] or Tuple[float, ...], optional
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}`. Default: 0.5.
        acs_region: List[int] or Tuple[int, int], optional
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
            split_type=MaskSplitterType.gaussian,
            ratio=ratio,
            acs_region=acs_region,
            keep_acs=keep_acs,
            use_seed=use_seed,
            kspace_key=kspace_key,
        )
        self.std_scale = std_scale

    def split_method(
        self, sampling_mask: torch.Tensor, acs_mask: Union[torch.Tensor, None], seed: Union[int, Iterable[int], None]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, theta_mask and lambda_mask.
        """

        theta_mask, lambda_mask = self._gaussian_split(
            mask=sampling_mask.squeeze(),
            seed=seed,
            std_scale=self.std_scale,
            acs_mask=acs_mask.squeeze() if self.keep_acs else None,
        )
        return theta_mask, lambda_mask


class HalfMaskSplitterModule(MaskSplitter):
    """Splits the input mask into two disjoint masks in a half line direction."""

    def __init__(
        self,
        acs_region: Union[List[int], Tuple[int, int]] = (0, 0),
        keep_acs: bool = False,
        use_seed: bool = True,
        direction: HalfSplitType = HalfSplitType.vertical,
        kspace_key: KspaceKey = KspaceKey.masked_kspace,
    ):
        """Inits :class:`GaussianMaskSplitterModule`.

        Parameters
        ----------
        acs_region: List[int] or Tuple[int, int], optional
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
            split_type=MaskSplitterType.half,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]:
            The two disjoint masks, theta_mask and lambda_mask.
        """

        theta_mask, lambda_mask = self._half_split(
            mask=sampling_mask.squeeze(),
            direction=self.direction,
            acs_mask=acs_mask.squeeze() if self.keep_acs else None,
        )
        return theta_mask, lambda_mask
