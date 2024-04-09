# Copyright (c) DIRECT Contributors

"""direct.nn.functionals.hfen module."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

__all__ = ["hfen_l1", "hfen_l2", "HFENLoss", "HFENL1Loss", "HFENL2Loss"]


def _get_log_kernel2d(kernel_size: int | list[int] = 5, sigma: Optional[float | list[float]] = None) -> torch.Tensor:
    """Generates a 2D LoG (Laplacian of Gaussian) kernel.

    Parameters
    ----------
    kernel_size : int or list of ints
        Size of the kernel. Default: 5.
    sigma : float or list of floats
        Standard deviation(s) for the Gaussian distribution. Default: None.

    Returns
    -------
    torch.Tensor: Generated LoG kernel.
    """
    dim = 2
    if not kernel_size and sigma:
        kernel_size = np.ceil(sigma * 6)
        kernel_size = [kernel_size] * dim
    elif kernel_size and not sigma:
        sigma = kernel_size / 6.0
        sigma = [sigma] * dim

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size - 1] * dim
    if isinstance(sigma, float):
        sigma = [sigma] * dim

    grids = torch.meshgrid([torch.arange(-size // 2, size // 2 + 1, 1) for size in kernel_size], indexing="ij")

    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, grids):
        kernel *= torch.exp(-(mgrid**2 / (2.0 * std**2)))

    final_kernel = (
        kernel
        * ((grids[0] ** 2 + grids[1] ** 2) - (2 * sigma[0] * sigma[1]))
        * (1 / ((2 * math.pi) * (sigma[0] ** 2) * (sigma[1] ** 2)))
    )

    final_kernel = -final_kernel / torch.sum(final_kernel)

    return final_kernel


def _compute_padding(kernel_size: int | list[int] = 5) -> int | tuple[int, ...]:
    """Computes padding tuple based on the kernel size.

    For square kernels, pad can be an int, else, a tuple with an element for each dimension.

    Parameters
    ----------
    kernel_size : int or list of ints
        Size(s) of the kernel.

    Returns
    -------
    int or tuple of ints
        Computed padding.
    """
    if isinstance(kernel_size, int):
        return kernel_size // 2
    # Else assumed a list
    computed = [k // 2 for k in kernel_size]
    out_padding = []
    for i, _ in enumerate(kernel_size):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding.append(padding)
        out_padding.append(computed_tmp)
    return tuple(out_padding)


class HFENLoss(nn.Module):
    r"""High Frequency Error Norm (HFEN) Loss as defined in _[1].

    Calculates:

    .. math::

        || \text{LoG}(x_\text{rec}) - \text{LoG}(x_\text{tar}) ||_C

    Where C can be any norm, LoG is the Laplacian of Gaussian filter, and :math:`x_\text{rec}), \text{LoG}(x_\text{tar}`
    are the reconstructed inp and target images.
    If normalized it scales it by :math:`|| \text{LoG}(x_\text{tar}) ||_C`.

    Code was borrowed and adapted from _[2] (not licensed).

    References
    ----------
    .. [1] S. Ravishankar and Y. Bresler, "MR Image Reconstruction From Highly Undersampled k-Space Data by
        Dictionary Learning," in IEEE Transactions on Medical Imaging, vol. 30, no.
        5, pp. 1028-1041, May 2011, doi: 10.1109/TMI.2010.2090538.
    .. [2] https://github.com/styler00dollar/pytorch-loss-functions/blob/main/vic/loss.py
    """

    def __init__(
        self,
        criterion: nn.Module,
        reduction: str = "mean",
        kernel_size: int | list[int] = 5,
        sigma: float | list[float] = 2.5,
        norm: bool = False,
    ) -> None:
        """Inits :class:`HFENLoss`.

        Parameters
        ----------
        criterion : nn.Module
            Loss function to calculate the difference between log1 and log2.
        reduction : str
            Criterion reduction. Default: "mean".
        kernel_size : int or list of ints
            Size of the LoG  filter kernel. Default: 15.
        sigma : float or list of floats
            Standard deviation of the LoG filter kernel. Default: 2.5.
        norm : bool
            Whether to normalize the loss.
        """
        super().__init__()
        self.criterion = criterion(reduction=reduction)
        self.norm = norm
        kernel = _get_log_kernel2d(kernel_size, sigma)
        self.filter = self._compute_filter(kernel, kernel_size)

    @staticmethod
    def _compute_filter(kernel: torch.Tensor, kernel_size: int | list[int] = 15) -> nn.Module:
        """Computes the LoG filter based on the kernel and kernel size.

        Parameters
        ----------
        kernel : torch.Tensor
            The kernel tensor.
        kernel_size : int or list of ints, optional
            Size of the filter kernel. Default: 15.

        Returns
        -------
        nn.Module
            The computed filter.
        """
        kernel = kernel.expand(1, 1, *kernel.size()).contiguous()
        pad = _compute_padding(kernel_size)
        _filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        _filter.weight.data = kernel

        return _filter

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the :class:`HFENLoss`.

        Parameters
        ----------
        inp : torch.Tensor
            inp tensor.
        target : torch.Tensor
            Target tensor.

        Returns
        -------
        torch.Tensor
            HFEN loss value.
        """
        self.filter.to(inp.device)
        log1 = self.filter(inp)
        log2 = self.filter(target)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= self.criterion(torch.zeros_like(target, dtype=target.dtype, device=target.device), target)
        return hfen_loss


class HFENL1Loss(HFENLoss):
    r"""High Frequency Error Norm (HFEN) Loss using L1Loss criterion.

    Calculates:

    .. math::

        || \text{LoG}(x_\text{rec}) - \text{LoG}(x_\text{tar}) ||_1

    Where LoG is the Laplacian of Gaussian filter, and :math:`x_\text{rec}), \text{LoG}(x_\text{tar}`
    are the reconstructed inp and target images.
    If normalized it scales it by :math:`|| \text{LoG}(x_\text{tar}) ||_1`.
    """

    def __init__(
        self,
        reduction: str = "mean",
        kernel_size: int | list[int] = 15,
        sigma: float | list[float] = 2.5,
        norm: bool = False,
    ) -> None:
        """Inits :class:`HFENL1Loss`.

        Parameters
        ----------
        reduction : str
            Criterion reduction. Default: "mean".
        kernel_size : int or list of ints
            Size of the LoG  filter kernel. Default: 15.
        sigma : float or list of floats
            Standard deviation of the LoG filter kernel. Default: 2.5.
        norm : bool
            Whether to normalize the loss.
        """
        super().__init__(nn.L1Loss, reduction, kernel_size, sigma, norm)


class HFENL2Loss(HFENLoss):
    r"""High Frequency Error Norm (HFEN) Loss using L1Loss criterion.

    Calculates:

    .. math::

        || \text{LoG}(x_\text{rec}) - \text{LoG}(x_\text{tar}) ||_2

    Where LoG is the Laplacian of Gaussian filter, and :math:`x_\text{rec}), \text{LoG}(x_\text{tar}`
    are the reconstructed inp and target images.
    If normalized it scales it by :math:`|| \text{LoG}(x_\text{tar}) ||_2`.
    """

    def __init__(
        self,
        reduction: str = "mean",
        kernel_size: int | list[int] = 15,
        sigma: float | list[float] = 2.5,
        norm: bool = False,
    ) -> None:
        """Inits :class:`HFENL2Loss`.

        Parameters
        ----------
        reduction : str
            Criterion reduction. Default: "mean".
        kernel_size : int or list of ints
            Size of the LoG  filter kernel. Default: 15.
        sigma : float or list of floats
            Standard deviation of the LoG filter kernel. Default: 2.5.
        norm : bool
            Whether to normalize the loss.
        """
        super().__init__(nn.MSELoss, reduction, kernel_size, sigma, norm)


def hfen_l1(
    inp: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    kernel_size: int | list[int] = 15,
    sigma: float | list[float] = 2.5,
    norm: bool = False,
) -> torch.Tensor:
    """Calculates HFENL1 loss between inp and target.

    Parameters
    ----------
    inp : torch.Tensor
        inp tensor.
    target : torch.Tensor
        Target tensor.
    reduction : str
        Criterion reduction. Default: "mean".
    kernel_size : int or list of ints
        Size of the LoG  filter kernel. Default: 15.
    sigma : float or list of floats
        Standard deviation of the LoG filter kernel. Default: 2.5.
    norm : bool
        Whether to normalize the loss.
    """
    hfen_metric = HFENL1Loss(reduction, kernel_size, sigma, norm)
    return hfen_metric(inp, target)


def hfen_l2(
    inp: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    kernel_size: int | list[int] = 15,
    sigma: float | list[float] = 2.5,
    norm: bool = False,
) -> torch.Tensor:
    """Calculates HFENL2 loss between inp and target.

    Parameters
    ----------
    inp : torch.Tensor
        inp tensor.
    target : torch.Tensor
        Target tensor.
    reduction : str
        Criterion reduction. Default: "mean".
    kernel_size : int or list of ints
        Size of the LoG  filter kernel. Default: 15.
    sigma : float or list of floats
        Standard deviation of the LoG filter kernel. Default: 2.5.
    norm : bool
        Whether to normalize the loss.
    """
    hfen_metric = HFENL2Loss(reduction, kernel_size, sigma, norm)
    return hfen_metric(inp, target)
