# Copyright (c) DIRECT Contributors

"""This module contains SSIM loss functions for the direct package."""


# Taken from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# Licensed under MIT.
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Some changes are made to work together with DIRECT.

# pylint: disable=too-many-locals

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SSIMLoss", "SSIM3DLoss")


class SSIMLoss(nn.Module):
    """SSIM loss module as implemented in [1]_.

    Parameters
    ----------
    win_size: int
        Window size for SSIM calculation. Default: 7.
    k1: float
        k1 parameter for SSIM calculation. Default: 0.1.
    k2: float
        k2 parameter for SSIM calculation. Default: 0.03.

    References
    ----------

    .. [1] https://github.com/facebookresearch/fastMRI/blob/master/fastmri/losses.py

    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03) -> None:
        """Inits :class:`SSIMLoss`.

        Parameters
        ----------
        win_size: int
            Window size for SSIM calculation. Default: 7.
        k1: float
            k1 parameter for SSIM calculation. Default: 0.1.
        k2: float
            k2 parameter for SSIM calculation. Default: 0.03.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SSIMloss`.

        Parameters
        ----------
        input_data : torch.Tensor
            2D Input data.
        target_data : torch.Tensor
            2D Target data.
        data_range : torch.Tensor
            Data range.

        Returns
        -------
        torch.Tensor
        """
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(input_data, self.w)
        uy = F.conv2d(target_data, self.w)
        uxx = F.conv2d(input_data * input_data, self.w)
        uyy = F.conv2d(target_data * target_data, self.w)
        uxy = F.conv2d(input_data * target_data, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class SSIM3DLoss(nn.Module):
    """SSIM loss module for 3D data.

    Parameters
    ----------
    win_size: int
        Window size for SSIM calculation. Default: 7.
    k1: float
        k1 parameter for SSIM calculation. Default: 0.1.
    k2: float
        k2 parameter for SSIM calculation. Default: 0.03.
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03) -> None:
        """Inits :class:`SSIM3DLoss`.

        Parameters
        ----------
        win_size: int
            Window size for SSIM calculation. Default: 7.
        k1: float
            k1 parameter for SSIM calculation. Default: 0.1.
        k2: float
            k2 parameter for SSIM calculation. Default: 0.03.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SSIM3Dloss`.

        Parameters
        ----------
        input_data : torch.Tensor
            3D Input data.
        target_data : torch.Tensor
            3D Target data.
        data_range : torch.Tensor
            Data range.

        Returns
        -------
        torch.Tensor
        """
        data_range = data_range[:, None, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        # window size across last dimension is chosen to be the last dimension size if smaller than given window size
        win_size_z = min(self.win_size, input_data.size(2))

        NP = win_size_z * self.win_size**2
        w = torch.ones(1, 1, win_size_z, self.win_size, self.win_size, device=input_data.device) / NP
        cov_norm = NP / (NP - 1)

        ux = F.conv3d(input_data, w)
        uy = F.conv3d(target_data, w)
        uxx = F.conv3d(input_data * input_data, w)
        uyy = F.conv3d(target_data * target_data, w)
        uxy = F.conv3d(input_data * target_data, w)

        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
