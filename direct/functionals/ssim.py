# coding=utf-8
# Copyright (c) DIRECT Contributors

# Taken from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# Licensed under MIT.
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Some changes are made to work together with DIRECT.

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


__all__ = ("SSIMLoss", "fastmri_ssim")


def _to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor

    else:
        return tensor.cpu().numpy()


def fastmri_ssim(gt, target):
    # TODO(jt): UGLY.
    from skimage.metrics import structural_similarity as skimage_ssim

    gt = _to_numpy(gt)[:, 0, ...]
    target = _to_numpy(target)[:, 0, ...]
    out = skimage_ssim(
        gt.transpose(1, 2, 0),
        target.transpose(1, 2, 0),
        multichannel=True,
        data_range=gt.max(),
    )
    return torch.from_numpy(np.array([out])).float()


class SSIMLoss(nn.Module):
    """
    SSIM loss module.

    From: https://github.com/facebookresearch/fastMRI/blob/master/fastmri/losses.py
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Args:
            win_size (int, default=7): Window size for SSIM calculation.
            k1 (float, default=0.1): k1 parameter for SSIM calculation.
            k2 (float, default=0.03): k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
