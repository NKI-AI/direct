# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ["init_weights", "norm", "pad", "pad_to_square", "unnorm", "unpad", "DropoutPath"]


def pad(x: torch.Tensor, pad_size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Pad the input tensor with zeros to make its spatial dimensions divisible by the pad size.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (\*, height, width).
    pad_size : tuple[int, int]
        Patch size to make dimensions divisible with, as a tuple of integers (pad_height, pad_width).

    Returns
    -------
    tuple containing the padded tensor, and the number of pixels padded in the width and height dimensions respectively.
    """
    h, w = x.shape[-2:]
    hp, wp = pad_size
    f1 = ((wp - w % wp) % wp) / 2
    f2 = ((hp - h % hp) % hp) / 2
    wpad = (floor(f1), ceil(f1))
    hpad = (floor(f2), ceil(f2))
    x = F.pad(x, wpad + hpad)

    return x, wpad, hpad


def pad_to_square(
    inp: torch.Tensor, factor: float
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Pad a tensor to a square shape with a given factor.

    Parameters
    ----------
    inp : torch.Tensor
        The input tensor to pad to square shape. Expected shape is (\*, height, width).
    factor : float
        The factor to which the input tensor will be padded.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]
        A tuple of two tensors, the first is the input tensor padded to a square shape, and the
        second is the corresponding mask for the padded tensor.

    Examples
    --------
    1.
        >>> x = torch.rand(1, 3, 224, 192)
        >>> padded_x, mask, wpad, hpad = pad_to_square(x, factor=16.0)
        >>> padded_x.shape, mask.shape
        (torch.Size([1, 3, 224, 224]), torch.Size([1, 1, 224, 224]))
    2.
        >>> x =  torch.rand(3, 13, 2, 234, 180)
        >>> padded_x, mask, wpad, hpad = pad_to_square(x, factor=16.0)
        >>> padded_x.shape, wpad, hpad
        (torch.Size([3, 13, 2, 240, 240]), (30, 30), (3, 3))
    """
    channels, h, w = inp.shape[-3:]

    # Calculate the maximum size and pad to the next multiple of the factor
    x = int(ceil(max(h, w) / float(factor)) * factor)

    # Create a tensor of zeros with the maximum size and copy the input tensor into the center
    img = torch.zeros(*inp.shape[:-3], channels, x, x, device=inp.device).type_as(inp)
    mask = torch.zeros(*((1,) * (img.ndim - 3)), 1, x, x, device=inp.device).type_as(inp)

    # Compute the offset and copy the input tensor into the center of the zero tensor
    offset_h = (x - h) // 2
    offset_w = (x - w) // 2
    hpad = (offset_h, offset_h + h)
    wpad = (offset_w, offset_w + w)
    img[..., hpad[0] : hpad[1], wpad[0] : wpad[1]] = inp.clone()
    mask[..., hpad[0] : hpad[1], wpad[0] : wpad[1]].fill_(1.0)
    # Return the padded tensor and the corresponding mask, and padding in spatial dimensions
    return (
        img,
        1 - mask,
        (wpad[0], wpad[1] - w + (1 if w % 2 != 0 else 0)),
        (hpad[0], hpad[1] - h + (1 if h % 2 != 0 else 0)),
    )


def unpad(x: torch.Tensor, wpad: tuple[int, int], hpad: tuple[int, int]) -> torch.Tensor:
    """Remove the padding added to the input tensor by _pad method.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H_pad, W_pad).
    wpad : tuple[int, int]
        Number of pixels padded in the width dimension as a tuple of integers (left_pad, right_pad).
    hpad : tuple[int, int]
        Number of pixels padded in the height dimension as a tuple of integers (top_pad, bottom_pad).

    Returns
    -------
    Tensor with the same shape as the original input tensor, but without the added padding.
    """
    return x[..., hpad[0] : x.shape[-2] - hpad[1], wpad[0] : x.shape[-1] - wpad[1]]


def norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the input tensor by subtracting the mean and dividing by the standard deviation across each channel and pixel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).

    Returns
    -------
    tuple containing the normalized tensor, mean tensor and standard deviation tensor.
    """
    mean = x.reshape(x.shape[0], 1, 1, -1).mean(-1, keepdim=True)
    std = x.reshape(x.shape[0], 1, 1, -1).std(-1, keepdim=True)
    x = (x - mean) / std

    return x, mean, std


def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize the input tensor by multiplying with the standard deviation and adding
    the mean across each channel and pixel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    mean : torch.Tensor
        Mean tensor obtained during normalization.
    std : torch.Tensor
        Standard deviation tensor obtained during normalization.

    Returns
    -------
    Tensor with the same shape as the original input tensor, but denormalized.
    """
    return x * std + mean


def init_weights(m: nn.Module) -> None:
    """Initializes the weights of the network using a truncated normal distribution.

    Parameters
    ----------
    m : nn.Module
        A module of the network whose weights need to be initialized.
    """

    if isinstance(m, nn.Linear):
        init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        init.constant_(m.bias, 0)
        init.constant_(m.weight, 1.0)


class DropoutPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Inits :class:`DropoutPath`.

        Parameters
        ----------
        drop_prob : float
            Probability of dropping a residual connection. Default: 0.0.
        scale_by_keep : bool
            Whether to scale the remaining activations by 1 / (1 - drop_prob) to maintain the expected value of
            the activations. Default: True.
        """
        super(DropoutPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    @staticmethod
    def _dropout_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self._dropout_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"dropout_prob={round(self.drop_prob, 3):0.3f}"
