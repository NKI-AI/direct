# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import torch.nn as nn

__all__ = (
    "batch_psnr",
    "PSNRLoss",
)


def batch_psnr(input, target, reduction="mean"):
    """
    This function is a torch implementation of skimage.metrics.compare_psnr

    Parameters
    ----------
    input : torch.Tensor
    target : torch.Tensor
    reduction : str

    Returns
    -------
    torch.Tensor
    """
    batch_size = target.size(0)
    input_view = input.view(batch_size, -1)
    target_view = target.view(batch_size, -1)
    maximum_value = torch.max(input_view, 1)[0]

    mean_square_error = torch.mean((input_view - target_view) ** 2, 1)
    psnrs = 20.0 * torch.log10(maximum_value) - 10.0 * torch.log10(mean_square_error)

    if reduction == "mean":
        return psnrs.mean()
    elif reduction == "sum":
        return psnrs.sum()
    elif reduction == "none":
        return psnrs

    else:
        raise ValueError(
            f"Reduction is either `mean`, `sum` or `none`. Got {reduction}."
        )


class PSNRLoss(nn.Module):
    __constants__ = ["reduction"]

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, input, target):
        return batch_psnr(input, target, reduction=self.reduction)
