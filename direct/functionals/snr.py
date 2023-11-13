# Copyright (c) DIRECT Contributors

"""direct.nn.functionals.snr module."""

import torch
from torch import nn

__all__ = ("snr", "SNRLoss")


def snr(input_data: torch.Tensor, target_data: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """This function is a torch implementation of SNR metric for batches.

    ..math::
        SNR = 10 \cdot \log_{10}\left(\frac{\sum \text{square_error}}{\sum \text{square_error_noise}}\right)

    Where:
    - \text{square_error} is the sum of squared values of the clean (target) data.
    - \text{square_error_noise} is the sum of squared differences between the input data and the clean (target) data.

    If reduction is "mean", the function returns the mean SNR value.
    If reduction is "sum", the function returns the sum of SNR values.
    If reduction is "none", the function returns a tensor of SNR values for each batch.


    Parameters
    ----------
    input_data : torch.Tensor
    target_data : torch.Tensor
    reduction : str

    Returns
    -------
    torch.Tensor
    """
    batch_size = target_data.size(0)
    input_view = input_data.view(batch_size, -1)
    target_view = target_data.view(batch_size, -1)

    square_error = torch.sum(target_view**2, 1)
    square_error_noise = torch.sum((input_view - target_view) ** 2, 1)
    snr_metric = 10.0 * (torch.log10(square_error) - torch.log10(square_error_noise))

    if reduction == "mean":
        return snr_metric.mean()
    if reduction == "sum":
        return snr_metric.sum()
    if reduction == "none":
        return snr_metric
    raise ValueError(f"Reduction is either `mean`, `sum` or `none`. Got {reduction}.")


class SNRLoss(nn.Module):
    """SNR loss function PyTorch implementation."""

    def __init__(self, reduction: str = "mean") -> None:
        """Inits :class:`SNRLoss`.

        Parameters
        ----------
        reduction : str
            Batch reduction. Default: str.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SNRLoss`.

        Parameters
        ----------
        input_data : torch.Tensor
        target_data : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return snr(input_data, target_data, reduction=self.reduction)
