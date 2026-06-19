# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Signal-to-noise ratio (SNR) metric for the direct package."""

import torch
from torch import nn

__all__ = ("snr_metric", "SNRLoss")


def snr_metric(input_data: torch.Tensor, target_data: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """This function is a torch implementation of SNR metric for batches.

    .. math::

        SNR = 10 \\cdot \\log_{10}\\left(\\frac{\\text{square_error}}{\\text{square_error_noise}}\\right)


    where:

    -   :math:`\\text{square_error}` is the sum of squared values of the clean (target) data.
    -   :math:`\\text{square_error_noise}` is the sum of squared differences between the input data and
        the clean (target) data.


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
            Batch reduction. Default: "mean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SNRLoss`.

        Parameters
        ----------
        input_data : torch.Tensor
            Input 2D data.
        target_data : torch.Tensor
            Target 2D data.

        Returns
        -------
        torch.Tensor
        """
        return snr_metric(input_data, target_data, reduction=self.reduction)
