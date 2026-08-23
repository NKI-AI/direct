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
"""Peak signal-to-noise ratio (pSNR) metric for the direct package."""

import torch
from torch import nn

__all__ = ("PSNRLoss", "batch_psnr")


def batch_psnr(input_data: torch.Tensor, target_data: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """This function is a torch implementation of skimage.metrics.compare_psnr.

    Args:
        input_data: Input data.
        target_data: Target data.
        reduction: Reduction.

    Returns:
        The result.
    """
    batch_size = target_data.size(0)
    input_view = input_data.view(batch_size, -1)
    target_view = target_data.view(batch_size, -1)
    maximum_value = torch.max(input_view, 1)[0]

    mean_square_error = torch.mean((input_view - target_view) ** 2, 1)
    psnrs = 20.0 * torch.log10(maximum_value) - 10.0 * torch.log10(mean_square_error)

    if reduction == "mean":
        return psnrs.mean()
    if reduction == "sum":
        return psnrs.sum()
    if reduction == "none":
        return psnrs
    raise ValueError(f"Reduction is either `mean`, `sum` or `none`. Got {reduction}.")


class PSNRLoss(nn.Module):
    """Peak signal-to-noise ratio loss function PyTorch implementation.

    Args:
        reduction: Batch reduction. Default is ``"mean"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Inits :class:`PSNRLoss`.

        Args:
            reduction: Batch reduction. Default is ``"mean"``.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`PSNRLoss`.

        Args:
            input_data: Input 2D data.
            target_data: Target 2D data.

        Returns:
            The result.
        """
        return batch_psnr(input_data, target_data, reduction=self.reduction)
