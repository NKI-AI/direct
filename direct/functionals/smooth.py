# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""Functionals for smoothness loss on displacement fields and images."""

import torch
from torch import nn

from direct.types import DirectEnum


class SmoothLossPenaltyType(DirectEnum):
    """Penalty type for :class:`SmoothLoss`."""

    L1 = "l1"
    L2 = "l2"


class SmoothLoss(nn.Module):
    """Compute the smoothness loss based on the L1 or L2 penalty of the gradients of the input tensor based on [#]_.

    The smoothness loss is defined as the mean of the absolute or squared differences of the gradients
    along each spatial dimension. The gradients are computed using finite differences.

    Args:
        penalty: Penalty type for the smoothness loss. Can be
            :attr:`~direct.functionals.smooth.SmoothLossPenaltyType.L1` or
            :attr:`~direct.functionals.smooth.SmoothLossPenaltyType.L2`.

    References:
        .. [#] https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
    """

    def __init__(self, penalty: SmoothLossPenaltyType, reduction: str = "mean") -> None:
        """Inits :class:`SmoothLoss`.

        Args:
            penalty: Penalty type for the smoothness loss. Can be
                :attr:`~direct.functionals.smooth.SmoothLossPenaltyType.L1` or
                :attr:`~direct.functionals.smooth.SmoothLossPenaltyType.L2`.
            reduction: Batch reduction. Can be ``"mean"`` or ``"sum"``. Default is ``"mean"``.

        Returns:
            ``None``.
        """
        super().__init__()
        self.penalty = penalty
        self.reduction = reduction

    def _diffs(self, y: torch.Tensor) -> list[torch.Tensor]:
        """Calculate the finite differences ``(gradients)`` of the tensor y along each spatial dimension.

        Args:
            y: The input tensor of shape ``(N, C, *D)``, where N is the batch size, C is the number of channels, and *D
                represents the spatial dimensions.

        Returns:
            A list of tensors containing the differences along each spatial dimension. Each tensor in the list has shape
            (N, C,
                *D'), where *D' has one less element along the dimension of differentiation.
        """
        vol_shape = y.shape[2:]
        ndims = len(vol_shape)
        diffs = [None] * ndims

        for i in range(ndims):
            dim = i + 2
            # Permute the dimensions to bring the current axis to the front
            perm_order = [dim] + list(range(dim)) + list(range(dim + 1, ndims + 2))
            y_permuted = y.permute(perm_order)

            # Compute finite differences along the permuted dimension
            diff_i = y_permuted[1:, ...] - y_permuted[:-1, ...]

            # Permute the dimensions back to the original order
            reverse_perm_order = list(range(1, dim)) + [0, dim] + list(range(dim + 1, ndims + 2))
            diffs[i] = diff_i.permute(reverse_perm_order)

        return diffs  # ty: ignore[invalid-return-type]

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Compute the smoothness loss based on the specified penalty type.

        Args:
            field: Field of shape ``(N, C, *D)``, where ``N`` is the batch size, ``C`` is the number of channels, and
                ``*D`` represents the spatial dimensions.

        Returns:
            The computed smoothness loss (scalar).
        """
        diffs = self._diffs(field)

        if self.penalty == SmoothLossPenaltyType.L1:
            # L1 penalty: absolute differences
            diffs = [torch.abs(diff) for diff in diffs]
        else:
            # L2 penalty: squared differences
            diffs = [diff**2 for diff in diffs]

        # Compute the mean of flattened differences along each dimension
        mean_diffs = [torch.mean(diff.flatten(start_dim=1), dim=-1) for diff in diffs]

        # Average the mean differences across all dimensions
        grad = sum(mean_diffs) / len(mean_diffs)

        return grad.mean() if self.reduction == "mean" else grad.sum()  # ty: ignore[unresolved-attribute]


class SmoothLossL1(SmoothLoss):
    """Compute the smoothness loss based on the L1 penalty of the gradients of the input tensor.

    Args:
        reduction: Batch reduction. Can be ``"mean"`` or ``"sum"``. Default is ``"mean"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Inits :class:`SmoothLossL1`.

        Args:
            reduction: Batch reduction. Can be ``"mean"`` or ``"sum"``. Default is ``"mean"``.

        Returns:
            ``None``.
        """
        super().__init__(penalty=SmoothLossPenaltyType.L1, reduction=reduction)


class SmoothLossL2(SmoothLoss):
    """Compute the smoothness loss based on the L2 penalty of the gradients of the input tensor.

    Args:
        reduction: Batch reduction. Can be ``"mean"`` or ``"sum"``. Default is ``"mean"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Inits :class:`SmoothLossL2`.

        Args:
            reduction: Batch reduction. Can be ``"mean"`` or ``"sum"``. Default is ``"mean"``.

        Returns:
            ``None``.
        """
        super().__init__(penalty=SmoothLossPenaltyType.L2, reduction=reduction)
