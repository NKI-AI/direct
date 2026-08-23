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
"""direct.functionals.nmae module."""

import torch
from torch import nn

__all__ = ["NMAELoss"]


class NMAELoss(nn.Module):
    r"""Computes the Normalized Mean Absolute Error (NMAE), i.e.:

    .. math::

        \frac{||u - v||_1}{||u||_1},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction="mean") -> None:
        """Inits :class:`NMAELoss`

        Args:
            reduction: Specifies the reduction to apply to the output. Can be ``"none"``, ``"mean"`` or ``"sum"``. Note that ``"mean"`` or ``"sum"``
                will yield the same output. Default is ``"mean"``.

        Returns:
            ``None``.
        """
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NMAELoss`.

        Args:
            input: Tensor of shape (*), where * means any number of dimensions.
            target: Tensor of same shape as the input.

        Returns:
            ``None``.
        """
        return self.mae_loss(input, target) / self.mae_loss(
            torch.zeros_like(target, dtype=target.dtype, device=target.device), target
        )
