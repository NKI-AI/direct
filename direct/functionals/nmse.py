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
import torch
import torch.nn as nn

__all__ = ["NMSELoss", "NRMSELoss"]


class NMSELoss(nn.Module):
    r"""Computes the Normalized Mean Squared Error (NMSE), i.e.:

    .. math::

        \frac{||u - v||_2^2}{||u||_2^2},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction="mean") -> None:
        """Inits :class:`NMSELoss`

        Parameters
        ----------
        reduction: str
             Specifies the reduction to apply to the output. Can be "none", "mean" or "sum".
             Note that "mean" or "sum" will yield the same output. Default: "mean".
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NMSELoss`.

        Parameters
        ----------
        input: torch.Tensor
            Tensor of shape (*), where * means any number of dimensions.
        target: torch.Tensor
            Tensor of same shape as the input.
        """
        return self.mse_loss(input, target) / self.mse_loss(
            torch.zeros_like(target, dtype=target.dtype, device=target.device), target
        )


class NRMSELoss(nn.Module):
    r"""Computes the Normalized Root Mean Squared Error (NRMSE), i.e.:

    .. math::

        \frac{||u - v||_2}{||u||_2},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction="mean") -> None:
        """Inits :class:`NRMSELos`

        Parameters
        ----------
        reduction: str
             Specifies the reduction to apply to the output. Can be "none", "mean" or "sum".
             Note that "mean" or "sum" will yield the same output. Default: "mean".
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NRMSELoss`.

        Parameters
        ----------
        input: torch.Tensor
            Tensor of shape (*), where * means any number of dimensions.
        target: torch.Tensor
            Tensor of same shape as the input.
        """
        return torch.sqrt(
            self.mse_loss(input, target)
            / self.mse_loss(torch.zeros_like(target, dtype=target.dtype, device=target.device), target)
        )
