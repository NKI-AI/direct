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
"""Adaptive Instance Normalization (AdaIN) modules for 2D and 3D tensors based on [1]_.

References
----------

.. [1] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
    Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
    PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import nn

__all__ = ["AdaIN2d", "AdaIN3d", "NormType"]


class NormType(str, Enum):
    INSTANCE = "instance"
    ADAIN = "adain"


class AdaIN2d(nn.Module):
    """Adaptive Instance Normalization for 2D tensors based on [1]_.

    Given input x of shape (B, C, H, W) and auxiliary vector y of shape (B, F),
    produces per-sample, per-channel affine parameters from y.

    References
    ----------

    .. [1] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        num_channels: int,
        aux_in_features: int,
        hidden_features: int | tuple[int, ...] | None = None,
        activation: nn.Module | None = None,
        eps: float = 1e-5,
        use_one_plus_gamma: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.use_one_plus_gamma = use_one_plus_gamma

        if activation is None:
            activation = nn.SiLU()

        if hidden_features is None:
            hidden = []
        elif isinstance(hidden_features, int):
            hidden = [hidden_features]
        else:
            hidden = list(hidden_features)

        layers: list[nn.Module] = []
        in_f = aux_in_features
        for h in hidden:
            layers += [nn.Linear(in_f, h), activation]
            in_f = h
        layers += [nn.Linear(in_f, 2 * num_channels)]
        self.mlp = nn.Sequential(*layers)

        if isinstance(self.mlp[-1], nn.Linear):
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        params = self.mlp(y)
        gamma, beta = params.chunk(2, 1)

        gamma = gamma.view(-1, self.num_channels, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1)

        if self.use_one_plus_gamma:
            return x_norm * (1.0 + gamma) + beta
        return x_norm * gamma + beta


class AdaIN3d(nn.Module):
    """Adaptive Instance Normalization for 3D tensors based on [1]_.

    Given input x of shape (B, C, Z, H, W) and auxiliary vector y of shape (B, F),
    produces per-sample, per-channel affine parameters from y.

    References
    ----------

    .. [1] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        num_channels: int,
        aux_in_features: int,
        hidden_features: int | tuple[int, ...] | None = None,
        activation: nn.Module | None = None,
        eps: float = 1e-5,
        use_one_plus_gamma: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.use_one_plus_gamma = use_one_plus_gamma

        if activation is None:
            activation = nn.SiLU()

        if hidden_features is None:
            hidden = []
        elif isinstance(hidden_features, int):
            hidden = [hidden_features]
        else:
            hidden = list(hidden_features)

        layers: list[nn.Module] = []
        in_f = aux_in_features
        for h in hidden:
            layers += [nn.Linear(in_f, h), activation]
            in_f = h
        layers += [nn.Linear(in_f, 2 * num_channels)]
        self.mlp = nn.Sequential(*layers)

        if isinstance(self.mlp[-1], nn.Linear):
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        params = self.mlp(y)
        gamma, beta = params.chunk(2, dim=-1)

        gamma = gamma.view(-1, self.num_channels, 1, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1, 1)

        if self.use_one_plus_gamma:
            return x_norm * (1.0 + gamma) + beta
        return x_norm * gamma + beta
