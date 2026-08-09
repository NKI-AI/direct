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
"""Tests for smoothness loss functionals."""

import pytest
import torch

from direct.functionals.smooth import SmoothLoss, SmoothLossL1, SmoothLossL2, SmoothLossPenaltyType


@pytest.mark.parametrize("penalty", [SmoothLossPenaltyType.L1, SmoothLossPenaltyType.L2])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_smooth_loss_2d(penalty: SmoothLossPenaltyType, reduction: str) -> None:
    loss = SmoothLoss(penalty=penalty, reduction=reduction)
    field = torch.randn(2, 2, 16, 16)
    value = loss(field)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_smooth_loss_l1_l2_wrappers() -> None:
    field = torch.randn(1, 2, 12, 10)
    assert torch.isfinite(SmoothLossL1()(field))
    assert torch.isfinite(SmoothLossL2()(field))


def test_smooth_loss_3d() -> None:
    field = torch.randn(1, 3, 8, 8, 8)
    value = SmoothLossL2(reduction="mean")(field)
    assert torch.isfinite(value)
