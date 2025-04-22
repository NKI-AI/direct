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
"""Tests for the direct.data.lr_scheduler module."""

import numpy as np
import pytest
import torch

from direct.data.lr_scheduler import LRScheduler, WarmupCosineLR, WarmupMultiStepLR


def create_model():
    return torch.nn.Linear(2, 3)


def create_optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.mark.parametrize(
    "gamma, milestones, warm_up_iters",
    [[0.25, [10, 20, 25, 30], 15], [0.1, [10, 20, 25, 30], 45]],
)
@pytest.mark.parametrize(
    "method",
    ["constant", "linear", None],
)
def test_WarmupMultiStepLR(milestones, warm_up_iters, gamma, method):
    model = create_model()
    optimizer = create_optimizer(model)
    if method:
        scheduler = WarmupMultiStepLR(
            optimizer, milestones, warmup_iterations=warm_up_iters, gamma=gamma, warmup_method=method
        )
        tmp = scheduler.get_lr()
        for iter in range(1, milestones[-1] * 2):
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            if iter >= warm_up_iters:
                if iter in milestones:
                    assert np.allclose(lr[0] / tmp[0], gamma)
                elif (iter - 1) in milestones and iter not in milestones and (iter + 1) not in milestones:
                    assert tmp == lr
            else:
                if iter in milestones:
                    assert lr[0] < tmp[0]
            tmp = lr
    else:
        with pytest.raises(ValueError):
            scheduler = WarmupMultiStepLR(optimizer, milestones, warmup_method=method)
