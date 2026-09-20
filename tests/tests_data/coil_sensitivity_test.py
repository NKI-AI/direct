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
import pytest
import torch

from direct.data.coil_sensitivity import simulate_coil_sensitivity_maps


@pytest.mark.parametrize("mode", ["birdcage", "surface", "biot_savart"])
def test_simulated_coil_maps_rss_and_shape(mode):
    maps = simulate_coil_sensitivity_maps((32, 40), 6, mode=mode, seed=0)
    assert maps.shape == (6, 32, 40, 2)
    power = maps.square().sum(-1).sum(0)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-5)


def test_surface_seed_is_reproducible():
    first = simulate_coil_sensitivity_maps((16, 16), 4, mode="surface", seed=3)
    second = simulate_coil_sensitivity_maps((16, 16), 4, mode="surface", seed=3)
    other = simulate_coil_sensitivity_maps((16, 16), 4, mode="surface", seed=4)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_empirical_reuse_and_compression():
    reference = simulate_coil_sensitivity_maps((24, 24), 8, mode="birdcage", seed=0)
    reused = simulate_coil_sensitivity_maps(
        (16, 16), 4, mode="empirical", reference_maps=reference, seed=0
    )
    assert reused.shape == (4, 16, 16, 2)
    power = reused.square().sum(-1).sum(0)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-5)
