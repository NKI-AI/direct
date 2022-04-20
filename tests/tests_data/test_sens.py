# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest

from direct.data.sens import simulate_sensitivity_maps


@pytest.mark.parametrize(
    "num_coils",
    [1, 8],
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (10, 32, 32), (11, 12, 13)],
)
@pytest.mark.parametrize(
    "var",
    [0.5],
)
@pytest.mark.parametrize(
    "seed",
    [None, 0],
)
def test_simulate_sens_maps(num_coils, shape, var, seed):

    sensitivity_map = simulate_sensitivity_maps(shape, num_coils, var, seed)

    assert tuple(sensitivity_map.shape) == (num_coils,) + tuple(shape)
