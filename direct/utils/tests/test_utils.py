# coding=utf-8
# Copyright (c) DIRECT Contributors
"""Tests for the direct.utils module"""

import numpy as np
import pytest
import torch

from direct.utils import is_power_of_two, is_complex_data
from direct.data.transforms import tensor_to_complex_numpy


def create_input(shape):
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 2],
        [5, 8, 4, 2],
        [5, 2, 8, 4],
        [3, 5, 8, 4, 2],
        [3, 5, 2, 8, 4],
        [3, 2, 5, 8, 4],
        [3, 3, 5, 8, 4, 2],
        [3, 3, 2, 5, 8, 4],
    ],
)
def test_is_complex_data(shape):
    data = create_input(shape)

    assert is_complex_data(data, False)


@pytest.mark.parametrize(
    "num",
    [1, 2, 4, 32, 128, 1024],
)
def test_is_power_of_two(num):

    assert is_power_of_two(num)
