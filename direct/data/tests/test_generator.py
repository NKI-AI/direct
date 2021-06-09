# coding=utf-8
# Copyright (c) DIRECT Contributors

import sys
sys.path.insert(0, '../')
import pytest

from direct.data.generator import FakeMRIDataGenerator

@pytest.mark.parametrize(
    "size",
    [
        1,
        3,
    ],
)

@pytest.mark.parametrize(
    "num_coils",
    [
        1,
        8,
    ],
)

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (32, 32),
        (10, 32, 32),
        [10, 32, 32]
    ],
)

def test_generator(size, num_coils, spatial_shape):

    generator = FakeMRIDataGenerator(ndim=len(spatial_shape))

    samples = generator(size, num_coils, spatial_shape)
    keys = ["kspace", "reconstruction_rss", "attrs"]

    assert all(_ in samples[0].keys() for _ in keys)

    assert len(samples) == size

    assert all(sample[keys[0]].shape[1] == num_coils for sample in samples)

    assert all(tuple(sample[keys[0]].shape)[-2:] == tuple(spatial_shape)[-2:] for sample in samples)
    assert all(tuple(sample[keys[1]].shape)[-2:] == tuple(spatial_shape)[-2:] for sample in samples)

    slice_num = 1 if len(spatial_shape) == 2 else spatial_shape[0]
    assert all(sample[keys[0]].shape[0] == slice_num for sample in samples)
    assert all(sample[keys[1]].shape[0] == slice_num for sample in samples)



