# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Tests for the direct.common.subsample module."""

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code has been adjusted to our needs.

import numpy as np
import pytest
import torch

from direct.common.subsample import (
    FastMRIEquispacedMaskFunc,
    FastMRIMagicMaskFunc,
    FastMRIRandomMaskFunc,
    RadialMaskFunc,
    SpiralMaskFunc,
    VariableDensityPoissonMaskFunc,
)


@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
def test_fastmri_mask_reuse(mask_func, center_fracs, accelerations, batch_size, dim):
    mask_func = mask_func(center_fractions=center_fracs, accelerations=accelerations)
    shape = (batch_size, dim, dim, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
def test_fastmri_mask_low_freqs(mask_func, center_fracs, accelerations, batch_size, dim):
    mask_func = mask_func(center_fractions=center_fracs, accelerations=accelerations)
    shape = (batch_size, dim, dim, 2)
    mask = mask_func(shape, seed=123)
    mask_shape = [1] * (len(shape) + 1)
    mask_shape[-2] = dim
    mask_shape[-3] = dim

    assert list(mask.shape) == mask_shape

    num_low_freqs_matched = False
    for center_frac in center_fracs:
        num_low_freqs = int(round(dim * center_frac))
        pad = (dim - num_low_freqs + 1) // 2
        if np.all(mask[pad : pad + num_low_freqs].numpy() == 1):
            num_low_freqs_matched = True
    assert num_low_freqs_matched


@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc],
)
@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_apply_mask_fastmri(mask_func, shape, center_fractions, accelerations):
    mask_func = mask_func(center_fractions=center_fractions, accelerations=accelerations)
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc],
)
@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_same_across_volumes_mask_fastmri(mask_func, shape, center_fractions, accelerations):
    mask_func = mask_func(center_fractions=center_fractions, accelerations=accelerations)
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_apply_mask_radial(shape, accelerations):
    mask_func = RadialMaskFunc(
        accelerations=accelerations,
    )
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_same_across_volumes_mask_radial(shape, accelerations):
    mask_func = RadialMaskFunc(
        accelerations=accelerations,
    )
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_apply_mask_spiral(shape, accelerations):
    mask_func = SpiralMaskFunc(
        accelerations=accelerations,
    )
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([4, 32, 32, 2], [4]),
        ([2, 64, 64, 2], [8, 4]),
    ],
)
def test_same_across_volumes_mask_spiral(shape, accelerations):
    mask_func = SpiralMaskFunc(
        accelerations=accelerations,
    )
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))


@pytest.mark.parametrize(
    "shape, accelerations, center_scales",
    [
        ([4, 32, 32, 2], [4], [0.08]),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08]),
    ],
)
@pytest.mark.parametrize(
    "seed",
    [
        None,
        10,
        100,
        1000,
        np.random.randint(0, 10000),
        list(np.random.randint(0, 10000, 20)),
        tuple(np.random.randint(100000, 1000000, 30)),
    ],
)
def test_apply_mask_poisson(shape, accelerations, center_scales, seed):
    mask_func = VariableDensityPoissonMaskFunc(
        accelerations=accelerations,
        center_scales=center_scales,
    )
    mask = mask_func(shape[1:], seed=seed)
    acs_mask = mask_func(shape[1:], seed=seed, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)
    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    if seed is not None:
        assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations, center_scales",
    [
        ([4, 32, 32, 2], [4], [0.08]),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08]),
    ],
)
def test_same_across_volumes_mask_spiral(shape, accelerations, center_scales):
    mask_func = VariableDensityPoissonMaskFunc(
        accelerations=accelerations,
        center_scales=center_scales,
    )
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))
