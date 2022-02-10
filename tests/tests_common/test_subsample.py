# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Tests for the direct.common.subsample module."""

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code has been adjusted to our needs.

import numpy as np
import pytest
import torch

from direct.common.subsample import FastMRIRandomMaskFunc, RadialMaskFunc, SpiralMaskFunc


@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
def test_fastmri_random_mask_reuse(center_fracs, accelerations, batch_size, dim):
    mask_func = FastMRIRandomMaskFunc(center_fracs, accelerations)
    shape = (batch_size, dim, dim, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
def test_fastmri_random_mask_low_freqs(center_fracs, accelerations, batch_size, dim):
    mask_func = FastMRIRandomMaskFunc(center_fracs, accelerations)
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
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_apply_mask_fastmri(shape, center_fractions, accelerations):
    mask_func = FastMRIRandomMaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        uniform_range=False,
    )
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_same_across_volumes_mask_fastmri(shape, center_fractions, accelerations):
    mask_func = FastMRIRandomMaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        uniform_range=False,
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
