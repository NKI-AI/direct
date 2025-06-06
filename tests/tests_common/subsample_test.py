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
"""Tests for the direct.common.subsample module."""

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code has been adjusted to our needs.

import numpy as np
import pytest
import torch

from direct.common.subsample import *


@pytest.mark.parametrize(
    "mask_func",
    [
        FastMRIRandomMaskFunc,
        FastMRIEquispacedMaskFunc,
        FastMRIMagicMaskFunc,
        Gaussian1DMaskFunc,
        Gaussian2DMaskFunc,
        VariableDensityPoissonMaskFunc,
    ],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE],
)
def test_mask_reuse(mask_func, center_fracs, accelerations, batch_size, dim, mode):
    mask_func = mask_func(center_fractions=center_fracs, accelerations=accelerations, mode=mode)
    shape = (batch_size, dim, dim, 2) if mode == MaskFuncMode.STATIC else (batch_size, dim // 100, dim, dim, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize(
    "mask_func",
    [
        RadialMaskFunc,
        SpiralMaskFunc,
    ],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE],
)
def test_mask_reuse_circus(mask_func, center_fracs, accelerations, batch_size, dim, mode):
    mask_func = mask_func(accelerations=accelerations, center_fractions=center_fracs, mode=mode)
    shape = (batch_size, dim, dim, 2) if mode == MaskFuncMode.STATIC else (batch_size, dim // 100, dim, dim, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize(
    "mask_func",
    [
        CartesianEquispacedMaskFunc,
        CartesianMagicMaskFunc,
        CartesianRandomMaskFunc,
    ],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([10], [4], 4, 320),
        ([30, 20], [4, 8], 2, 368),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE],
)
def test_mask_reuse_cartesian(mask_func, center_fracs, accelerations, batch_size, dim, mode):
    mask_func = mask_func(center_fractions=center_fracs, accelerations=accelerations, mode=mode)
    shape = (batch_size, dim, dim, 2) if mode == MaskFuncMode.STATIC else (batch_size, dim // 100, dim, dim, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize(
    "mask_func",
    [KtGaussian1DMaskFunc, KtRadialMaskFunc, KtUniformMaskFunc],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, shape",
    [
        ([0.2], [4], 4, [10, 200, 300]),
        ([0.2, 0.4], [4, 8], 2, [4, 220, 200]),
    ],
)
def test_mask_reuse_kt(mask_func, center_fracs, accelerations, batch_size, shape):
    mask_func = mask_func(center_fractions=center_fracs, accelerations=accelerations)
    shape = (batch_size, *shape, 2)
    mask1 = mask_func(shape, seed=123)
    mask2 = mask_func(shape, seed=123)
    mask3 = mask_func(shape, seed=123)
    assert torch.all(mask1 == mask2)
    assert torch.all(mask2 == mask3)


@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc, Gaussian1DMaskFunc],
)
@pytest.mark.parametrize(
    "center_fracs, accelerations, batch_size, dim",
    [
        ([0.2], [4], 4, 320),
        ([0.2, 0.4], [4, 8], 2, 368),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, MaskFuncMode.MULTISLICE],
)
def test_cartesian_mask_low_freqs(mask_func, center_fracs, accelerations, batch_size, dim, mode):
    mask_func = mask_func(center_fractions=center_fracs, accelerations=accelerations, mode=mode)
    shape = (batch_size, dim, dim, 2) if mode == MaskFuncMode.STATIC else (batch_size, dim // 100, dim, dim, 2)
    mask = mask_func(shape, seed=123)

    mask_shape = [1] * (len(shape) + 1)
    mask_shape[-3:-1] = shape[-3:-1]
    mask_shape[-4] = mask_shape[-4] if mode == MaskFuncMode.STATIC else shape[-4]

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
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc, Gaussian1DMaskFunc],
)
@pytest.mark.parametrize(
    "shape, center_fractions, accelerations, mode",
    [
        ([4, 32, 32, 2], [0.08], [4], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4], MaskFuncMode.STATIC),
        ([4, 5, 32, 32, 2], [0.08], [4], MaskFuncMode.STATIC),
        ([4, 5, 32, 32, 2], [0.08], [4], MaskFuncMode.DYNAMIC),
        ([4, 5, 32, 32, 2], [0.04], [8], MaskFuncMode.MULTISLICE),
    ],
)
def test_apply_mask_cartesian(mask_func, shape, center_fractions, accelerations, mode):
    mask_func = mask_func(center_fractions=center_fractions, accelerations=accelerations, mode=mode)
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = [1] * len(shape)
    expected_mask_shape[-3:-1] = shape[-3:-1]
    expected_mask_shape[-4] = expected_mask_shape[-4] if mode == MaskFuncMode.STATIC else shape[-4]

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == tuple(expected_mask_shape)
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "mask_func",
    [FastMRIRandomMaskFunc, FastMRIEquispacedMaskFunc, FastMRIMagicMaskFunc, Gaussian1DMaskFunc],
)
@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_same_across_volumes_mask_cartesian_fraction_center(mask_func, shape, center_fractions, accelerations):
    mask_func = mask_func(center_fractions=center_fractions, accelerations=accelerations)
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))


@pytest.mark.parametrize(
    "mask_func",
    [CartesianEquispacedMaskFunc, CartesianMagicMaskFunc, CartesianRandomMaskFunc],
)
@pytest.mark.parametrize(
    "shape, center_fractions, accelerations, mode",
    [
        ([4, 32, 32, 2], [6], [4], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [4, 6], [8, 4], MaskFuncMode.STATIC),
        ([4, 5, 32, 32, 2], [6], [4], MaskFuncMode.STATIC),
        ([4, 5, 32, 32, 2], [6], [4], MaskFuncMode.DYNAMIC),
        ([4, 5, 32, 32, 2], [6], [4], MaskFuncMode.MULTISLICE),
    ],
)
def test_same_across_volumes_mask_cartesian(mask_func, shape, center_fractions, accelerations, mode):
    mask_func = mask_func(center_fractions=center_fractions, accelerations=accelerations, mode=mode)
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([1, 300, 200, 2], [5]),
        ([1, 218, 170, 2], [4]),
        ([1, 218, 170, 2], [5]),
        ([1, 218, 174, 2], [5]),
        ([1, 218, 180, 2], [5]),
        ([1, 218, 170, 2], [10]),
        ([1, 218, 174, 2], [10]),
        ([1, 218, 180, 2], [10]),
    ],
)
def test_apply_mask_calgary_campinas(shape, accelerations):
    if any(r not in [5, 10] for r in accelerations):
        with pytest.raises(ValueError):
            mask_func = CalgaryCampinasMaskFunc(accelerations=accelerations)
    else:
        mask_func = CalgaryCampinasMaskFunc(accelerations=accelerations)
        if tuple(shape[1:-1]) not in mask_func.shapes:
            with pytest.raises(ValueError):
                mask = mask_func(shape[1:], seed=123)
        else:
            mask = mask_func(shape[1:], seed=123)
            acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
            expected_mask_shape = (1, shape[1], shape[2], 1)

            assert mask.max() == 1
            assert mask.min() == 0
            assert mask.shape == expected_mask_shape
            assert acs_mask.shape == expected_mask_shape


@pytest.mark.parametrize(
    "shape, accelerations",
    [
        ([3, 218, 170, 2], [5, 10]),
    ],
)
def test_same_across_volumes_mask_calgary_campinas(shape, accelerations):
    mask_func = CalgaryCampinasMaskFunc(
        accelerations=accelerations,
    )
    num_slices = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(num_slices)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(num_slices - 1))


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 3, 32, 32, 2], [4], None, MaskFuncMode.DYNAMIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
        ([4, 1, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], None, MaskFuncMode.STATIC),
    ],
)
def test_apply_mask_radial(shape, accelerations, center_fractions, mode):
    mask_func = RadialMaskFunc(accelerations=accelerations, center_fractions=center_fractions, mode=mode)
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = [1] * len(shape)
    expected_mask_shape[-3:-1] = shape[-3:-1]
    expected_mask_shape[-4] = expected_mask_shape[-4] if mode == MaskFuncMode.STATIC else shape[-4]

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == tuple(expected_mask_shape)
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], None, MaskFuncMode.STATIC),
        ([1, 3, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([1, 2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([1, 3, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([1, 3, 64, 64, 2], [8, 4], None, MaskFuncMode.STATIC),
        ([1, 3, 32, 32, 2], [4], [0.08], MaskFuncMode.DYNAMIC),
        ([1, 2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
        ([1, 3, 32, 32, 2], [4], None, MaskFuncMode.DYNAMIC),
        ([1, 3, 64, 64, 2], [8, 4], None, MaskFuncMode.DYNAMIC),
    ],
)
def test_same_across_volumes_mask_radial(shape, accelerations, center_fractions, mode):
    mask_func = RadialMaskFunc(accelerations=accelerations, center_fractions=center_fractions, mode=mode)
    batch_sz = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(batch_sz)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(batch_sz - 1))


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 3, 32, 32, 2], [4], None, MaskFuncMode.DYNAMIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
        ([4, 1, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], None, MaskFuncMode.STATIC),
    ],
)
def test_apply_mask_spiral(shape, accelerations, center_fractions, mode):
    mask_func = SpiralMaskFunc(accelerations=accelerations, center_fractions=center_fractions, mode=mode)
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)
    expected_mask_shape = [1] * len(shape)
    expected_mask_shape[-3:-1] = shape[-3:-1]
    expected_mask_shape[-4] = expected_mask_shape[-4] if mode == MaskFuncMode.STATIC else shape[-4]

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == tuple(expected_mask_shape)
    assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], None, MaskFuncMode.STATIC),
        ([1, 3, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([1, 2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([1, 3, 32, 32, 2], [4], None, MaskFuncMode.STATIC),
        ([1, 3, 64, 64, 2], [8, 4], None, MaskFuncMode.STATIC),
        ([1, 3, 32, 32, 2], [4], [0.08], MaskFuncMode.DYNAMIC),
        ([1, 2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
        ([1, 3, 32, 32, 2], [4], None, MaskFuncMode.DYNAMIC),
        ([1, 3, 64, 64, 2], [8, 4], None, MaskFuncMode.DYNAMIC),
    ],
)
def test_same_across_volumes_mask_spiral(shape, accelerations, center_fractions, mode):
    mask_func = SpiralMaskFunc(accelerations=accelerations, center_fractions=center_fractions, mode=mode)
    batch_sz = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(batch_sz)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(batch_sz - 1))


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
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
def test_apply_mask_poisson(shape, accelerations, center_fractions, seed, mode):
    mask_func = VariableDensityPoissonMaskFunc(
        accelerations=accelerations,
        center_fractions=center_fractions,
        mode=mode,
    )
    mask = mask_func(shape[1:], seed=seed)
    acs_mask = mask_func(shape[1:], seed=seed, return_acs=True)
    expected_mask_shape = [1] * len(shape)
    expected_mask_shape[-3:-1] = shape[-3:-1]
    expected_mask_shape[-4] = expected_mask_shape[-4] if mode == MaskFuncMode.STATIC else shape[-4]
    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == tuple(expected_mask_shape)
    if seed is not None:
        assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 2, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([4, 2, 32, 32, 2], [4], [0.08], MaskFuncMode.DYNAMIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.MULTISLICE),
    ],
)
def test_same_across_volumes_mask_poisson(shape, accelerations, center_fractions, mode):
    mask_func = VariableDensityPoissonMaskFunc(
        accelerations=accelerations,
        center_fractions=center_fractions,
        mode=mode,
    )
    batch_sz = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(batch_sz)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(batch_sz - 1))


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([4, 2, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([4, 2, 32, 32, 2], [4], [0.08], MaskFuncMode.MULTISLICE),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
    ],
)
def test_same_across_volumes_mask_gaussian_2d(shape, accelerations, center_fractions, mode):
    mask_func = Gaussian2DMaskFunc(accelerations=accelerations, center_fractions=center_fractions, mode=mode)
    batch_sz = shape[0]
    masks = [mask_func(shape[1:], seed=123) for _ in range(batch_sz)]

    assert all(np.allclose(masks[_], masks[_ + 1]) for _ in range(batch_sz - 1))


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions, mode",
    [
        ([4, 32, 32, 2], [4], [0.08], MaskFuncMode.STATIC),
        ([2, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.DYNAMIC),
        ([2, 3, 64, 64, 2], [8, 4], [0.04, 0.08], MaskFuncMode.STATIC),
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
def test_apply_mask_gaussian_2d(shape, accelerations, center_fractions, seed, mode):
    mask_func = Gaussian2DMaskFunc(accelerations=accelerations, center_fractions=center_fractions, mode=mode)
    mask = mask_func(shape[1:], seed=seed)
    acs_mask = mask_func(shape[1:], seed=seed, return_acs=True)
    expected_mask_shape = [1] * len(shape)
    expected_mask_shape[-3:-1] = shape[-3:-1]
    expected_mask_shape[-4] = expected_mask_shape[-4] if mode == MaskFuncMode.STATIC else shape[-4]
    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == tuple(expected_mask_shape)
    if seed is not None:
        assert np.allclose(mask & acs_mask, acs_mask)


@pytest.mark.parametrize(
    "shape, accelerations, center_fractions",
    [
        ([2, 10, 64, 64, 2], [8, 4], [0.04, 0.08]),
    ],
)
@pytest.mark.parametrize(
    "mask_func",
    [
        KtGaussian1DMaskFunc,
        KtRadialMaskFunc,
        KtUniformMaskFunc,
    ],
)
def test_apply_kt_mask(mask_func, shape, accelerations, center_fractions):
    mask_func = mask_func(accelerations=accelerations, center_fractions=center_fractions)
    mask = mask_func(shape[1:], seed=123)
    acs_mask = mask_func(shape[1:], seed=123, return_acs=True)

    expected_mask_shape = [1] * len(shape)
    expected_mask_shape[-3:-1] = shape[-3:-1]
    expected_mask_shape[-4] = shape[-4]

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == tuple(expected_mask_shape)

    assert all(not np.allclose(mask[:, _], mask[:, _ + 1]) for _ in range(shape[1] - 1))
    assert all(np.allclose(acs_mask[:, _], acs_mask[:, _ + 1]) for _ in range(shape[1] - 1))
