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
"""Tests for the direct.data.fake module."""

import numpy as np
import pytest

from direct.data.fake import FakeMRIData


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
    [(32, 32), (10, 32, 32), [10, 32, 32]],
)
def test_fake(size, num_coils, spatial_shape):
    fake_data = FakeMRIData(ndim=len(spatial_shape))

    samples = fake_data(size, num_coils, spatial_shape)
    keys = ["kspace", "reconstruction_rss", "attrs"]

    assert all(_ in samples[0].keys() for _ in keys)

    assert len(samples) == size

    assert all(sample[keys[0]].shape[1] == num_coils for sample in samples)

    assert all(tuple(sample[keys[0]].shape)[-2:] == tuple(spatial_shape)[-2:] for sample in samples)
    assert all(tuple(sample[keys[1]].shape)[-2:] == tuple(spatial_shape)[-2:] for sample in samples)

    slice_num = 1 if len(spatial_shape) == 2 else spatial_shape[0]
    assert all(sample[keys[0]].shape[0] == slice_num for sample in samples)
    assert all(sample[keys[1]].shape[0] == slice_num for sample in samples)

    assert all(np.allclose(rss(sample[keys[0]]), sample[keys[1]]) for sample in samples)


def ifft(data, dims=(-2, -1)):
    data = np.fft.ifftshift(data, dims)
    out = np.fft.ifft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)
    return out


def rss(data, coil_dim=1):
    return np.sqrt((np.abs(ifft(data)) ** 2).sum(coil_dim))
