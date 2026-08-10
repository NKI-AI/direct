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
"""Tests for inference H5 writers."""

import pathlib

import h5py
import numpy as np
import torch

from direct.utils.writers import write_output_to_h5


def test_write_output_to_h5_plain_volume(tmp_path: pathlib.Path) -> None:
    volume = torch.randn(4, 1, 8, 8)
    mask = torch.ones(4, 1, 8, 8)
    metrics = {"file.h5": {"ssim": 0.9}}
    write_output_to_h5(
        ([(volume, mask, pathlib.Path("file.h5"))], metrics),
        tmp_path,
        output_key="reconstruction",
    )
    assert (tmp_path / "metrics_inference.json").exists()
    assert (tmp_path / "file.h5").exists()
    with h5py.File(tmp_path / "file.h5", "r") as f:
        assert "sampling_mask" in f
        assert "sampling_masks" not in f


def test_write_output_to_h5_ads_mask_history(tmp_path: pathlib.Path) -> None:
    volume = torch.randn(3, 1, 8, 8)
    # History on last axis: initial ACS, intermediate, final predicted.
    masks = torch.zeros(3, 1, 8, 8, 3)
    masks[..., 0] = 1.0  # initial
    masks[..., -1] = 1.0  # final (superset)
    masks[:, :, 4, :, -1] = 1.0
    write_output_to_h5(
        ([(volume, masks, "ads.h5")], {"ads.h5": {"ssim": 0.8}}),
        tmp_path,
    )
    with h5py.File(tmp_path / "ads.h5", "r") as f:
        assert f["sampling_masks"].shape == (3, 8, 8, 3)
        assert f["initial_sampling_mask"].shape == (3, 8, 8)
        assert f["sampling_mask"].shape == (3, 8, 8)
        assert np.allclose(f["initial_sampling_mask"][...], masks.numpy()[:, 0, ..., 0])
        assert np.allclose(f["sampling_mask"][...], masks.numpy()[:, 0, ..., -1])


def test_write_output_to_h5_registration_tuple(tmp_path: pathlib.Path) -> None:
    volume = torch.randn(2, 1, 6, 6)
    registration = torch.randn(2, 1, 6, 6)
    displacement = torch.randn(2, 2, 6, 6)
    write_output_to_h5(
        (
            [((volume, registration, displacement), None, "reg.h5")],
            {"reg.h5": {"psnr": 30.0}},
        ),
        tmp_path,
        volume_processing_func=lambda x: x * 2.0,
    )
    assert (tmp_path / "reg.h5").exists()
    assert (tmp_path / "metrics_inference.json").exists()
