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
