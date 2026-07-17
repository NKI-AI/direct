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
"""Tests for field-strength parsing from dataset filenames."""

import numpy as np
import pytest

from direct.utils.dataset import maybe_attach_field_strength, parse_field_strength_tesla


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("2025_train_Center005_UIH_15T_umr670_P001_lge_lax_2ch.mat", 1.5),
        ("2025_train_Center001_UIH_30T_umr780_P001_cine_sax.mat", 3.0),
        ("subject_055T_scan.mat", 0.55),
        ("scanner_70T_data.h5", 7.0),
        ("prefix_7T_suffix.h5", 7.0),
        ("/data/path/Center_15T_file.mat", 1.5),
        ("file1000000.h5", None),
        ("no_field_token.h5", None),
        ("P015_cine.mat", None),  # digits without trailing T
    ],
)
def test_parse_field_strength_tesla(filename, expected):
    assert parse_field_strength_tesla(filename) == expected


def test_maybe_attach_field_strength_adds_key():
    sample = {"filename": "2025_train_Center001_UIH_30T_umr780_P001.mat"}
    maybe_attach_field_strength(sample)
    assert "field_strength" in sample
    np.testing.assert_allclose(sample["field_strength"], np.array([3.0]))


def test_maybe_attach_field_strength_skips_when_absent():
    sample = {"filename": "file1000000.h5"}
    maybe_attach_field_strength(sample)
    assert "field_strength" not in sample
