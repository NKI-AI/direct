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
import pytest

from direct.utils.io import _normalize_huggingface_url, check_is_valid_url


@pytest.mark.parametrize(
    ["path", "is_url"],
    [
        ("https://s3.aiforoncology.nl/checkpoint.ckpt", True),
        ("http://localhost:8000/checkpoint.ckpt", True),
        ("ftp://aiforoncology.nl/checkpoint.ckpt", True),
        ("checkpoint.ckpt", False),
        ("/mnt/checkpoint.ckpt", False),
    ],
)
def test_check_valid_url(path, is_url):
    if is_url:
        assert check_is_valid_url(path)
    else:
        assert not check_is_valid_url(path)


@pytest.mark.parametrize(
    ["url", "expected_url"],
    [
        (
            "https://huggingface.co/NKI-AI/direct/blob/main/recurrentvarnet/model_148500.pt",
            "https://huggingface.co/NKI-AI/direct/resolve/main/recurrentvarnet/model_148500.pt",
        ),
        (
            "https://huggingface.co/datasets/NKI-AI/direct-dataset/blob/main/config.yaml?download=true",
            "https://huggingface.co/datasets/NKI-AI/direct-dataset/resolve/main/config.yaml?download=true",
        ),
        (
            "https://files.aiforoncology.nl/direct-project/recurrentvarnet.zip",
            "https://files.aiforoncology.nl/direct-project/recurrentvarnet.zip",
        ),
    ],
)
def test_normalize_huggingface_url(url, expected_url):
    assert _normalize_huggingface_url(url) == expected_url
