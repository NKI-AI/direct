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
"""Guards against version drift across the project's version sources.

The package version is declared in three hand-edited places: ``direct/__init__.py``
(``__version__``), ``pyproject.toml`` (``[project] version``), and ``meson.build``
(the ``project()`` version). meson-python uses the static ``pyproject.toml`` value
for wheel metadata, while Bazel/Meson consumers read ``meson.build``. These tests
fail if any of them fall out of sync.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

import direct

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _meson_version() -> str:
    """Extracts the ``version : '...'`` argument from the top-level meson.build."""
    text = (_REPO_ROOT / "meson.build").read_text(encoding="utf-8")
    match = re.search(r"version\s*:\s*'(?P<value>[^']+)'", text)
    assert match is not None, "could not find a version in meson.build"
    return match.group("value")


def _pyproject_version() -> str:
    """Reads ``[project] version`` from pyproject.toml."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["version"]


def test_dunder_version_matches_installed_metadata() -> None:
    assert direct.__version__ == importlib.metadata.version("direct")


def test_pyproject_version_matches_dunder_version() -> None:
    assert _pyproject_version() == direct.__version__


def test_meson_version_matches_dunder_version() -> None:
    assert _meson_version() == direct.__version__


@pytest.mark.parametrize("version", [_meson_version(), _pyproject_version(), direct.__version__])
def test_versions_are_pep440(version: str) -> None:
    # A minimal sanity check that each source holds a plausible release string.
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-].+)?", version), version
