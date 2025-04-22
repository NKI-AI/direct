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
"""General utilities for module imports."""

from importlib.util import find_spec


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    Adapted from: https://github.com/PyTorchLightning/pytorch-lightning/blob/ef7d41692ca04bb9877da5c743f80fceecc6a100/pytorch_lightning/utilities/imports.py#L27
    Under Apache 2.0 license.
    """
    try:
        return find_spec(module_path) is not None
    except ModuleNotFoundError:
        return False
