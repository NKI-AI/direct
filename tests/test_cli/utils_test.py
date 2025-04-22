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
import argparse
import pathlib
import tempfile

import pytest

from direct.cli.utils import file_or_url, is_file


@pytest.mark.parametrize("real_file", [True, False])
def test_is_file(real_file):
    if real_file:
        with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
            assert pathlib.Path(temp_file.name) == is_file(temp_file.name)
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            is_file("fake_name")
