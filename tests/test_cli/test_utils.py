# coding=utf-8
# Copyright (c) DIRECT Contributors

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
