# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest

from direct.checkpointer import _check_is_valid_url


@pytest.mark.parametrize(
    "paths",
    [
        ("https://s3.aiforoncology.nl/checkpoint.ckpt", True),
        ("http://localhost:8000/checkpoint.ckpt", True),
        ("ftp://aiforoncology.nl/checkpoint.ckpt", True),
        ("checkpoint.ckpt", False),
        ("/mnt/checkpoint.ckpt", False),
    ],
)
def test_check_valid_url(paths, is_url):
    if is_url:
        assert _check_is_valid_url(path)
    else:
        assert _check_is_valid_url(path)
