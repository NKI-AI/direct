# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest

from direct.utils.io import check_is_valid_url


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
