# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest

from direct.utils.imports import _module_available


@pytest.mark.parametrize(
    ["module", "is_available"],
    [
        ("torch", True),
        ("numpy", True),
        ("non-existent", False),
    ],
)
def test_module_available(module, is_available):
    assert _module_available(module) == is_available
