# coding=utf-8
# Copyright (c) DIRECT Contributors
"""Tests for the direct.utils module"""

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from direct.data.transforms import tensor_to_complex_numpy
from direct.utils import is_complex_data, is_power_of_two
from direct.utils.bbox import crop_to_largest
from direct.utils.dataset import get_filenames_for_datasets


def create_input(shape):
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


def mock_cfg(**kwargs):
    class Config(dict):
        def __init__(self, *args, **kwargs):
            super(Config, self).__init__(*args, **kwargs)
            self.__dict__ = self

    return Config(**kwargs)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 2],
        [5, 8, 4, 2],
        [5, 2, 8, 4],
        [3, 5, 8, 4, 2],
        [3, 5, 2, 8, 4],
        [3, 2, 5, 8, 4],
        [3, 3, 5, 8, 4, 2],
        [3, 3, 2, 5, 8, 4],
    ],
)
def test_is_complex_data(shape):
    data = create_input(shape)

    assert is_complex_data(data, False)


@pytest.mark.parametrize(
    "num",
    [1, 2, 4, 32, 128, 1024],
)
def test_is_power_of_two(num):

    assert is_power_of_two(num)


@pytest.mark.parametrize(
    "shapes",
    [
        [[5, 4, 2], [4, 3, 2]],
        [[5, 8, 4, 2], [6, 7, 3, 2]],
    ],
)
@pytest.mark.parametrize("to_numpy", [True, False])
def test_crop_to_largest(shapes, to_numpy):
    data = [create_input(shape) for shape in shapes]
    if to_numpy:
        data = [_.numpy() for _ in data]

    cropped_data = crop_to_largest(data)
    crop_to_largest_shape = tuple(max([shape[i] for shape in shapes]) for i in range(len(shapes[0])))

    assert all(tuple(_.shape) == crop_to_largest_shape for _ in cropped_data)


@pytest.mark.parametrize("file_list", [True, None])
@pytest.mark.parametrize("num_samples", [3, 4])
def test_get_filenames_for_datasets(file_list, num_samples):

    with tempfile.TemporaryDirectory() as tempdir:
        data_root = pathlib.Path(tempdir) / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        for _ in range(num_samples):
            with open(data_root / f"file_{_}.txt", "w") as f:
                f.write("Fake file.")
        path_to_list = pathlib.Path(tempdir) / "lists"
        path_to_list.mkdir(parents=True, exist_ok=True)
        for _ in range(num_samples):
            with open(path_to_list / "mock_list.lst", "a") as f:
                f.write(f"file_{_}.txt" + "\n")

        cfg = mock_cfg(lists=["mock_list.lst"]) if file_list else mock_cfg()
        filenames = get_filenames_for_datasets(cfg, files_root=path_to_list, data_root=data_root)
        if file_list:
            assert len(filenames) == num_samples
        else:
            assert filenames is None
