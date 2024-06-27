# Copyright (c) DIRECT Contributors

"""Tests for the direct.utils module."""

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from direct.utils import is_power_of_two, normalize_image, remove_keys, reshape_array_to_shape, set_all_seeds
from direct.utils.asserts import assert_complex
from direct.utils.bbox import crop_to_largest
from direct.utils.dataset import get_filenames_for_datasets_from_config


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
    "shape, complex_axis, complex_last",
    [
        [[3, 3, 2], None, True],
        [[5, 8, 4, 2], -1, None],
        [[5, 2, 8, 4], 1, None],
        [[3, 5, 8, 4, 2], None, True],
        [[3, 5, 2, 8, 4], -3, None],
        [[3, 2, 5, 8, 4], 1, None],
        [[3, 3, 5, 8, 4, 2], None, True],
        [[3, 3, 2, 5, 8, 4], 2, None],
    ],
)
def test_is_complex_data(shape, complex_axis, complex_last):
    data = create_input(shape)

    assert_complex(data, complex_axis=complex_axis, complex_last=complex_last)


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

        cfg = mock_cfg(filenames_lists=["mock_list.lst"]) if file_list else mock_cfg()
        filenames = get_filenames_for_datasets_from_config(cfg, files_root=path_to_list, data_root=data_root)
        if file_list:
            assert len(filenames) == num_samples
        else:
            assert filenames is None


@pytest.mark.parametrize("seed", [100, 10, None])
@pytest.mark.parametrize("shape", [[3, 3, 2], [5, 8, 4, 2]])
def test_set_all_seeds(seed, shape):
    arrays = []
    for _ in range(2):
        if seed is not None:
            set_all_seeds(seed)
        arrays.append(np.random.randn(*shape))

    assert np.allclose(arrays[0], arrays[1]) == (seed is not None)


@pytest.mark.parametrize("keys", [["test_key1", "test_key2"]])
@pytest.mark.parametrize("del_keys", [["test_key1"], []])
def test_remove_keys(keys, del_keys):
    dictionary = {k: None for k in keys}
    dictionary = remove_keys(dictionary, del_keys)
    assert set(dictionary.keys()) == (set(keys) - set(del_keys))


@pytest.mark.parametrize("shape", [[4, 3, 3, 2], [5, 8, 2]])
@pytest.mark.parametrize("eps", [0.00001, 0.0001])
def test_normalize_image(shape, eps):
    img = np.random.randn(*shape)
    normalized_img = normalize_image(img, eps)
    assert normalized_img.min() >= 0.0 and normalized_img.max() <= 1.0


@pytest.mark.parametrize(
    "array, requested_shape, expected_shape",
    [
        (np.random.rand(4, 5), (4, 5, 1), (4, 5, 1)),
        (np.random.rand(4, 5), (1, 4, 5, 1), (1, 4, 5, 1)),
        (np.random.rand(2, 4, 5), (2, 4, 5, 1), (2, 4, 5, 1)),
        (np.random.rand(3, 3), (1, 3, 1, 3, 1), (1, 3, 1, 3, 1)),
        (np.random.rand(2, 3), (2, 1, 3), (2, 1, 3)),
        (np.random.rand(4), (1, 1, 4, 1), (1, 1, 4, 1)),
        (np.random.rand(6), (1, 6, 1), (1, 6, 1)),
    ],
)
def test_reshape_array_to_shape(array, requested_shape, expected_shape):
    result = reshape_array_to_shape(array, requested_shape)
    assert result.shape == expected_shape
