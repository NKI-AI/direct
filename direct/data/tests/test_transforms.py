# coding=utf-8
# Copyright (c) DIRECT Contributors

# Some of this code is written by Facebook for the FastMRI challenge and is licensed under the MIT license.
# The code has been heavily edited, but some parts could still be recognized.

import numpy as np
import pytest
import torch

from direct.data import transforms
from direct.data.transforms import tensor_to_complex_numpy

from direct.common.subsample import FastMRIRandomMaskFunc


def create_input(shape):
    # data = np.arange(np.product(shape)).reshape(shape).copy()
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


# def add_names(tensor, named=True):
#     shape = tensor.shape
#
#     if len(shape) == 2:
#         names = ("height", "width")
#     elif len(shape) == 3:
#         names = ("height", "width", "complex")
#     elif len(shape) == 4:
#         names = ("coils", "height", "width", "complex")
#     else:
#         names = ("batch", "coils", "height", "width", "complex")
#
#     if named:
#         tensor = tensor.refine_names(*names)
#
#     return tensor


@pytest.mark.parametrize(
    "shape, center_fractions, accelerations",
    [
        ([4, 32, 32, 2], [0.08], [4]),
        ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
    ],
)
def test_apply_mask_fastmri(shape, center_fractions, accelerations):
    mask_func = FastMRIRandomMaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        uniform_range=False,
    )
    mask = mask_func(shape[1:], seed=123)
    expected_mask_shape = (1, shape[1], shape[2], 1)

    assert mask.max() == 1
    assert mask.min() == 0
    assert mask.shape == expected_mask_shape

@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
        [3, 4, 2, 2],
    ],
)

def test_fft2(shape):
    shape = shape + [2]
    data = create_input(shape)

    dim = (-2, -1)

    out_torch = transforms.fft2(data, dim=dim).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    data_numpy = tensor_to_complex_numpy(data)
    data_numpy = np.fft.ifftshift(data_numpy, (-2, -1))
    out_numpy = np.fft.fft2(data_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    z = out_torch - out_numpy
    assert np.allclose(out_torch, out_numpy)

#
@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
        [3, 4, 2, 2],
    ],
)

def test_ifft2(shape):
    shape = shape + [2]
    data = create_input(shape)

    dim = (-2, -1)
    out_torch = transforms.ifft2(data, dim=dim).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_numpy(data)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
        [3, 4, 2, 2],
    ],
)
def test_modulus(shape):
    shape = shape + [2]
    data = create_input(shape)
    out_torch = transforms.modulus(data).numpy()
    input_numpy = tensor_to_complex_numpy(data)
    out_numpy = np.abs(input_numpy)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape, dims",
    [
        [[3, 3], 0],
        [[4, 6], 1],
    ],
)
def test_root_sum_of_squares_real(shape, dims):
    data = create_input(shape)  # noqa
    out_torch = transforms.root_sum_of_squares(data, dims).numpy()
    out_numpy = np.sqrt(np.sum(data.numpy() ** 2, dims))
    assert np.allclose(out_torch, out_numpy)


coils_dim = 0
@pytest.mark.parametrize(
    "shape, dims",
    [
        [[3, 3, 9], coils_dim],
        [[4, 6, 4], coils_dim],
        [[15, 66, 43], coils_dim],
        [[15, 36, 23, 2], coils_dim],
    ],
)
def test_root_sum_of_squares_complex(shape, dims):
    shape = shape + [
        2,
    ]
    data = create_input(shape)  # noqa
    out_torch = transforms.root_sum_of_squares(data, dims).numpy()
    input_numpy = tensor_to_complex_numpy(data)
    out_numpy = np.sqrt(np.sum(np.abs(input_numpy) ** 2, dims))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape, target_shape",
    [
        [[10, 10], [4, 4]],
        [[4, 6], [2, 4]],
        [[8, 4], [4, 4]],
    ],
)

def test_center_crop(shape, target_shape):
    input = create_input(shape)
    out_torch = transforms.center_crop(input, target_shape).numpy()
    assert list(out_torch.shape) == target_shape


@pytest.mark.parametrize(
    "shape, target_shape",
    [
        [[10, 10], [4, 4]],
        [[4, 6], [2, 4]],
        [[8, 4], [4, 4]],
    ],
)
# @pytest.mark.parametrize("named", [True, False])
def test_complex_center_crop(shape, target_shape):
    shape = shape + [2]
    input = create_input(shape)

    out_torch = transforms.complex_center_crop(input, target_shape, offset=0).numpy()
    assert list(out_torch.shape) == target_shape + [
        2,
    ]


@pytest.mark.parametrize(
    "shift, dims",
    [
        (0, 0),
        (1, 0),
        (-1, 0),
        (100, 0),
        ((1, 2), (1, 2)),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [5, 6, 2],
        [3, 4, 5],
        [3, 11, 4, 5],
    ],
)

def test_roll(shift, dims, shape):
    data = np.arange(np.product(shape)).reshape(shape)
    torch_tensor = torch.from_numpy(data)
    out_torch = transforms.roll(torch_tensor, shift, dims).numpy()
    out_numpy = np.roll(data, shift, dims)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 7],
        [5, 6, 2],
        [3, 4, 5],
    ],
)
def test_complex_multiplication(shape):
    data_0 = np.arange(np.product(shape)).reshape(shape) + 1j * (np.arange(np.product(shape)).reshape(shape) + 1)
    data_1 = data_0 + 0.5 + 1j
    torch_tensor_0 = transforms.to_tensor(data_0)
    torch_tensor_1 = transforms.to_tensor(data_1)

    out_torch = tensor_to_complex_numpy(transforms.complex_multiplication(torch_tensor_0, torch_tensor_1))
    out_numpy = data_0 * data_1
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 7],
        [5, 6, 2],
        [3, 4, 5],
    ],
)
def test_conjugate(shape):
    data = np.arange(np.product(shape)).reshape(shape) + 1j * (np.arange(np.product(shape)).reshape(shape) + 1)
    torch_tensor = transforms.to_tensor(data)

    out_torch = tensor_to_complex_numpy(transforms.conjugate(torch_tensor))
    out_numpy = np.conjugate(data)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize("shape", [[5, 3], [2, 4, 6], [2, 11, 4, 7]])

def test_fftshift(shape):
    data = np.arange(np.product(shape)).reshape(shape)
    torch_tensor = torch.from_numpy(data)
    out_torch = transforms.fftshift(torch_tensor).numpy()
    out_numpy = np.fft.fftshift(data)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [
        [5, 3],
        [2, 4, 5],
        [2, 11, 7, 5],
    ],
)

def test_ifftshift(shape):
    data = np.arange(np.product(shape)).reshape(shape)
    torch_tensor = torch.from_numpy(data)
    out_torch = transforms.ifftshift(torch_tensor).numpy()
    out_numpy = np.fft.ifftshift(data)
    assert np.allclose(out_torch, out_numpy)
