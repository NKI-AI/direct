# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Tests for the direct.data.transforms module"""

import numpy as np
import pytest
import torch

from direct.data import transforms
from direct.data.transforms import tensor_to_complex_numpy


def create_input(shape):
    data = np.random.randn(*shape).copy()
    data = torch.from_numpy(data).float()

    return data


@pytest.mark.parametrize(
    "shape, dim",
    [
        [[3, 3], (0, 1)],
        [[4, 6], (0, 1)],
        [[10, 8, 4], (1, 2)],
        [[3, 4, 2, 2], (2, 3)],
    ],
)
def test_fft2(shape, dim):
    shape = shape + [2]
    data = create_input(shape)

    out_torch = transforms.fft2(data, dim=dim).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    data_numpy = tensor_to_complex_numpy(data)
    data_numpy = np.fft.ifftshift(data_numpy, dim)
    out_numpy = np.fft.fft2(data_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, dim)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape, dim",
    [
        [[3, 3], (0, 1)],
        [[4, 6], (0, 1)],
        [[10, 8, 4], (1, 2)],
        [[3, 4, 2, 2], (2, 3)],
    ],
)
def test_ifft2(shape, dim):
    shape = shape + [2]
    data = create_input(shape)

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
    "shape",
    [
        [3, 3],
        [4, 6],
        [10, 8, 4],
        [3, 4, 3, 5],
    ],
)
@pytest.mark.parametrize("complex", [True, False])
def test_modulus_if_complex(shape, complex):
    if complex:
        shape += [
            2,
        ]
    test_modulus(shape)


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


@pytest.mark.parametrize(
    "shape, dim",
    [
        [[3, 4, 5], 0],
        [[3, 3, 4, 5], 1],
        [[3, 6, 4, 5], 0],
        [[3, 3, 6, 4, 5], 1],
    ],
)
def test_expand_operator(shape, dim):
    shape = shape + [
        2,
    ]
    data = create_input(shape)  # noqa
    shape = shape[:dim] + shape[dim + 1 :]
    sens = create_input(shape)  # noqa

    out_torch = tensor_to_complex_numpy(transforms.expand_operator(data, sens, dim))

    input_numpy = np.expand_dims(tensor_to_complex_numpy(data), dim)
    input_sens_numpy = tensor_to_complex_numpy(sens)
    out_numpy = input_sens_numpy * input_numpy

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape, dim",
    [
        [[3, 4, 5], 0],
        [[3, 3, 4, 5], 1],
        [[3, 6, 4, 5], 0],
        [[3, 3, 6, 4, 5], 1],
    ],
)
def test_reduce_operator(shape, dim):
    shape = shape + [
        2,
    ]
    coil_data = create_input(shape)  # noqa
    sens = create_input(shape)  # noqa
    out_torch = tensor_to_complex_numpy(transforms.reduce_operator(coil_data, sens, dim))

    input_numpy = tensor_to_complex_numpy(coil_data)
    input_sens_numpy = tensor_to_complex_numpy(sens)
    out_numpy = (input_sens_numpy.conj() * input_numpy).sum(dim)

    assert np.allclose(out_torch, out_numpy)
