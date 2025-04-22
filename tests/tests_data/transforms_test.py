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
"""Tests for the direct.data.transforms module."""

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
        [[3, 3], (-2, -1)],
        [[4, 6], (0, 1)],
        [[10, 8, 4], (1, 2)],
        [[3, 4, 2, 2], (2, 3)],
    ],
)
def test_ifft2(shape, dim):
    shape = shape + [2]
    data = create_input(shape)
    if any(_ < 0 for _ in dim):
        with pytest.raises(TypeError):
            out_torch = transforms.ifft2(data, dim=dim).numpy()
    else:
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
        [
            3,
            4,
            8,
        ],
        [3, 4, 8, 5],
    ],
)
@pytest.mark.parametrize("complex_axis", [0, 1, 2, -1, None])
def test_modulus_if_complex(shape, complex_axis):
    if complex_axis is not None:
        if complex_axis != -1:
            shape = (
                shape[:complex_axis]
                + [
                    2,
                ]
                + shape[complex_axis:]
            )
        else:
            shape.append(2)
    data = create_input(shape)
    if complex_axis is not None:
        data = transforms.modulus_if_complex(data, complex_axis)
        shape.pop(complex_axis)
    else:
        data = transforms.modulus_if_complex(data)
        print(data.shape)

    assert list(data.shape) == shape


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
        ((0,), (0,)),
        ((1,), (0,)),
        ((-1,), (0,)),
        ((100,), (0,)),
        ((1, 2), (1,)),
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
    data = np.arange(np.prod(shape)).reshape(shape)
    torch_tensor = torch.from_numpy(data)
    if not isinstance(shift, int) and not isinstance(dims, int) and len(shift) != len(dims):
        with pytest.raises(ValueError):
            out_torch = transforms.roll(torch_tensor, shift, dims).numpy()
    else:
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
    data_0 = np.arange(np.prod(shape)).reshape(shape) + 1j * (np.arange(np.prod(shape)).reshape(shape) + 1)
    data_1 = data_0 + 0.5 + 1j
    torch_tensor_0 = transforms.to_tensor(data_0)
    torch_tensor_1 = transforms.to_tensor(data_1)

    out_torch = tensor_to_complex_numpy(transforms.complex_multiplication(torch_tensor_0, torch_tensor_1))
    out_numpy = data_0 * data_1
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shape",
    [[3, 7], [5, 6, 2], [3, 4, 5], [4, 20, 42], [3, 4, 20, 40]],
)
def test_complex_division(shape):
    data_0 = np.arange(np.prod(shape)).reshape(shape) + 1j * (np.arange(np.prod(shape)).reshape(shape) + 1)
    data_1 = np.arange(np.prod(shape)).reshape(shape) + 1j * (np.arange(np.prod(shape)).reshape(shape) + 1)
    torch_tensor_0 = transforms.to_tensor(data_0)
    torch_tensor_1 = transforms.to_tensor(data_1)
    out_torch = tensor_to_complex_numpy(transforms.complex_division(torch_tensor_0, torch_tensor_1))
    out_numpy = data_0 / data_1
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize(
    "shapes",
    [
        [[3, 7], [7, 4]],
        [[5, 6], [6, 3]],
    ],
)
@pytest.mark.parametrize(
    "is_first_complex",
    [True, False],
)
def test_complex_matrix_multiplication(shapes, is_first_complex):
    data_1 = torch.randn(*shapes[1]) + 1.0j * torch.randn(*shapes[1])
    mult_func = lambda x, y: x @ y
    if not is_first_complex:
        data_0 = torch.randn(*shapes[0])
        with pytest.raises(ValueError):
            out = transforms._complex_matrix_multiplication(data_0, data_1, mult_func)
    else:
        data_0 = torch.randn(*shapes[0]) + 1.0j * torch.randn(*shapes[0])
        out = transforms._complex_matrix_multiplication(data_0, data_1, mult_func)
        out_torch = torch.view_as_complex(
            torch.stack(
                (
                    data_0.real @ data_1.real - data_0.imag @ data_1.imag,
                    data_0.real @ data_1.imag + data_0.imag @ data_1.real,
                ),
                dim=2,
            )
        )
        assert torch.allclose(out, out_torch)


@pytest.mark.parametrize(
    "shapes",
    [
        [[3, 7], [7, 4]],
        [[5, 6], [6, 3]],
    ],
)
def test_complex_matrix_multiplication(shapes):
    data_0 = torch.randn(*shapes[0]) + 1.0j * torch.randn(*shapes[0])
    data_1 = torch.randn(*shapes[1]) + 1.0j * torch.randn(*shapes[1])

    out = transforms.complex_mm(data_0, data_1)
    out_torch = torch.view_as_complex(
        torch.stack(
            (
                data_0.real @ data_1.real - data_0.imag @ data_1.imag,
                data_0.real @ data_1.imag + data_0.imag @ data_1.real,
            ),
            dim=2,
        )
    )
    assert torch.allclose(out, out_torch)


@pytest.mark.parametrize(
    "shape",
    [[3, 32, 32, 2], [4, 10, 23, 2]],
)
def test_dot_product(shape):
    a = create_input(shape)
    b = create_input(shape)
    direct_dot = torch.view_as_complex(transforms.complex_dot_product(a, b, (1, 2)))
    torch_dot = (torch.view_as_complex(a).conj() * torch.view_as_complex(b)).sum((1, 2))

    assert torch.allclose(direct_dot, torch_dot)


@pytest.mark.parametrize(
    "shapes",
    [
        [[3, 7], [7, 4]],
        [[5, 6], [6, 3]],
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [3, 4],
)
def test_complex_bmm(shapes, batch_size):
    data_0 = torch.randn(batch_size, *shapes[0]) + 1.0j * torch.randn(batch_size, *shapes[0])
    data_1 = torch.randn(batch_size, *shapes[1]) + 1.0j * torch.randn(batch_size, *shapes[1])

    out = transforms.complex_bmm(data_0, data_1)
    out_torch = torch.stack(
        [
            torch.view_as_complex(
                torch.stack(
                    (
                        data_0[_].real @ data_1[_].real - data_0[_].imag @ data_1[_].imag,
                        data_0[_].real @ data_1[_].imag + data_0[_].imag @ data_1[_].real,
                    ),
                    dim=2,
                )
            )
            for _ in range(batch_size)
        ],
        dim=0,
    )
    assert torch.allclose(out, out_torch)


@pytest.mark.parametrize(
    "shape",
    [
        [3, 7],
        [5, 6, 2],
        [3, 4, 5],
    ],
)
def test_conjugate(shape):
    data = np.arange(np.prod(shape)).reshape(shape) + 1j * (np.arange(np.prod(shape)).reshape(shape) + 1)
    torch_tensor = transforms.to_tensor(data)

    out_torch = tensor_to_complex_numpy(transforms.conjugate(torch_tensor))
    out_numpy = np.conjugate(data)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize("shape", [[5, 3], [2, 4, 6], [2, 11, 4, 7]])
def test_fftshift(shape):
    data = np.arange(np.prod(shape)).reshape(shape)
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
    data = np.arange(np.prod(shape)).reshape(shape)
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


@pytest.mark.parametrize(
    "shapes, crop_shape, sampler, sigma, expect_error",
    [
        [[[2, 7, 7]] * 2, (8, 8), "uniform", None, True],
        [[[2, 7, 7]] * 2, (4, 3), "uniform", None, False],
        [[[2, 7, 7]] * 2, (4, 3), "uniform", 0.2, True],
        [[[2, 7, 7]] * 2, (4, 3), "gaussian", 0.2, False],
        [[[2, 7, 7]] * 2, (4, 3), "gaussian", [0.1, 0.2], False],
        [[[2, 7, 7], [3, 8, 6]], (4, 3), "gaussian", 0.2, True],
        [[[2, 7, 7]] * 2, (4, 3), "invalid_sampler", False, True],
        [[[3, 4, 8, 6]] * 2, (2, 6, 4), "uniform", None, False],
        [[[3, 4, 8, 6]] * 2, (2, 6, 4), "uniform", 0.2, True],
        [[[3, 4, 8, 6]] * 2, (2, 6, 4), "gaussian", 0.2, False],
        [[[3, 4, 8, 6]] * 2, (2, 6, 4), "gaussian", [0.1, 0.2], True],
        [[[3, 4, 8, 6]] * 2, (2, 6, 4), "invalid_sampler", False, True],
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
)
def test_complex_random_crop(shapes, crop_shape, sampler, sigma, expect_error, contiguous):
    data_list = [create_input(shape + [2]) for shape in shapes]
    if expect_error:
        with pytest.raises(ValueError):
            samples = transforms.complex_random_crop(
                data_list, crop_shape, sampler=sampler, sigma=sigma, contiguous=contiguous
            )
    else:
        data_list = transforms.complex_random_crop(
            data_list, crop_shape, sampler=sampler, sigma=sigma, contiguous=contiguous
        )
        assert all(data_list[i].shape == (shapes[i][0],) + crop_shape + (2,) for i in range(len(data_list)))
        if contiguous:
            assert all(data.is_contiguous() for data in data_list)


@pytest.mark.parametrize(
    "shape, crop_shape",
    [
        [[3, 7, 9], [4, 5]],
        [[3, 6, 6], [4, 4]],
        [[3, 6, 6, 7], [3, 4, 4]],
        [[3, 8, 6, 8], [3, 4, 4]],
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
)
def test_complex_center_crop(shape, crop_shape, contiguous):
    data_list = [create_input(shape + [2]) for _ in range(np.random.randint(2, 5))]
    data_list = transforms.complex_center_crop(data_list, crop_shape, contiguous=contiguous)
    assert all(data.shape == tuple([data.shape[0]] + crop_shape + [2]) for data in data_list)
    if contiguous:
        assert all(data.is_contiguous() for data in data_list)


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10, 20, 22],
        [1, 10, 20, 22],
    ],
)
def test_apply_padding(shape):
    data = create_input(shape + [2])
    padding = torch.from_numpy(np.random.randn(shape[0], 1, shape[-2], shape[-1], 1)).round().bool()
    padded_data = transforms.apply_padding(data, padding)

    assert torch.allclose(data * (~padding), padded_data)


@pytest.mark.parametrize(
    "input_shape, resize_shape, mode",
    [
        ((1, 6, 3, 2), (5,), "nearest"),
        ((1, 3, 6, 3, 2), (5, 5), "nearest"),
        ((1, 7, 3, 6, 3, 2), (5, 5, 5), "nearest"),
        ((1, 6, 3, 2), (5,), "area"),
        ((1, 3, 6, 3, 2), (5, 5), "area"),
        ((1, 7, 3, 6, 3, 2), (5, 5, 5), "area"),
        ((1, 6, 3, 2), (5,), "linear"),
        ((1, 3, 6, 3, 2), (5, 5), "bilinear"),
        ((1, 3, 6, 3, 2), (5, 5), "bicubic"),
        ((1, 7, 3, 6, 3, 2), (5, 5, 5), "trilinear"),
    ],
)
def test_complex_image_resize(input_shape, resize_shape, mode):
    # Create a random complex_image tensor with the specified input shape
    complex_image = torch.randn(input_shape)

    # Perform the resize operation
    resized_image = transforms.complex_image_resize(complex_image, resize_shape, mode)

    # Determine the expected shape based on the resize_shape
    expected_shape = input_shape[: -(len(resize_shape) + 1)] + tuple(resize_shape) + (2,)

    # Assert that the shape of the resized image matches the expected shape
    assert resized_image.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape, target_shape, value",
    [
        [(30, 20, 25), (40, 40), 0],
        [(30, 20, 25), (40, 40), 3],
        [(30, 20, 25), (40, 39, 28), 0],
        [(30, 20, 25), (40, 39, 28), 6],
        [(11, 30, 20, 25), (40, 39, 28), 1],
        [(11, 30, 20, 25), (40, 10, 20), 1],
    ],
)
def test_pad_tensor(input_shape, target_shape, value):
    data = torch.ones(input_shape)
    padded_data = transforms.pad_tensor(data, target_shape, value)

    expected_shape = list(input_shape[: -len(target_shape)]) + list(target_shape)
    for i in range(1, len(target_shape) + 1):
        if target_shape[-i] < input_shape[-i]:
            expected_shape[-i] = input_shape[-i]

    assert list(padded_data.shape) == expected_shape

    assert data.sum() + (value * (np.prod(expected_shape) - np.prod(input_shape))) == padded_data.sum()
