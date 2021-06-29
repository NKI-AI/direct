# coding=utf-8
# Copyright (c) DIRECT Contributors

import warnings

import numpy as np
import torch

from direct.data import transforms as T
from direct.data.transforms import tensor_to_complex_numpy

warnings.filterwarnings("ignore")


def numpy_fft(data, dims=(-2, -1)):
    """
    Fast Fourier Transform.
    """
    data = np.fft.ifftshift(data, dims)
    out = np.fft.fft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def numpy_ifft(data, dims=(-2, -1)):
    """
    Inverse Fast Fourier Transform.
    """
    data = np.fft.ifftshift(data, dims)
    out = np.fft.ifft2(data, norm="ortho")
    out = np.fft.fftshift(out, dims)

    return out


def create_input(shape):
    data = np.arange(np.product(shape)).reshape(shape).copy()
    data = torch.from_numpy(data).float()

    return data


batch, coil, height, width, complex = 3, 15, 100, 80, 2

input_image = create_input([batch, height, width, complex])
sensitivity_map = create_input([batch, coil, height, width, complex]) * 0.1
masked_kspace = create_input([batch, coil, height, width, complex]) + 0.33
sampling_mask = torch.from_numpy(np.random.binomial(size=(batch, 1, height, width, 1), n=1, p=0.5))

input_image_numpy = tensor_to_complex_numpy(input_image)
sensitivity_map_numpy = tensor_to_complex_numpy(sensitivity_map)
masked_kspace_numpy = tensor_to_complex_numpy(masked_kspace)
sampling_mask_numpy = sampling_mask.numpy()[..., 0]

mul = T.complex_multiplication(sensitivity_map, input_image.unsqueeze(1))

dims = (2, 3)
mr_forward = torch.where(
    sampling_mask == 0,
    torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
    T.fft2(mul, dim=dims),
)

error = mr_forward - torch.where(
    sampling_mask == 0,
    torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
    masked_kspace,
)

mr_backward = T.ifft2(error, dim=dims)

coil_dim = 1
out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum(coil_dim)


# numpy
mul_numpy = sensitivity_map_numpy * input_image_numpy.reshape(batch, 1, height, width)
mr_forward_numpy = sampling_mask_numpy * numpy_fft(mul_numpy)
error_numpy = mr_forward_numpy - sampling_mask_numpy * masked_kspace_numpy
mr_backward_numpy = numpy_ifft(error_numpy)
out_numpy = (sensitivity_map_numpy.conjugate() * mr_backward_numpy).sum(1)

np.allclose(tensor_to_complex_numpy(out), out_numpy)

# numpy 2
mr_backward_numpy = numpy_ifft(
    sampling_mask_numpy * numpy_fft(sensitivity_map_numpy * input_image_numpy[:, np.newaxis, ...])
    - sampling_mask_numpy * masked_kspace_numpy
)
out_numpy = (sensitivity_map_numpy.conjugate() * mr_backward_numpy).sum(1)

np.allclose(tensor_to_complex_numpy(out), out_numpy)
