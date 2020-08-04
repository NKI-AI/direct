# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch

from direct.nn.rim.mri_models import MRILogLikelihood
from direct.data import transforms


def numpy_fft(data_numpy):
    data_numpy = np.fft.ifftshift(data_numpy, (-2, -1))
    out_numpy = np.fft.fft2(data_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    return out_numpy


def numpy_ifft(input_numpy):
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    return out_numpy


def create_input(shape, named=True):
    data = np.arange(np.product(shape)).reshape(shape).copy()
    data = torch.from_numpy(data).float()
    if named:
        data = add_names(data, named)
    return data


def add_names(tensor, named=True):
    shape = tensor.shape

    if len(shape) == 2:
        names = ("height", "width")
    elif len(shape) == 3:
        names = ("height", "width", "complex")
    elif len(shape) == 4:
        names = ("coil", "height", "width", "complex")
    else:
        names = ("batch", "coil", "height", "width", "complex")

    if named:
        tensor = tensor.refine_names(*names)

    return tensor


#
# def test_mri_log_likelihood():
#     log_likelihood = MRILogLikelihood(forward_operator=transforms.fft2, backward_operator=transforms.ifft2)
#     # Generate unit sensitivity.
#     sensitivity_map = transforms.to_tensor(np.zeros((15, 320, 480), dtype=np.complex64))
#     mask = transforms.to_tensor(np.ones((15, 320, 320, 1)))
#     masked_kspace = create_input((15, 320, 480, 2), named=True)
#
#     target = transforms.ifft2(input_image)
#
#     log_likelihood(input_image, input_image)

import numpy as np
import torch

from direct.data.transforms import tensor_to_complex_numpy
from direct.nn.rim.mri_models import MRILogLikelihood
from direct.data import transforms


input_image = create_input([1, 4, 4, 2]).rename("batch", "height", "width", "complex")
sensitivity_map = create_input([1, 15, 4, 4, 2]) * 0.1
masked_kspace = create_input([1, 15, 4, 4, 2]) + 0.33
sampling_mask = torch.from_numpy(
    np.random.binomial(size=(1, 1, 4, 4, 1), n=1, p=0.5)
).refine_names(*sensitivity_map.names)

input_image_numpy = tensor_to_complex_numpy(input_image)
sensitivity_map_numpy = tensor_to_complex_numpy(sensitivity_map)
masked_kspace_numpy = tensor_to_complex_numpy(masked_kspace)
sampling_mask_numpy = sampling_mask.numpy()[..., 0]

# Torch
input_image = input_image.align_to("batch", "height", "width", "complex")
sensitivity_map = sensitivity_map.align_to(
    "batch", "coil", "height", "width", "complex"
)
masked_kspace = masked_kspace.align_to("batch", "coil", "height", "width", "complex")

mul = transforms.complex_multiplication(
    sensitivity_map, input_image.align_as(sensitivity_map)
)

mul_names = mul.names
mr_forward = torch.where(
    sampling_mask.rename(None) == 0,
    torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
    transforms.fft2(mul).rename(None),
)

error = mr_forward - torch.where(
    sampling_mask.rename(None) == 0,
    torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
    masked_kspace.rename(None),
)
error = error.refine_names(*mul_names)

mr_backward = transforms.ifft2(error)

out = transforms.complex_multiplication(
    transforms.conjugate(sensitivity_map), mr_backward
).sum("coil")


# numpy
# mul_numpy = sensitivity_map_numpy * input_image_numpy
# mr_forward_numpy = sampling_mask_numpy * numpy_fft(mul_numpy)
# error_numpy = mr_forward_numpy - sampling_mask_numpy * masked_kspace_numpy
# mr_backward_numpy = numpy_ifft(error_numpy)
# out_numpy = (sensitivity_map_numpy.conjugate() * mr_backward_numpy).sum(1)

# np.allclose(tensor_to_complex_numpy(out), out_numpy)

# numpy 2
mr_backward_numpy = numpy_ifft(
    sampling_mask_numpy
    * numpy_fft(sensitivity_map_numpy * input_image_numpy[:, np.newaxis, ...])
    - sampling_mask_numpy * masked_kspace_numpy
)
out_numpy = (sensitivity_map_numpy.conjugate() * mr_backward_numpy).sum(1)

np.allclose(tensor_to_complex_numpy(out), out_numpy)
