# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

import numpy as np
import torch

from typing import Union, Optional, List, Tuple, Callable, Any

from direct.utils import is_power_of_two, ensure_list


def assert_complex(data: torch.Tensor, enforce_named: bool = False, complex_last: bool = True) -> None:
    """
    Assert if a tensor is a complex named tensor.

    Parameters
    ----------
    data : torch.Tensor
    enforce_named : bool
        If true, will not only check if a possible complex dimension satisfies the requirements, but additionally if
        the complex dimension is there and on the last axis.
    complex_last : bool
        If true, will require complex axis to be at the last axis.

    Returns
    -------

    """
    # TODO: This is because ifft and fft or torch expect the last dimension to represent the complex axis.
    if complex_last and data.size(-1) != 2:
        raise ValueError(f'Last dimension assumed to be 2 (complex valued). Got {data.size(-1)}.')

    if 'complex' in data.names and not data.size('complex') == 2:
        raise ValueError(f"Named dimension 'complex' has to be size 2.")

    if enforce_named:
        if complex_last and data.names[-1] != 'complex':
            raise ValueError(f"Named dimension 'complex' missing, or not at the last axis. Got {data.names}.")
        else:
            if 'complex' not in data.names:
                raise ValueError(f"Named dimension 'complex' missing. Got {data.names}.")


# TODO: Allow arbitrary list of inputs.
def assert_named(data: torch.Tensor):
    """
    Ensure tensor is named (at least one dimension name is not None).

    Parameters
    ----------
    data : torch.Tensor
    """

    if all([_ is None for _ in data.names]):
        raise ValueError(f'Expected `data` to be named.')


def to_tensor(data: np.ndarray, names: Optional[Union[List[Any], Tuple[Any, ...]]] = None) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor. Complex arrays will have real and imaginary parts on the last axis.

    Parameters
    ----------
    data : np.ndarray
    names : tuple or list

    Returns
    -------
    torch.Tensor

    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
        if not names:
            names = [None] * (data.ndim - 1) + ['complex']  # type: ignore
        else:
            names = list(names) + ['complex']

    data = torch.from_numpy(data)
    if names:
        data = data.refine_names(*names)

    return data


def verify_fft_dtype_possible(data: torch.Tensor, dims: Tuple[Union[str, int], ...]) -> bool:
    """
    Fft and ifft can only be performed on GPU in float16 if the shapes are powers of 2.
    This function verifies if this is the case.

    Parameters
    ----------
    data : torch.Tensor
    dims : tuple

    Returns
    -------
    bool
    """

    return (data.dtype == torch.float16) and all([is_power_of_two(_) for _ in [data.size(idx) for idx in dims]])


def fft2(data: torch.Tensor, dim: Tuple[str, ...] = ('height', 'width'),
         centered: bool = True, normalized: bool = True) -> torch.Tensor:
    """
    Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when
    input shapes are powers of two.

    Parameters
    ----------
    data : torch.Tensor
        Complex-valued input tensor.
    dim : tuple, list or int
        Dimensions over which to compute.
    centered : bool
        Whether to apply a centered fft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized : bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.

    Returns
    -------
    torch.Tensor: the fft of the output.
    """
    assert_complex(data)

    if centered:
        data = ifftshift(data, dim=dim)
    names = data.names

    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft(data.rename(None), 2, normalized=normalized)
    else:
        data = torch.fft(data.rename(None).float(), 2, normalized=normalized).type(data.type())

    if any(names):
        data = data.refine_names(*names)  # typing: ignore

    if centered:
        data = fftshift(data, dim=dim)
    return data


def fft2_uncentered(data: torch.Tensor, dim: Tuple[str, ...] = ('height', 'width')) -> torch.Tensor:
    return fft2(data, dim, centered=False)


def ifft2(data: torch.Tensor, dim: Tuple[str, ...] = ('height', 'width'),
          centered: bool = True, normalized: bool = True) -> torch.Tensor:
    """
    Apply centered two-dimensional Inverse Fast Fourier Transform

    Parameters
    ----------
    data : torch.Tensor
        Complex-valued input tensor.
    dim : tuple, list or int
        Dimensions over which to compute.
    centered : bool
        Whether to apply a centered ifft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized : bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.

    Returns
    -------
    torch.Tensor: the ifft of the output.
    """
    assert_complex(data)

    if centered:
        data = ifftshift(data, dim=dim)
    names = data.names
    # TODO: Fix when ifft supports named tensors
    # Verify whether half precision and if ifft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.ifft(data.rename(None), 2, normalized=normalized)
    else:
        data = torch.ifft(data.rename(None).float(), 2, normalized=normalized).type(data.type())

    if any(names):
        data = data.refine_names(*names)

    if centered:
        data = fftshift(data, dim=dim)
    return data


def ifft2_uncentered(data: torch.Tensor, dim: Tuple[str, ...] = ('height', 'width')) -> torch.Tensor:
    return ifft2(data, dim, centered=False)


def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Divide a and b safely, set the output to zero where the divisor b is zero.

    Parameters
    ----------
    a : torch.Tensor
    b : torch.Tensor

    Returns
    -------
    torch.Tensor: the division.

    """
    assert_named(a)
    assert_named(b)

    b = b.align_as(a)
    data = torch.where(
        b.rename(None) == 0,
        torch.tensor([0.], dtype=a.dtype).to(a.device),
        (a / b).rename(None)).refine_names(*a.names)
    return data


def modulus(data: torch.Tensor) -> torch.Tensor:
    """
    Compute modulus of complex input data. Assumes there is a dimension called "complex" in the data.

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor: modulus of data.
    """
    assert_complex(data, enforce_named=True, complex_last=False)
    # TODO: Named tensors typing not yet fully supported in pytorch.
    return (data ** 2).sum('complex').sqrt() # noqa


def modulus_if_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Compute modulus if complex-valued.

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    # TODO: This can be merged with modulus if the tensor is real.
    assert_named(data)
    if 'complex' in data.names:
        return modulus(data)
    else:
        return data


def roll(data: torch.Tensor,
         shift: Union[int, Union[Tuple[int, ...], List[int]]],
         dims: Union[str, int, Union[Tuple, List]]) -> torch.Tensor:
    """
    Similar to numpy roll but applies to (named) pytorch tensors.
    """
    if isinstance(shift, (tuple, list)):
        if len(shift) != len(dims):
            raise ValueError(f'Length of shifts and dimensions should be equal. Got {len(shift)} and {len(dims)}.')
        for curr_shift, curr_dim in zip(shift, dims):
            data = roll(data, curr_shift, curr_dim)
        return data
    dim_index = data.names.index(dims) if isinstance(dims, str) else dims
    shift = shift % data.size(dims)

    if shift == 0:
        return data
    left_part = data.narrow(dim_index, 0, data.size(dims) - shift)
    right_part = data.narrow(dim_index, data.size(dims) - shift, shift)
    return torch.cat([right_part, left_part], dim=dim_index)


def fftshift(data: torch.Tensor, dim: Tuple[Union[str, int], ...] = None) -> torch.Tensor:
    """
    Similar to numpy fftshift but applies to (named) pytorch tensors.

    Parameters
    ----------
    data : torch.Tensor
    dim : tuple, list or int

    Returns
    -------
    torch.Tensor

    """
    if dim is None:
        dim = tuple(range(data.dim()))

    if isinstance(dim, int):
        dim = [dim]

    shift = [data.size(curr_dim) // 2 for curr_dim in dim]
    return roll(data, shift, dim)


def ifftshift(data: torch.Tensor, dim: Tuple[Union[str, int], ...] = None) -> torch.Tensor:
    """
    Similar to numpy ifftshift but applies to (named) pytorch tensors.

    Parameters
    ----------
    data : torch.Tensor
    dim : tuple, list or int

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        dim = tuple(range(data.dim()))
        shift = [(dim + 1) // 2 for dim in data.shape]
    elif isinstance(dim, int):
        shift = (data.shape[dim] + 1) // 2
    else:
        shift = [(data.size(curr_dim) + 1) // 2 for curr_dim in dim]
    return roll(data, shift, dim)


def complex_multiplication(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Multiplies two complex-valued tensors. Assumes the tensor has a named dimension "complex".

    Parameters
    ----------
    x : torch.Tensor
        Input data
    y : torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert_complex(x, enforce_named=True)
    assert_complex(y, enforce_named=True)

    # TODO: Unsqueezing is not yet supported for named tensors, fix when it is.
    complex_index = x.names.index('complex')

    real_part = x.select('complex', 0) * y.select('complex', 0) - x.select('complex', 1) * y.select('complex', 1)
    imaginary_part = x.select('complex', 0) * y.select('complex', 1) + x.select('complex', 1) * y.select('complex', 0)

    real_part = real_part.rename(None)
    imaginary_part = imaginary_part.rename(None)

    multiplication = torch.cat(
        [real_part.unsqueeze(dim=complex_index), imaginary_part.unsqueeze(dim=complex_index)], dim=complex_index)

    return multiplication.refine_names(*x.names)


def conjugate(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the complex conjugate of a torch tensor where the last axis denotes the real and complex part.

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    assert_complex(data, enforce_named=True)
    names = data.names
    data = data.rename(None).clone()  # Clone is required as the data in the next line is changed in-place.
    data[..., 1] = data[..., 1] * -1.0
    data = data.refine_names(*names)
    return data


def apply_mask(
        kspace: torch.Tensor,
        mask_func: Union[Callable, torch.Tensor],
        seed: Optional[int] = None, return_mask: bool = True) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Subsample kspace by setting kspace to zero as given by a binary mask.

    Parameters
    ----------
    kspace : torch.Tensor
        k-space as a complex-valued tensor.
    mask_func : callable or torch.tensor
        Masking function, taking a shape and returning a mask with this shape or can be broadcasted as such
        Can also be a sampling mask.
    seed : int
        Seed for the random number generator
    return_mask : bool
        If true, mask will be returned

    Returns
    -------
    masked data (torch.Tensor), mask (torch.Tensor)
    """
    # TODO: Split the function to apply_mask_func and apply_mask

    assert_complex(kspace, enforce_named=True)
    names = kspace.names
    kspace = kspace.rename(None)

    if not isinstance(mask_func, torch.Tensor):
        shape = np.array(kspace.shape)[1:]  # The first dimension is always the coil dimension.
        mask = mask_func(shape, seed)
    else:
        mask = mask_func

    masked_kspace = torch.where(mask == 0, torch.tensor([0.], dtype=kspace.dtype), kspace)

    mask = mask.refine_names(*names)
    masked_kspace = masked_kspace.refine_names(*names)
    if not return_mask:
        return masked_kspace

    return masked_kspace, mask


def tensor_to_complex_numpy(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex pytorch tensor to a complex numpy array.
    The last axis denote the real and imaginary parts respectively.

    Parameters
    ----------
    data : torch.Tensor
        Input data

    Returns
    -------
    Complex valued np.ndarray
    """
    assert_complex(data)
    data = data.detach().cpu().numpy()
    return data[..., 0] + 1j * data[..., 1]


def root_sum_of_squares(data: torch.Tensor, dim: Union[int, str] = 'coil') -> torch.Tensor:
    """
    Compute the root sum of squares (RSS) transform along a given (perhaps named) dimension of the input tensor.

    $$x_{\textrm{rss}} = \sqrt{\sum_{i \in \textrm{coil}} |x_i|^2}$$

    Parameters
    ----------
    data : torch.Tensor
        Input tensor

    dim : Union[int, str]
        Coil dimension.

    Returns
    -------
    torch.Tensor : RSS of the input tensor.
    """

    if 'complex' in data.names:
        assert_complex(data, complex_last=True)
        return torch.sqrt((data ** 2).sum('complex').sum(dim))
    else:
        return torch.sqrt((data ** 2).sum(dim))


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop along the last two dimensions.

    Parameters
    ----------
    data : torch.Tensor
    shape : Tuple[int, int]
        The output shape, should be smaller than the corresponding data dimensions.

    Returns
    -------
    torch.Tensor : The center cropped data.
    """
    # TODO: Make dimension independent.
    if not (0 < shape[0] <= data.shape[-2]) or not (0 < shape[1] <= data.shape[-1]):
        raise ValueError(f'Crop shape should be smaller than image.')

    width_lower = (data.shape[-2] - shape[0]) // 2
    width_upper = width_lower + shape[0]
    height_lower = (data.shape[-1] - shape[1]) // 2
    height_upper = height_lower + shape[1]
    return data[..., width_lower:width_upper, height_lower:height_upper]


def complex_center_crop(data_list, shape, didx=-3, contiguous=False):
    """
    Apply a center crop to the input image, or to a list of complex images


    Parameters
    ----------
    data_list : List[torch.Tensor] or torch.Tensor
        The complex input tensor to be center cropped. It should have at least 3 dimensions
         and the cropping is applied along dimensions didx and didx+1 and the last dimensions should have a size of 2.
    shape : Tuple[int, int]
        The output shape. The shape should be smaller than the corresponding dimensions of data.
    didx : int
        Starting dimension for cropping.
    contiguous : bool
        Return as a contiguous array. Useful for fast reshaping or viewing.

    Returns
    -------
    torch.Tensor or list[torch.Tensor]: The center cropped input_image

    """
    data_list = ensure_list(data_list)
    for data in data_list:
        assert didx in [-3, -2], 'Cropping needs to be done in the spatial dimensions.'
        assert 0 < shape[0] <= data.shape[didx]
        assert 0 < shape[1] <= data.shape[didx + 1]

    w_from = (data.shape[didx] - shape[0]) // 2
    h_from = (data.shape[didx + 1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    if didx == -3:
        output = [data[..., w_from:w_to, h_from:h_to, :] for data in data_list]
    else:
        output = [data[..., w_from:w_to, h_from:h_to] for data in data_list]

    if contiguous:
        output = [_.contiguous() for _ in output]

    if len(output) == 1:  # Only one element:
        output = output[0]
    return output


def complex_random_crop(data_list, shape, contiguous=False):
    """
    Apply a random crop to the input data tensor or a list of complex.

    Parameters
    ----------
    data_list : Union[List[torch.Tensor], torch.Tensor]
        The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is applied
        along dimensions -3 and -2 and the last dimensions should have a size of 2.
    shape : Tuple[int, int]
        The output shape. The shape should be smaller than the corresponding dimensions of data.
    contiguous : bool
            Return as a contiguous array. Useful for fast reshaping or viewing.

    Returns
    -------
    torch.Tensor: The center cropped input tensor or list of tensors

    """
    data_list = ensure_list(data_list)

    # TODO: Check if all have same shape
    for data in data_list:
        assert 0 < shape[0] <= data.shape[-3]
        assert 0 < shape[1] <= data.shape[-2]

    w_from = np.random.randint(0, data_list[0].shape[-3] - shape[0] + 1)
    h_from = np.random.randint(0, data_list[0].shape[-2] - shape[1] + 1)

    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    output = [data[..., w_from:w_to, h_from:h_to, :] for data in data_list]

    if contiguous:
        output = [_.contiguous() for _ in output]

    if len(output) == 1:
        return output[0]

    return output
