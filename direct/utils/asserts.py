# coding=utf-8
# Copyright (c) DIRECT Contributors
import inspect
from typing import List

import torch

from direct.utils import is_complex_data


def assert_positive_integer(*variables, strict: bool = False) -> None:
    """Assert if given variables are positive integer.

    Parameters
    ----------
    variables: Any
    strict: bool
        If true, will allow zero values.
    """
    if not strict:
        type_name = "positive integer"
    else:
        type_name = "positive integer larger than zero"

    for variable in variables:
        if not isinstance(variable, int) or (variable <= 0 and strict) or (variable < 0 and not strict):
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()  # type: ignore
            variable_name = [var_name for var_name, var_val in callers_local_vars if var_val is variable][0]

            raise ValueError(f"{variable_name} has to be a {type_name}. " f"Got {variable} of type {type(variable)}.")


def assert_same_shape(data_list: List[torch.Tensor]):
    """Check if all tensors in the list have the same shape.

    Parameters
    ----------
    data_list: list
        List of tensors
    """
    shape_list = set(_.shape for _ in data_list)
    if not len(shape_list) == 1:
        raise ValueError(f"All inputs are expected to have the same shape. Got {shape_list}.")


def assert_complex(data: torch.Tensor, complex_last: bool = True) -> None:
    """Assert if a tensor is a complex tensor.

    Parameters
    ----------
    data: torch.Tensor
        For 2D data the shape is assumed ([batch], [coil], height, width, [complex])
            or ([batch], [coil], [complex], height, width).
        For 3D data the shape is assumed ([batch], [coil], slice, height, width, [complex])
            or ([batch], [coil], [complex], slice, height, width).
    complex_last: bool
        If true, will require complex axis to be at the last axis.
    Returns
    -------
    """
    # TODO: This is because ifft and fft or torch expect the last dimension to represent the complex axis.
    if not is_complex_data(data, complex_last):
        raise ValueError(f"Complex dimension assumed to be 2 (complex valued), but not found in shape {data.shape}.")
