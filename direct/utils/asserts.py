# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import inspect

from typing import List


def assert_positive_integer(*variables, strict: bool = False) -> None:
    """
    Assert if a variable if a positive integer.

    Parameters
    ----------
    variables : Any
    strict : bool
        If true, will allow zero values
    """
    for variable in variables:
        if (
            not isinstance(variable, int)
            or (variable <= 0 and strict)
            or (variable < 0 and not strict)
        ):
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            variable_name = [
                var_name
                for var_name, var_val in callers_local_vars
                if var_val is variable
            ][0]
            raise ValueError(
                f"{variable_name} has to be a positive integer larger than zero. "
                f"Got {variable} of type {type(variable)}."
            )


def assert_same_shape(data_list: List[torch.tensor]):
    """
    Check if all tensors in the list have the same shape.

    Parameters
    ----------
    data_list : list
        List of tensors
    """
    shape_list = set([_.shape for _ in data_list])
    if not len(shape_list) == 1:
        raise ValueError(
            f"complex_center_crop expects all inputs to have the same shape. Got {shape_list}."
        )


def assert_complex(
    data: torch.Tensor, enforce_named: bool = False, complex_last: bool = True
) -> None:
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
        raise ValueError(
            f"Last dimension assumed to be 2 (complex valued). Got {data.size(-1)}."
        )

    if "complex" in data.names and not data.size("complex") == 2:
        raise ValueError(f"Named dimension 'complex' has to be size 2.")

    if enforce_named:
        if complex_last and data.names[-1] != "complex":
            raise ValueError(
                f"Named dimension 'complex' missing, or not at the last axis. Got {data.names}."
            )
        else:
            if "complex" not in data.names:
                raise ValueError(
                    f"Named dimension 'complex' missing. Got {data.names}."
                )


# TODO(jt): Allow arbitrary list of inputs.
def assert_named(data: torch.Tensor):
    """
    Ensure tensor is named (at least one dimension name is not None).

    Parameters
    ----------
    data : torch.Tensor
    """

    if all([_ is None for _ in data.names]):
        raise ValueError(f"Expected `data` to be named. Got {data.names}.")
