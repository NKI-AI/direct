# coding=utf-8
# Copyright (c) DIRECT Contributors
import abc
import ast
import functools
import gc
import importlib
import logging
import pathlib
import random
import subprocess
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, KeysView, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def is_complex_data(data: torch.Tensor, complex_last: bool = True) -> bool:
    """Returns True if data is a complex tensor, i.e. has a complex axis of dimension 2, and False otherwise.

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
    if 2 not in data.shape:
        return False
    if complex_last:
        if data.size(-1) != 2:
            return False
    else:
        if data.ndim == 6:
            if data.size(2) != 2 and data.size(-1) != 2:  # (B, C, 2, S, H, 2) or (B, C, S, H, W, 2)
                return False

        elif data.ndim == 5:
            # (B, 2, S, H, W) or (B, C, 2, H, W) or (B, S, H, W, 2) or (B, C, H, W, 2)
            if data.size(1) != 2 and data.size(2) != 2 and data.size(-1) != 2:
                return False

        elif data.ndim == 4:
            if data.size(1) != 2 and data.size(-1) != 2:  # (B, 2, H, W) or (B, H, W, 2) or (S, H, W, 2)
                return False

        elif data.ndim == 3:
            if data.size(-1) != 2:  # (H, W, 2)
                return False

        else:
            raise ValueError(f"Not compatible number of dimensions for complex data. Got {data.ndim}.")

    return True


def is_power_of_two(number: int) -> bool:
    """Check if input is a power of 2.

    Parameters
    ----------
    number: int

    Returns
    -------
    bool
    """
    return number != 0 and ((number & (number - 1)) == 0)


def ensure_list(data: Any) -> List:
    """Ensure input is a list.

    Parameters
    ----------
    data: object

    Returns
    -------
    list
    """
    if data is None:
        return []

    if not isinstance(data, (list, tuple)):
        return [data]

    return list(data)


def cast_as_path(data: Optional[Union[pathlib.Path, str]]) -> Optional[pathlib.Path]:
    """Ensure the the input is a path.

    Parameters
    ----------
    data: str or pathlib.Path

    Returns
    -------
    pathlib.Path
    """
    if data is None:
        return None

    return pathlib.Path(data)


def str_to_class(module_name: str, function_name: str) -> Callable:
    """Convert a string to a class Base on: https://stackoverflow.com/a/1176180/576363.

    Also support function arguments, e.g. ifft(dim=2) will be parsed as a partial and return ifft where dim has been
    set to 2.


    Examples
    --------
    >>> def mult(f, mul=2):
    >>>    return f*mul

    >>> str_to_class(".", "mult(mul=4)")
    >>> str_to_class(".", "mult(mul=4)")
    will return a function which multiplies the input times 4, while

    >>> str_to_class(".", "mult")
    just returns the function itself.

    Parameters
    ----------
    module_name: str
        e.g. direct.data.transforms
    function_name: str
        e.g. Identity
    Returns
    -------
    object
    """
    tree = ast.parse(function_name)
    func_call = tree.body[0].value  # type: ignore
    args = [ast.literal_eval(arg) for arg in func_call.args] if hasattr(func_call, "args") else []
    kwargs = (
        {arg.arg: ast.literal_eval(arg.value) for arg in func_call.keywords} if hasattr(func_call, "keywords") else {}
    )

    # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
    module = importlib.import_module(module_name)

    if not args and not kwargs:
        return getattr(module, function_name)
    return functools.partial(getattr(module, func_call.func.id), *args, **kwargs)


def dict_to_device(
    data: Dict[str, torch.Tensor],
    device: Union[torch.device, str, None],
    keys: Union[List, Tuple, KeysView, None] = None,
) -> Dict:
    """Copy tensor-valued dictionary to device. Only torch.Tensor is copied.

    Parameters
    ----------
    data: Dict[str, torch.Tensor]
    device: torch.device, str
    keys: List, Tuple
        Subselection of keys to copy.

    Returns
    -------
    Dictionary at the new device.
    """
    if keys is None:
        keys = data.keys()
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items() if k in keys}


def detach_dict(data: Dict[str, torch.Tensor], keys: Optional[Union[List, Tuple, KeysView]] = None) -> Dict:
    """Return a detached copy of a dictionary. Only torch.Tensor's are detached.

    Parameters
    ----------
    data: Dict[str, torch.Tensor]
    keys: List, Tuple
        Subselection of keys to detach

    Returns
    -------
    Dictionary at the new device.
    """
    if keys is None:
        keys = data.keys()
    return {k: v.detach() for k, v in data.items() if k in keys if isinstance(v, torch.Tensor)}


def reduce_list_of_dicts(data: List[Dict[str, torch.Tensor]], mode="average", divisor=None) -> Dict[str, torch.Tensor]:
    """Average a list of dictionary mapping keys to Tensors.

    Parameters
    ----------
    data: List[Dict[str, torch.Tensor]])
    mode: str
        Which reduction mode, average reduces the dictionary, sum just adds while average computes the average.
    divisor: None or int
        If given values are divided by this factor.

    Returns
    -------
    Dict[str, torch.Tensor]: Reduced dictionary.
    """
    if not data:
        return {}

    if mode not in ["average", "sum"]:
        raise ValueError("Reduction can only be `sum` or `average`.")

    if not divisor:
        divisor = 1.0

    result_dict = {k: torch.zeros_like(v) for k, v in data[0].items()}

    for elem in data:
        result_dict = {k: result_dict[k] + v for k, v in elem.items()}

    if mode == "average":
        divisor *= len(data)

    return {k: v / divisor for k, v in result_dict.items()}


def merge_list_of_dicts(list_of_dicts):
    """A list of dictionaries is merged into one dictionary.

    Parameters
    ----------
    list_of_dicts: List[Dict]

    Returns
    -------
    Dict
    """
    if not list_of_dicts:
        return {}

    return functools.reduce(lambda a, b: {**dict(a), **dict(b)}, list_of_dicts)


def evaluate_dict(fns_dict, source, target, reduction="mean"):
    """Evaluate a dictionary of functions.

    Examples
    --------
    > evaluate_dict({'l1_loss: F.l1_loss, 'l2_loss': F.l2_loss}, a, b)

    Will return
    > {'l1_loss', F.l1_loss(a, b, reduction='mean'), 'l2_loss': F.l2_loss(a, b, reduction='mean')

    Parameters
    ----------
    fns_dict: Dict[str, Callable]
    source: torch.Tensor
    target: torch.Tensor
    reduction: str

    Returns
    -------
    Dict[str, torch.Tensor]
    """
    return {k: fns_dict[k](source, target, reduction=reduction) for k, v in fns_dict.items()}


def prefix_dict_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Append a prefix to a dictionary keys.

    Parameters
    ----------
    data: Dict[str, Any]
    prefix: str

    Returns
    -------
    Dict[str, Any]
    """
    return {prefix + k: v for k, v in data.items()}


def git_hash() -> str:
    """Returns the current git hash.

    Returns
    -------
    str: the current git hash.
    """
    try:
        _git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE).decode().strip()
    except FileNotFoundError:
        _git_hash = "git not installed."
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        stdout = e.output.decode(sys.getfilesystemencoding())
        stderr = e.stderr.decode(sys.getfilesystemencoding())
        _git_hash = f"cannot get git hash: git returned {exit_code}\n" f"stdout: {stdout}.\n" f"stderr: {stderr}."

    return _git_hash


def normalize_image(image: torch.Tensor, eps: float = 0.00001) -> torch.Tensor:
    """Normalize image to range [0,1] for visualization.

    Parameters
    ----------
    image: torch.Tensor
    eps: float

    Returns
    -------
    torch.Tensor: scaled data.
    """

    image = image - image.min()
    image = image / (image.max() + eps)
    return image


#
# class MultiplyFunction:
#     def __init__(self, multiplier: float, func: Callable):
#         self.multiplier = multiplier
#         self._func = func
#
#     def __call__(self, *x):
#         return self.multiplier * self._func(*x)


def multiply_function(multiplier: float, func: Callable) -> Callable:
    """Create a function which multiplier another one with a multiplier.

    Parameters
    ----------
    multiplier: float
        Number to multiply with.
    func: callable
        Function to multiply.

    Returns
    -------
    Callable
    """

    def return_func(*args, **kwargs):
        return multiplier * func(*args, **kwargs)

    return return_func


class DirectTransform:
    def __init__(self):
        super().__init__()

    def __repr__(self):
        repr_string = self.__class__.__name__ + "("
        for k, v in self.__dict__.items():
            if k == "logger":
                continue
            repr_string += f"{k}="
            if callable(v):
                if hasattr(v, "__class__"):
                    repr_string += type(v).__name__ + ", "
                else:
                    # TODO(jt): better way to log functions
                    repr_string += str(v) + ", "
            elif isinstance(v, (dict, OrderedDict)):
                repr_string += f"{k}=dict(len={len(v)}), "
            elif isinstance(v, list):
                repr_string = f"{k}=list(len={len(v)}), "
            elif isinstance(v, tuple):
                repr_string = f"{k}=tuple(len={len(v)}), "
            else:
                repr_string += str(v) + ", "

        if repr_string[-2:] == ", ":
            repr_string = repr_string[:-2]
        return repr_string + ")"


class DirectModule(torch.nn.Module, DirectTransform, abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(self, sample: Dict):
        pass  # This comment passes "Function/method with an empty body PTC-W0049" error.


def count_parameters(models: dict) -> None:
    """Count the number of parameters of a dict of models.

    Parameters
    ----------
    models: dict
        Dictionary mapping model name to model.

    Returns
    -------
    """
    total_number_of_parameters = 0
    for model_name in models:
        n_params = sum(p.numel() for p in models[model_name].parameters())
        logger.info(f"Number of parameters model {model_name}: {n_params} ({n_params / 10.0 ** 3:.2f}k).")
        logger.debug(models[model_name])
        total_number_of_parameters += n_params
    logger.info(
        f"Total number of parameters model: {total_number_of_parameters} "
        f"({total_number_of_parameters / 10.0 ** 3:.2f}k)."
    )


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chunks(list_to_chunk, number_of_chunks):
    """Yield number_of_chunks number of sequential chunks from list_to_chunk.

    From https://stackoverflow.com/a/54802737
    """
    d, r = divmod(len(list_to_chunk), number_of_chunks)
    for idx in range(number_of_chunks):
        si = (d + 1) * (idx if idx < r else r) + d * (0 if idx < r else idx - r)
        yield list_to_chunk[si : si + (d + 1 if idx < r else d)]


def remove_keys(input_dict, keys):
    input_dict = dict(input_dict).copy()
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for key in keys:
        if key not in input_dict:
            continue
        del input_dict[key]
    return input_dict


def actualsizeMB(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    memory_size *= 9.537 * 10 ** (-7)
    return memory_size
