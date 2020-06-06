# coding=utf-8
# Copyright (c) DIRECT Contributors
import importlib
import subprocess
import torch
import pathlib

from typing import List, Tuple, Dict, Any, Optional, Union, Callable, KeysView


def is_power_of_two(number: int) -> bool:
    """Check if input is a power of 2"""
    return number != 0 and ((number & (number - 1)) == 0)


def ensure_list(data: Any) -> List:
    """
    Ensure input is a list.

    Parameters
    ----------
    data :

    Returns
    -------

    """
    if data is None:
        return []

    if not isinstance(data, (list, tuple)):
        return [data]

    return list(data)


def cast_as_path(data: Optional[Union[pathlib.Path, str]]) -> Optional[pathlib.Path]:
    if data is None:
        return None

    return pathlib.Path(data)


def str_to_class(module_name: str, class_name: str) -> Callable:
    """
    Convert a string to a class
    From: https://stackoverflow.com/a/1176180/576363

    Parameters
    ----------
    module_name : str
        e.g. direct.data.transforms
    class_name : str
        e.g. Identity
    Returns
    -------
    object
    """

    # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
    module = importlib.import_module(module_name)
    # Get the class, will raise AttributeError if class cannot be found.
    the_class = getattr(module, class_name)

    return the_class


def dict_to_device(data: Dict[str, torch.Tensor],
                   device: Union[int, str], keys: Optional[Union[List, Tuple, KeysView]] = None) -> Dict:
    """
    Copy tensor-valued dictionary to device.

    Parameters
    ----------
    data : Dict[str, torch.Tensor]
    device : str
    keys : List, Tuple
        Subselection of keys to copy

    Returns
    -------
    Dictionary at the new device.
    """
    if keys is None:
        keys = data.keys()
    return {k: v.to(device) for k, v in data.items() if k in keys if isinstance(v, torch.Tensor)}


def detach_dict(data: Dict[str, torch.Tensor], keys: Optional[Union[List, Tuple, KeysView]] = None) -> Dict:
    """
    Return a detached copy of a dictionary.

    Parameters
    ----------
    data : Dict[str, torch.Tensor]
    keys : List, Tuple
        Subselection of keys to detach

    Returns
    -------
    Dictionary at the new device.
    """
    if keys is None:
        keys = data.keys()
    return {k: v.detach() for k, v in data.items() if k in keys if isinstance(v, torch.Tensor)}


def reduce_list_of_dicts(data: List[Dict[str, torch.Tensor]], mode='average', divisor=None) -> Dict[str, torch.Tensor]:
    """
    Average a list of dictionary mapping keys to Tensors


    Parameters
    ----------
    data : List[Dict[str, torch.Tensor]])
    mode : str
        Which reduction mode, average reduces the dictionary, sum just adds while sum_di
    divisor : None or int
        If given values are divided by this factor.

    Returns
    -------
    Dict[str, torch.Tensor] : Averaged dictionary
    """
    if mode not in ['average', 'sum']:
        raise ValueError(f'Reduction can only be `sum` or `average`.')

    if not divisor:
        divisor = 1.

    result_dict = {k: torch.zeros_like(v) for k, v in data[0].items()}

    for elem in data:
        result_dict = {k: result_dict[k] + v for k, v in elem.items()}

    if mode == 'average':
        divisor *= len(data)

    return {k: v / divisor for k, v in result_dict.items()}


def evaluate_dict(fns_dict, source, target, reduction='mean'):
    """
    Evaluate a dictionary of functions.

    Example
    -------
    > evaluate_dict({'l1_loss: F.l1_loss, 'l2_loss': F.l2_loss}, a, b)

    Will return
    > {'l1_loss', F.l1_loss(a, b, reduction='mean'), 'l2_loss': F.l2_loss(a, b, reduction='mean')

    Parameters
    ----------
    fns_dict : Dict[str, Callable]
    source : torch.Tensor
    target : torch.Tensor
    reduction : str

    Returns
    -------
    Dict[str, torch.Tensor]
    """
    return {k: fns_dict[k](source, target, reduction=reduction) for k, v in fns_dict.items()}


def prefix_dict_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Append a prefix to a dictionary keys.

    Parameters
    ----------
    data : Dict[str, Any]
    prefix : str

    Returns
    -------
    Dict[str, Any]
    """
    return {prefix + k: v for k, v in data.items()}


def git_hash() -> str:
    """
    Returns the current git hash.

    Returns
    -------
    str : the current git hash.
    """
    try:
        _git_hash = str(subprocess.check_output(['git', 'rev-parse', 'HEAD'])).strip()
    except FileNotFoundError:
        _git_hash = ''

    return _git_hash


def normalize_image(data: torch.Tensor, eps: float = 0.00001) -> torch.Tensor:
    """
    Normalize image to range [0,1] for visualization.

    Parameters
    ----------
    data : torch.Tensor
    eps : float

    Returns
    -------
    torch.Tensor: scaled data.
    """

    data = data - data.min()
    data = data / (data.max() + eps)
    return data
