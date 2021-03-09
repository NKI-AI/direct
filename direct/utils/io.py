# coding=utf-8
# Copyright (c) DIRECT Contributors
import json
import numpy as np
import pathlib
import torch
import warnings
from typing import Dict, List, Union


def read_json(fn: Union[Dict, str, pathlib.Path]) -> Dict:
    """
    Read file and output dict, or take dict and output dict.


    Parameters
    ----------
    fn : Union[Dict, str, pathlib.Path]


    Returns
    -------
    dict

    """
    if isinstance(fn, dict):
        return fn

    with open(fn, "r") as f:
        data = json.load(f)
    return data


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()

        if isinstance(obj, np.ndarray):
            if obj.size > 10e4:
                warnings.warn(
                    "Trying to JSON serialize a very large array of size {obj.size}. "
                    "Reconsider doing this differently"
                )
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_json(fn: Union[str, pathlib.Path], data: Dict, indent=2) -> None:
    """
    Write dict data to fn.

    Parameters
    ----------
    fn : Path or str
    data : dict
    indent: int

    Returns
    -------
    None
    """
    with open(fn, "w") as f:
        json.dump(data, f, indent=indent, cls=ArrayEncoder)


def read_list(fn: Union[List, str, pathlib.Path]) -> List:
    """
    Read file and output list, or take list and output list.

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list

    Returns
    -------
    list
    """
    if isinstance(fn, (pathlib.Path, str)):
        with open(fn) as f:
            filter_fns = f.readlines()
        return [_.strip() for _ in filter_fns if not _.startswith("#")]
    return fn


def write_list(fn: Union[str, pathlib.Path], data) -> None:
    """
    Write list line by line to file.

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list
    data : list or tuple
    Returns
    -------
    None
    """
    with open(fn, "w") as f:
        for line in data:
            f.write(f"{line}\n")
