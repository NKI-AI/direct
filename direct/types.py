# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

import pathlib
from enum import Enum
from typing import NewType, Union

import torch
from torch import nn as nn
from torch.cuda.amp import GradScaler

Number = Union[float, int]
PathOrString = Union[pathlib.Path, str]
FileOrUrl = NewType("FileOrUrl", PathOrString)
HasStateDict = Union[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, GradScaler]
TensorOrNone = Union[None, torch.Tensor]


class DirectEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str) -> DirectEnum | None:
        statuses = cls.__members__.keys()
        for st in statuses:
            if st.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            _other = str(other.value)
        else:
            _other = str(other)
        return bool(self.value.lower() == _other.lower())

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        return hash(self.value.lower())
