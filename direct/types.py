# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
from typing import NewType, Union

import torch
from torch import nn as nn
from torch.cuda.amp import GradScaler

Number = Union[float, int]
PathOrString = Union[pathlib.Path, str]
FileOrUrl = NewType("FileOrUrl", PathOrString)
HasStateDict = Union[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, GradScaler]
