# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
from typing import Union

import torch
from torch import nn as nn
from torch.cuda.amp import GradScaler

Number = Union[float, int]
PathOrString = Union[pathlib.Path, str]
HasStateDict = Union[nn.Module, torch.optim.Optimizer, GradScaler]
