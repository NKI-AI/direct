# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch

from direct.functionals.challenges import *
from direct.functionals.grad import *
from direct.functionals.nmae import NMAELoss
from direct.functionals.nmse import *
from direct.functionals.psnr import *
from direct.functionals.ssim import *


def accuracy_metric(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return accuracy
