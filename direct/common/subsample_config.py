# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, Optional, Union, Any

from direct.config.defaults import BaseConfig


@dataclass
class MaskingConfig(BaseConfig):
    name: str = MISSING
    accelerations: Tuple[int, ...] = (4,)  # Ideally Union[float, int].
    center_fractions: Optional[Tuple[float, ...]] = (0.08,)  # Ideally Optional[Tuple[float, ...]]
    uniform_range: bool = False
    image_center_crop: bool = False

    val_accelerations: Tuple[int, ...] = (4, 8)
    val_center_fractions: Optional[Tuple[float, ...]] = (0.08, 0.04)
