# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import MISSING

from direct.config.defaults import BaseConfig


@dataclass
class MaskingConfig(BaseConfig):
    name: str = MISSING
    accelerations: Tuple[float, ...] = (5.0,)  # Ideally Union[float, int].
    center_fractions: Optional[Tuple[float, ...]] = (0.1,)  # Ideally Optional[Tuple[float, ...]]
    uniform_range: bool = False
    image_center_crop: bool = False
    dynamic_mask: Optional[bool] = None

    val_accelerations: Tuple[float, ...] = (5.0, 10.0)
    val_center_fractions: Optional[Tuple[float, ...]] = (0.1, 0.05)
