# Copyright (c) DIRECT Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from direct.config.defaults import BaseConfig
from direct.types import MaskFuncMode


@dataclass
class MaskingConfig(BaseConfig):
    name: str = MISSING
    accelerations: tuple[float, ...] = (5.0,)
    center_fractions: Optional[tuple[float, ...]] = (0.1,)
    uniform_range: bool = False
    mode: MaskFuncMode = MaskFuncMode.STATIC

    val_accelerations: tuple[float, ...] = (5.0, 10.0)
    val_center_fractions: Optional[tuple[float, ...]] = (0.1, 0.05)
