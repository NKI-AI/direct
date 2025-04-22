# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
