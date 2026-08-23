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
"""direct.nn.conjgradnet.conjgradnet_engine module."""

from typing import Any

import torch
from torch import nn

from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from direct.types import FFTOperator


class ConjGradNetEngine(MRIModelEngine):
    """ConjGradNetEngine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`ConjGradNetEngine`.

        Args:
            cfg: Configuration file.
            model: Model.
            device: Device. Can be "cuda:{idx}" or "cpu".
            forward_operator: The forward operator. Default is ``None``.
            backward_operator: The backward operator. Default is ``None``.
            mixed_precision: Use mixed precision. Default is ``False``.
            **models: Additional models.
        """
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, None]:
        """Forward function.

        Args:
            data: Data.

        Returns:
            The result.
        """
        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, height,  width)
        output_kspace = None
        return output_image, output_kspace
