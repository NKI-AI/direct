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

"""Engines for End-to-End Variational Network model.

Includes supervised, self-supervised and joint supervised and self-supervised learning engines.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import JSSLMRIModelEngine, SSLMRIModelEngine
from direct.types import FFTOperator


class EndToEndVarNetEngine(MRIModelEngine):
    """End-to-End Variational Network Engine.

    Args:
        cfg: Configuration file.
        model: Model.
        device: Device. Can be "cuda:{idx}" or ``"cpu"``.
        forward_operator: The forward FFT operator (e.g. ``direct.data.transforms.fft2``).
        backward_operator: The backward FFT operator (e.g. ``direct.data.transforms.ifft2``).
        mixed_precision: Use mixed precision. Default is ``False``.
        **models: Additional models.
    """

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
        """Inits :class:`EndToEndVarNetEngine`.

        Args:
            cfg: Configuration file.
            model: Model.
            device: Device. Can be "cuda:{idx}" or ``"cpu"``.
            forward_operator: The forward FFT operator (e.g. ``direct.data.transforms.fft2``).
            backward_operator: The backward FFT operator (e.g. ``direct.data.transforms.ifft2``).
            mixed_precision: Use mixed precision. Default is ``False``.
            **models: Additional models.

        Returns:
            ``None``.
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

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            data: Data.

        Returns:
            The result.
        """
        auxiliary_data = self.auxiliary_data_from(data)

        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
            auxiliary_data=auxiliary_data,
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )  # shape (batch, height,  width)

        return output_image, output_kspace


class EndToEndVarNetSSLEngine(SSLMRIModelEngine):
    """Self-supervised Learning End-to-End Variational Network Engine.

    Used for supplementary experiments for End-to-End Variational Network model with SLL in the JSSL paper [1].

    Args:
        cfg: Configuration file.
        model: Model.
        device: Device. Can be "cuda:{idx}" or ``"cpu"``.
        forward_operator: The forward FFT operator (e.g. ``direct.data.transforms.fft2``).
        backward_operator: The backward FFT operator (e.g. ``direct.data.transforms.ifft2``).
        mixed_precision: Use mixed precision. Default is ``False``.
        **models: Additional models.

    References:
        .. [#] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
            Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
            https://doi.org/10.48550/arXiv.2311.15856.
    """

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
        """Inits :class:`EndToEndVarNetSSLEngine`.

        Args:
            cfg: Configuration file.
            model: Model.
            device: Device. Can be "cuda:{idx}" or ``"cpu"``.
            forward_operator: The forward FFT operator (e.g. ``direct.data.transforms.fft2``).
            backward_operator: The backward FFT operator (e.g. ``direct.data.transforms.ifft2``).
            mixed_precision: Use mixed precision. Default is ``False``.
            **models: Additional models.

        Returns:
            ``None``.
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

    def forward_function(self, data: dict[str, Any]) -> tuple[None, torch.Tensor]:
        """Forward function for :class:`EndToEndVarNetSSLEngine`.

        Args:
            data: Data dictionary. Should contain the following keys: - ``"input_kspace"`` if training,
                ``"masked_kspace"`` if inference - ``"input_sampling_mask"`` if training, ``"sampling_mask"`` if
                inference - ``"sensitivity_map"``

        Returns:
            ``None`` for image and output k-space.
        """

        kspace = data["input_kspace"] if self.model.training else data["masked_kspace"]
        mask = data["input_sampling_mask"] if self.model.training else data["sampling_mask"]

        output_kspace = self.model(
            masked_kspace=kspace,
            sampling_mask=mask,
            sensitivity_map=data["sensitivity_map"],
            auxiliary_data=self.auxiliary_data_from(data),
        )
        output_image = None

        return output_image, output_kspace


class EndToEndVarNetJSSLEngine(JSSLMRIModelEngine):
    """Joint Supervised and Self-supervised Learning End-to-End Variational Network Engine.

    Used for supplementary experiments for End-to-End Variational Network model with JSLL in the JSSL paper [1].

    Args:
        cfg: Configuration file.
        model: Model.
        device: Device. Can be "cuda:{idx}" or ``"cpu"``.
        forward_operator: The forward FFT operator (e.g. ``direct.data.transforms.fft2``).
        backward_operator: The backward FFT operator (e.g. ``direct.data.transforms.ifft2``).
        mixed_precision: Use mixed precision. Default is ``False``.
        **models: Additional models.

    References:
        .. [#] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
            Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
            https://doi.org/10.48550/arXiv.2311.15856.
    """

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
        """Inits :class:`EndToEndVarNetJSSLEngine`.

        Args:
            cfg: Configuration file.
            model: Model.
            device: Device. Can be "cuda:{idx}" or ``"cpu"``.
            forward_operator: The forward FFT operator (e.g. ``direct.data.transforms.fft2``).
            backward_operator: The backward FFT operator (e.g. ``direct.data.transforms.ifft2``).
            mixed_precision: Use mixed precision. Default is ``False``.
            **models: Additional models.

        Returns:
            ``None``.
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

    def forward_function(self, data: dict[str, Any]) -> tuple[None, torch.Tensor]:
        """Forward function for :class:`EndToEndVarNetJSSLEngine`.

        Args:
            data: Data dictionary. Should contain the following keys: - ``"is_ssl"`` boolean tensor indicating if
                training is SSL - ``"input_kspace"`` if training and training is SSL, ``"masked_kspace"`` if inference -
                ``"input_sampling_mask"`` if training and training is SSL, ``"sampling_mask"`` if inference -
                ``"sensitivity_map"``

        Returns:
            ``None`` for image and output k-space.
        """

        if data["is_ssl"][0] and self.model.training:
            kspace, mask = data["input_kspace"], data["input_sampling_mask"]
        else:
            kspace, mask = data["masked_kspace"], data["sampling_mask"]

        output_kspace = self.model(
            masked_kspace=kspace,
            sampling_mask=mask,
            sensitivity_map=data["sensitivity_map"],
            auxiliary_data=self.auxiliary_data_from(data),
        )
        output_image = None

        return output_image, output_kspace


class EndToEndVarNet3DEngine(MRIModelEngine):
    """End-to-End Variational Network Engine for 3D data."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Callable | None = None,
        backward_operator: Callable | None = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Initialize the instance.

        Args:
            cfg: Cfg.
            model: Model.
            device: Device.
            forward_operator: Forward operator.
            backward_operator: Backward operator.
            mixed_precision: Mixed precision.
            **models: Models.

        Returns:
            ``None``.
        """
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,  # ty: ignore[invalid-argument-type]
            backward_operator=backward_operator,  # ty: ignore[invalid-argument-type]
            mixed_precision=mixed_precision,
            **models,
        )
        self._spatial_dims = (3, 4)

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            data: Data.

        Returns:
            The result.
        """
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )
        return output_image, output_kspace
