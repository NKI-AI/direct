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

from typing import Optional

import torch
import torch.nn as nn

from direct.data.transforms import expand_operator, reduce_operator
from direct.nn.conv.modulated_conv import ModConvActivation, ModConvType
from direct.nn.unet import UnetModel2d
from direct.types import FFTOperator


class EndToEndVarNet(nn.Module):
    """End-to-End Variational Network based on [1]_.

    Supports conditional weight modulation as proposed in [2]_.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. "End-to-End Variational Networks for Accelerated MRI Reconstruction."
        ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.

    .. [2] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_layers: int,
        regularizer_num_filters: int = 18,
        regularizer_num_pull_layers: int = 4,
        regularizer_dropout: float = 0.0,
        in_channels: int = 2,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        **kwargs,
    ):
        """Inits :class:`EndToEndVarNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_layers: int
            Number of cascades.
        regularizer_num_filters: int
            Regularizer model number of filters.
        regularizer_num_pull_layers: int
            Regularizer model number of pulling layers.
        regularizer_dropout: float
            Regularizer model dropout probability.
        in_channels: int
            Number of input channels. Default: 2.
        conv_modulation : ModConvType
            Modulation type for convolutional layers. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of features in the auxiliary input for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features in the modulation MLP.
        fc_groups : int
            Groups for modulation MLP output. Default: 1.
        fc_activation : ModConvActivation
            Activation after modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(
                    f"{type(self).__name__} got key `{extra_key}` which is not supported."
                )

        self.conv_modulation = conv_modulation
        self.layers_list = nn.ModuleList()

        for _ in range(num_layers):
            self.layers_list.append(
                EndToEndVarNetBlock(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    regularizer_model=UnetModel2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        num_filters=regularizer_num_filters,
                        num_pool_layers=regularizer_num_pull_layers,
                        dropout_probability=regularizer_dropout,
                        modulation=conv_modulation,
                        aux_in_features=aux_in_features,
                        fc_hidden_features=fc_hidden_features,
                        fc_groups=fc_groups,
                        fc_activation=fc_activation,
                        num_weights=num_weights,
                    ),
                    conv_modulation=conv_modulation,
                )
            )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        auxiliary_data: torch.Tensor, optional
            Auxiliary data for modulation of shape (N, aux_in_features).

        Returns
        -------
        kspace_prediction: torch.Tensor
            K-space prediction of shape (N, coil, height, width, complex=2).
        """

        kspace_prediction = masked_kspace.clone()
        for layer in self.layers_list:
            kspace_prediction = layer(
                kspace_prediction,
                masked_kspace,
                sampling_mask,
                sensitivity_map,
                auxiliary_data,
            )
        return kspace_prediction


class EndToEndVarNetBlock(nn.Module):
    """End-to-End Variational Network block."""

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        regularizer_model: nn.Module,
        conv_modulation: ModConvType = ModConvType.NONE,
    ):
        """Inits :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        regularizer_model: nn.Module
            Regularizer model.
        conv_modulation: ModConvType
            Modulation type. Default: ModConvType.NONE.
        """
        super().__init__()
        self.regularizer_model = regularizer_model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        self.conv_modulation = conv_modulation
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        auxiliary_data: torch.Tensor, optional
            Auxiliary data for modulation of shape (N, aux_in_features).

        Returns
        -------
        torch.Tensor
            Next k-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )
        regularization_term = torch.cat(
            [
                reduce_operator(
                    self.backward_operator(kspace, dim=self._spatial_dims),
                    sensitivity_map,
                    dim=self._coil_dim,
                )
                for kspace in torch.split(current_kspace, 2, self._complex_dim)
            ],
            dim=self._complex_dim,
        ).permute(0, 3, 1, 2)

        if self.conv_modulation != ModConvType.NONE:
            regularization_term = self.regularizer_model(
                regularization_term, auxiliary_data
            ).permute(0, 2, 3, 1)
        else:
            regularization_term = self.regularizer_model(regularization_term).permute(
                0, 2, 3, 1
            )

        regularization_term = torch.cat(
            [
                self.forward_operator(
                    expand_operator(image, sensitivity_map, dim=self._coil_dim),
                    dim=self._spatial_dims,
                )
                for image in torch.split(regularization_term, 2, self._complex_dim)
            ],
            dim=self._complex_dim,
        )
        return current_kspace - self.learning_rate * kspace_error + regularization_term
