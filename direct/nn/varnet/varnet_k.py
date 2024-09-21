# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn

from direct.data.transforms import expand_operator, reduce_operator
from direct.nn.unet.unet_3d import UnetModel3d


class EndToEndVarNetKSpace3D(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_layers: int,
        regularizer_num_filters: int = 18,
        regularizer_num_pull_layers: int = 4,
        regularizer_dropout: float = 0.0,
        in_channels: int = 2,
        **kwargs,
    ):
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.layers_list = nn.ModuleList()

        for _ in range(num_layers):
            self.layers_list.append(
                EndToEndVarNetKSpace3DBlock(
                    regularizer_model=UnetModel3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        num_filters=regularizer_num_filters,
                        num_pool_layers=regularizer_num_pull_layers,
                        dropout_probability=regularizer_dropout,
                    ),
                )
            )
        self.backward_operator = backward_operator

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        kspace_prediction = masked_kspace.clone()
        for layer in self.layers_list:
            kspace_prediction = layer(kspace_prediction, masked_kspace, sampling_mask)
        image_prediction = reduce_operator(
            self.backward_operator(kspace_prediction, dim=self._spatial_dims), sensitivity_map, dim=self._coil_dim
        )
        return image_prediction


class EndToEndVarNetKSpace3DBlock(nn.Module):

    def __init__(
        self,
        regularizer_model: nn.Module,
    ):
        super().__init__()
        self.regularizer_model = regularizer_model
        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        self._coil_dim = 1

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> torch.Tensor:

        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )
        regularization_term = current_kspace.permute(0, 1, 5, 2, 3, 4)
        regularization_term = torch.stack(
            [
                self.regularizer_model(regularization_term[:, coil_idx]).permute(0, 2, 3, 4, 1)
                for coil_idx in range(current_kspace.shape[self._coil_dim])
            ],
            dim=self._coil_dim,
        )
        return current_kspace - self.learning_rate * kspace_error + regularization_term
