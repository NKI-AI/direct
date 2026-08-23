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

"""direct.nn.iterdualnet.iterdualnet module."""

from __future__ import annotations

import torch
from torch import nn

import direct.data.transforms as T
from direct.constants import COMPLEX_SIZE
from direct.nn.conv.modulated import ModConvActivation, ModConvType, ModulationParams
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.types import FFTOperator


class IterDualNet(nn.Module):
    r"""Iterative Dual Network solves iteratively the following problem

    .. math ::

        \min_{x} ||A(x) - y||_2^2 + \lambda_I ||x - D_I(x)||_2^2 + \lambda_F ||x - \mathcal{Q}``(D_F(f)``)||_2^2, \quad
        \left\{ \begin{array} Q = \mathcal{F}^{-1}, f = \mathcal{F}(x) & \text{if compute_per_coil is ``False``} \\
        Q = \mathcal{F}^{-1} \circ \mathcal{E}, f = \mathcal{R} \circ \mathcal{F}(x) & \text{otherwise} \end{array}

    by unrolling a gradient descent scheme where :math:`\mathcal{E}` and :math:`\mathcal{R}` are the expand and
    reduce operators which use the sensitivity maps. :math:`D_I` and :math:`D_F` are trainable U-Nets operating
    in the image and k-space domain.

    Supports conditional weight modulation as proposed in [#]_.

    References:
        .. [#] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
            Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning, PMLR
            315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_iter: int = 10,
        image_normunet: bool = False,
        kspace_normunet: bool = False,
        image_no_parameter_sharing: bool = True,
        kspace_no_parameter_sharing: bool = True,
        compute_per_coil: bool = True,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        **kwargs,
    ):
        """Inits :class:`IterDualNet`.

        Args:
            forward_operator: Forward Operator.
            backward_operator: Backward Operator.
            num_iter: Number of iterations. Default is ``10``.
            image_normunet: If ``True`` will use NormUNet for the image model. Default is ``False``.
            kspace_normunet: If ``True`` will use NormUNet for the kspace model. Default is ``False``.
            image_no_parameter_sharing: If ``False``, a single image model will be shared across all iterations.
                Default is ``True``.
            kspace_no_parameter_sharing: If ``False``, a single kspace model will be shared across all iterations.
                Default is ``True``.
            compute_per_coil: If ``True``:math:`f` will be transformed into a multi-coil kspace.
            conv_modulation: Modulation type for convolutional layers. Default is
                :attr:`~direct.nn.conv.modulated.modulated_conv.ModConvType.NONE`.
            aux_in_features: Number of features in the auxiliary input for modulation.
            fc_hidden_features: Hidden features in the modulation MLP.
            fc_groups: Groups for modulation MLP output. Default is ``1``.
            fc_activation: Activation after modulation MLP. Default is
                :attr:`~direct.nn.conv.modulated.modulated_conv.ModConvActivation.SIGMOID`.
            num_weights: Number of weight bases for :attr:`~direct.nn.conv.modulated.modulated_conv.ModConvType.SUM`.
            kwargs: Kwargs for unet models.

        Returns:
            ``None``.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.conv_modulation = conv_modulation

        self.image_no_parameter_sharing = image_no_parameter_sharing
        self.kspace_no_parameter_sharing = kspace_no_parameter_sharing
        image_unet_architecture = NormUnetModel2d if image_normunet else UnetModel2d
        kspace_unet_architecture = NormUnetModel2d if kspace_normunet else UnetModel2d

        modulation_params = ModulationParams(
            modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        self.image_block_list = nn.ModuleList()
        self.kspace_block_list = nn.ModuleList()

        for _ in range(self.num_iter if self.image_no_parameter_sharing else 1):
            self.image_block_list.append(
                image_unet_architecture(
                    in_channels=COMPLEX_SIZE,
                    out_channels=COMPLEX_SIZE,
                    num_filters=kwargs.get("image_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("image_unet_dropout", 0.0),
                    modulation_params=modulation_params,
                )
            )
        for _ in range(self.num_iter if self.kspace_no_parameter_sharing else 1):
            self.kspace_block_list.append(
                kspace_unet_architecture(
                    in_channels=COMPLEX_SIZE,
                    out_channels=COMPLEX_SIZE,
                    num_filters=kwargs.get("kspace_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
                    modulation_params=modulation_params,
                )
            )
        self.compute_per_coil = compute_per_coil

        self.lr = nn.Parameter(torch.ones(num_iter))
        self.reg_param_I = nn.Parameter(torch.ones(num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(num_iter))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def _image_model(
        self,
        image: torch.Tensor,
        step: int,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Image model.

        Args:
            image: Image.
            step: Step.
            auxiliary_data: Auxiliary data.

        Returns:
            The result.
        """
        image = image.permute(0, 3, 1, 2)
        block_idx = step if self.image_no_parameter_sharing else 0
        if self.conv_modulation != ModConvType.NONE:
            return self.image_block_list[block_idx](image, auxiliary_data).permute(0, 2, 3, 1).contiguous()
        return self.image_block_list[block_idx](image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(
        self,
        kspace: torch.Tensor,
        step: int,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Kspace model.

        Args:
            kspace: Kspace.
            step: Step.
            auxiliary_data: Auxiliary data.

        Returns:
            The result.
        """
        block_idx = step if self.kspace_no_parameter_sharing else 0
        if self.compute_per_coil:
            kspace = (
                self._compute_model_per_coil(
                    self.kspace_block_list[block_idx],
                    kspace.permute(0, 1, 4, 2, 3),
                    auxiliary_data,
                )
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        else:
            if self.conv_modulation != ModConvType.NONE:
                kspace = (
                    self.kspace_block_list[block_idx](kspace.permute(0, 3, 1, 2), auxiliary_data)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                )
            else:
                kspace = self.kspace_block_list[block_idx](kspace.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        return kspace

    def _compute_model_per_coil(
        self,
        model: nn.Module,
        data: torch.Tensor,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute model per coil.

        Args:
            model: Model.
            data: Data.
            auxiliary_data: Auxiliary data.

        Returns:
            The result.
        """
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            if self.conv_modulation != ModConvType.NONE:
                output.append(model(subselected_data, auxiliary_data))
            else:
                output.append(model(subselected_data))
        return torch.stack(output, dim=self._coil_dim)

    def _forward_operator(
        self,
        image: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward operator.

        Args:
            image: Image.
            sampling_mask: Sampling mask.
            sensitivity_map: Sensitivity map.

        Returns:
            The result.
        """
        return T.apply_mask(
            self.forward_operator(
                T.expand_operator(image, sensitivity_map, self._coil_dim),
                dim=self._spatial_dims,
            ),
            sampling_mask,
            return_mask=False,
        )

    def _backward_operator(
        self,
        kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Backward operator.

        Args:
            kspace: Kspace.
            sampling_mask: Sampling mask.
            sensitivity_map: Sensitivity map.

        Returns:
            The result.
        """
        return T.reduce_operator(
            self.backward_operator(
                T.apply_mask(kspace, sampling_mask, return_mask=False),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`IterDualNet`.

        Args:
            masked_kspace: Masked k-space of shape ``(N, coil, height, width, complex=2)``.
            sampling_mask: Sampling mask of shape ``(N, 1, height, width, 1)``.
            sensitivity_map: Sensitivity map of shape ``(N, coil, height, width, complex=2)``.
            auxiliary_data: Auxiliary data for modulation of shape ``(N, aux_in_features)``.

        Returns:
            Output image of shape ``(N, height, width, complex=2)``.
        """
        x = T.reduce_operator(
            self.backward_operator(masked_kspace, self._spatial_dims),
            sensitivity_map,
            self._coil_dim,
        )

        for step in range(self.num_iter):
            f = (
                self.forward_operator(
                    T.expand_operator(x, sensitivity_map, self._coil_dim),
                    dim=self._spatial_dims,
                )
                if self.compute_per_coil
                else self.forward_operator(x, dim=[d - 1 for d in self._spatial_dims])
            )
            kspace_model_out = self._kspace_model(f, step, auxiliary_data)
            kspace_model_out = (
                T.reduce_operator(
                    self.backward_operator(kspace_model_out, self._spatial_dims),
                    sensitivity_map,
                    self._coil_dim,
                )
                if self.compute_per_coil
                else self.backward_operator(kspace_model_out, dim=[d - 1 for d in self._spatial_dims])
            )

            img_model_out = self._image_model(x, step, auxiliary_data)

            dc_out = self._backward_operator(
                self._forward_operator(x, sampling_mask, sensitivity_map) - masked_kspace,
                sampling_mask,
                sensitivity_map,
            )
            x = (1 - self.lr[step] * (self.reg_param_I[step] + self.reg_param_F[step])) * x + self.lr[step] * (
                self.reg_param_I[step] * img_model_out + self.reg_param_F[step] * kspace_model_out - dc_out
            )
        return x
