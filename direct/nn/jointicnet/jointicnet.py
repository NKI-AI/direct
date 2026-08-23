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

"""direct.nn.jointicnet.jointicnet module."""

from __future__ import annotations

import torch
from torch import nn

import direct.data.transforms as T
from direct.nn.conv.modulated import ModConvActivation, ModConvType, ModulationParams
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.types import FFTOperator


class JointICNet(nn.Module):
    """Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network ``(Joint-ICNet)`` implementation as

    presented in [#]_.

    Supports conditional weight modulation as proposed in [#]_.

    References:
        .. [#] Jun, Yohan, et al. "Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
            ``(Joint-ICNet)`` for Fast MRI." 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
            IEEE, 2021, pp. 5266-75. https://doi.org/10.1109/CVPR46437.2021.00523.

        .. [#] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
            Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning, PMLR
            315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_iter: int = 10,
        use_norm_unet: bool = False,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        **kwargs,
    ):
        """Inits :class:`JointICNet`.

        Args:
            forward_operator: Forward Transform.
            backward_operator: Backward Transform.
            num_iter: Number of unrolled iterations. Default is ``10``.
            use_norm_unet: If ``True``, a Normalized U-Net is used. Default is ``False``.
            conv_modulation: Modulation type for convolutional layers. Default is
                :attr:`~direct.nn.conv.modulated.modulated_conv.ModConvType.NONE`.
            aux_in_features: Number of features in the auxiliary input for modulation.
            fc_hidden_features: Hidden features in the modulation MLP.
            fc_groups: Groups for modulation MLP output. Default is ``1``.
            fc_activation: Activation after modulation MLP. Default is
                :attr:`~direct.nn.conv.modulated.modulated_conv.ModConvActivation.SIGMOID`.
            num_weights: Number of weight bases for :attr:`~direct.nn.conv.modulated.modulated_conv.ModConvType.SUM`.
            kwargs: Image, k-space and sensitivity-map U-Net models keyword-arguments.

        Returns:
            ``None``.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.conv_modulation = conv_modulation

        unet_architecture = NormUnetModel2d if use_norm_unet else UnetModel2d

        modulation_params = ModulationParams(
            modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        self.image_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("image_unet_num_filters", 8),
            num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("image_unet_dropout", 0.0),
            modulation_params=modulation_params,
        )
        self.kspace_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("kspace_unet_num_filters", 8),
            num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
            modulation_params=modulation_params,
        )
        self.sens_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("sens_unet_num_filters", 8),
            num_pool_layers=kwargs.get("sens_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("sens_unet_dropout", 0.0),
            modulation_params=modulation_params,
        )
        self.conv_out = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

        self.reg_param_I = nn.Parameter(torch.ones(num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(num_iter))
        self.reg_param_C = nn.Parameter(torch.ones(num_iter))

        self.lr_image = nn.Parameter(torch.ones(num_iter))
        self.lr_sens = nn.Parameter(torch.ones(num_iter))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def _image_model(self, image: torch.Tensor, auxiliary_data: torch.Tensor | None = None) -> torch.Tensor:
        """Image model.

        Args:
            image: Image.
            auxiliary_data: Auxiliary data.

        Returns:
            The result.
        """
        image = image.permute(0, 3, 1, 2)
        if self.conv_modulation != ModConvType.NONE:
            return self.image_model(image, auxiliary_data).permute(0, 2, 3, 1).contiguous()
        return self.image_model(image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace: torch.Tensor, auxiliary_data: torch.Tensor | None = None) -> torch.Tensor:
        """Kspace model.

        Args:
            kspace: Kspace.
            auxiliary_data: Auxiliary data.

        Returns:
            The result.
        """
        kspace = kspace.permute(0, 3, 1, 2)
        if self.conv_modulation != ModConvType.NONE:
            return self.kspace_model(kspace, auxiliary_data).permute(0, 2, 3, 1).contiguous()
        return self.kspace_model(kspace).permute(0, 2, 3, 1).contiguous()

    def _sens_model(
        self,
        sensitivity_map: torch.Tensor,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sens model.

        Args:
            sensitivity_map: Sensitivity map.
            auxiliary_data: Auxiliary data.

        Returns:
            The result.
        """
        return (
            self._compute_model_per_coil(self.sens_model, sensitivity_map.permute(0, 1, 4, 2, 3), auxiliary_data)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

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
        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            self.forward_operator(
                T.expand_operator(image, sensitivity_map, self._coil_dim),
                dim=self._spatial_dims,
            ),
        )
        return forward

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
        backward = T.reduce_operator(
            self.backward_operator(
                torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                    kspace,
                ),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`JointICNet`.

        Args:
            masked_kspace: Masked k-space of shape ``(N, coil, height, width, complex=2)``.
            sampling_mask: Sampling mask of shape ``(N, 1, height, width, 1)``.
            sensitivity_map: Sensitivity map of shape ``(N, coil, height, width, complex=2)``.
            auxiliary_data: Auxiliary data for modulation of shape ``(N, aux_in_features)``.

        Returns:
            Output image of shape ``(N, height, width, complex=2)``.
        """

        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)
        input_image = input_image / T.modulus(input_image).unsqueeze(self._coil_dim).amax(dim=self._spatial_dims).view(
            -1, 1, 1, 1
        )

        for curr_iter in range(self.num_iter):
            step_sensitivity_map = (
                2
                * self.lr_sens[curr_iter]
                * (
                    T.complex_multiplication(
                        self.backward_operator(
                            torch.where(
                                sampling_mask == 0,
                                torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
                                self._forward_operator(input_image, sampling_mask, sensitivity_map) - masked_kspace,
                            ),
                            self._spatial_dims,
                        ),
                        T.conjugate(input_image).unsqueeze(self._coil_dim),
                    )
                    + self.reg_param_C[curr_iter]
                    * (
                        sensitivity_map
                        - self._sens_model(
                            self.backward_operator(masked_kspace, dim=self._spatial_dims),
                            auxiliary_data,
                        )
                    )
                )
            )
            sensitivity_map = sensitivity_map - step_sensitivity_map
            sensitivity_map_norm = torch.sqrt(((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim))
            sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._complex_dim).unsqueeze(self._coil_dim)
            sensitivity_map = T.safe_divide(sensitivity_map, sensitivity_map_norm)
            input_kspace = self.forward_operator(input_image, dim=tuple(d - 1 for d in self._spatial_dims))

            step_image = (
                2
                * self.lr_image[curr_iter]
                * (
                    self._backward_operator(
                        self._forward_operator(input_image, sampling_mask, sensitivity_map) - masked_kspace,
                        sampling_mask,
                        sensitivity_map,
                    )
                    + self.reg_param_I[curr_iter] * (input_image - self._image_model(input_image, auxiliary_data))
                    + self.reg_param_F[curr_iter]
                    * (
                        input_image
                        - self.backward_operator(
                            self._kspace_model(input_kspace, auxiliary_data),
                            dim=tuple(d - 1 for d in self._spatial_dims),
                        )
                    )
                )
            )

            input_image = input_image - step_image
            input_image = input_image / T.modulus(input_image).unsqueeze(self._coil_dim).amax(
                dim=self._spatial_dims
            ).view(-1, 1, 1, 1)

        out_image = self.conv_out(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_image
