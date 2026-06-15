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

import direct.data.transforms as T
from direct.nn.conv.modulated_conv import ModConvActivation, ModConvType
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.types import FFTOperator


class JointICNet(nn.Module):
    """Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet) implementation as
    presented in [1]_.

    Supports conditional weight modulation as proposed in [2]_.

    References
    ----------
    .. [1] Jun, Yohan, et al. "Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
        (Joint-ICNet) for Fast MRI." 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
        IEEE, 2021, pp. 5266-75. https://doi.org/10.1109/CVPR46437.2021.00523.

    .. [2] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_iter: int = 10,
        use_norm_unet: bool = False,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: Optional[int] = None,
        fc_hidden_features: Optional[tuple[int] | int] = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: Optional[int] = None,
        **kwargs,
    ):
        """Inits :class:`JointICNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Transform.
        backward_operator: Callable
            Backward Transform.
        num_iter: int
            Number of unrolled iterations. Default: 10.
        use_norm_unet: bool
            If True, a Normalized U-Net is used. Default: False.
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
        kwargs: dict
            Image, k-space and sensitivity-map U-Net models keyword-arguments.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.conv_modulation = conv_modulation

        unet_architecture = NormUnetModel2d if use_norm_unet else UnetModel2d

        mod_kwargs = dict(
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
            **mod_kwargs,
        )
        self.kspace_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("kspace_unet_num_filters", 8),
            num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
            **mod_kwargs,
        )
        self.sens_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("sens_unet_num_filters", 8),
            num_pool_layers=kwargs.get("sens_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("sens_unet_dropout", 0.0),
            **mod_kwargs,
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

    def _image_model(
        self, image: torch.Tensor, auxiliary_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        image = image.permute(0, 3, 1, 2)
        if self.conv_modulation != ModConvType.NONE:
            return (
                self.image_model(image, auxiliary_data).permute(0, 2, 3, 1).contiguous()
            )
        return self.image_model(image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(
        self, kspace: torch.Tensor, auxiliary_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kspace = kspace.permute(0, 3, 1, 2)
        if self.conv_modulation != ModConvType.NONE:
            return (
                self.kspace_model(kspace, auxiliary_data)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        return self.kspace_model(kspace).permute(0, 2, 3, 1).contiguous()

    def _sens_model(
        self,
        sensitivity_map: torch.Tensor,
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (
            self._compute_model_per_coil(
                self.sens_model, sensitivity_map.permute(0, 1, 4, 2, 3), auxiliary_data
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def _compute_model_per_coil(
        self,
        model: nn.Module,
        data: torch.Tensor,
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        auxiliary_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`JointICNet`.

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
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """

        input_image = self._backward_operator(
            masked_kspace, sampling_mask, sensitivity_map
        )
        input_image = input_image / T.modulus(input_image).unsqueeze(
            self._coil_dim
        ).amax(dim=self._spatial_dims).view(-1, 1, 1, 1)

        for curr_iter in range(self.num_iter):
            step_sensitivity_map = (
                2
                * self.lr_sens[curr_iter]
                * (
                    T.complex_multiplication(
                        self.backward_operator(
                            torch.where(
                                sampling_mask == 0,
                                torch.tensor([0.0], dtype=masked_kspace.dtype).to(
                                    masked_kspace.device
                                ),
                                self._forward_operator(
                                    input_image, sampling_mask, sensitivity_map
                                )
                                - masked_kspace,
                            ),
                            self._spatial_dims,
                        ),
                        T.conjugate(input_image).unsqueeze(self._coil_dim),
                    )
                    + self.reg_param_C[curr_iter]
                    * (
                        sensitivity_map
                        - self._sens_model(
                            self.backward_operator(
                                masked_kspace, dim=self._spatial_dims
                            ),
                            auxiliary_data,
                        )
                    )
                )
            )
            sensitivity_map = sensitivity_map - step_sensitivity_map
            sensitivity_map_norm = torch.sqrt(
                ((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim)
            )
            sensitivity_map_norm = sensitivity_map_norm.unsqueeze(
                self._complex_dim
            ).unsqueeze(self._coil_dim)
            sensitivity_map = T.safe_divide(sensitivity_map, sensitivity_map_norm)
            input_kspace = self.forward_operator(
                input_image, dim=tuple(d - 1 for d in self._spatial_dims)
            )

            step_image = (
                2
                * self.lr_image[curr_iter]
                * (
                    self._backward_operator(
                        self._forward_operator(
                            input_image, sampling_mask, sensitivity_map
                        )
                        - masked_kspace,
                        sampling_mask,
                        sensitivity_map,
                    )
                    + self.reg_param_I[curr_iter]
                    * (input_image - self._image_model(input_image, auxiliary_data))
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
            input_image = input_image / T.modulus(input_image).unsqueeze(
                self._coil_dim
            ).amax(dim=self._spatial_dims).view(-1, 1, 1, 1)

        out_image = self.conv_out(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_image
