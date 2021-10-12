# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import torch
import torch.nn as nn

import direct.data.transforms as T
from direct.nn.unet.unet_2d import UnetModel2d, NormUnetModel2d


class JointICNet(nn.Module):
    """
    Joint-ICNet implementation as in "Joint Deep Model-based MR Image and Coil Sensitivity Reconstruction Network
    (Joint-ICNet) for Fast MRI".

    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iter: int = 10,
        use_norm_unet: bool = False,
        **kwargs,
    ):
        """

        :param forward_operator: Callable,
                Forward Fourier Transform.
        :param backward_operator: Callable,
                Backward Fourier Transform.
        :param num_iter: int,
                Number of unrolled iterations. Default: 10.
        :param use_norm_unet: bool,
                If True, a Normalized U-Net is used. Default: False.
        :param kwargs:
                Image, k-space and sensitivity-map U-Net models parameters.
        """

        super(JointICNet, self).__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter

        unet_architecture = NormUnetModel2d if use_norm_unet else UnetModel2d

        self.image_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("image_unet_num_filters", 8),
            num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("image_unet_dropout", 0.0),
        )
        self.kspace_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("kspace_unet_num_filters", 8),
            num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
        )
        self.sens_model = unet_architecture(
            in_channels=2,
            out_channels=2,
            num_filters=kwargs.get("sens_unet_num_filters", 8),
            num_pool_layers=kwargs.get("sens_unet_num_pool_layers", 4),
            dropout_probability=kwargs.get("sens_unet_dropout", 0.0),
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

    def _image_model(self, image):

        image = image.permute(0, 3, 1, 2)

        return self.image_model(image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace):

        kspace = kspace.permute(0, 3, 1, 2)

        return self.kspace_model(kspace).permute(0, 2, 3, 1).contiguous()

    def _sens_model(self, sensitivity_map):

        return (
            self._compute_model_per_coil(self.sens_model, sensitivity_map.permute(0, 1, 4, 2, 3))
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def _compute_model_per_coil(self, model, data):

        output = []

        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self._coil_dim)

        return output

    def _forward_operator(self, image, sampling_mask, sensitivity_map):

        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )
        return forward

    def _backward_operator(self, kspace, sampling_mask, sensitivity_map):

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
    ) -> torch.Tensor:

        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)
        scaling_factor = T.modulus(input_image).unsqueeze(self._coil_dim).amax(dim=self._spatial_dims)
        input_image = input_image / scaling_factor.view(-1, 1, 1, 1)

        for iter in range(self.num_iter):

            step_sensitivity_map = (
                2
                * self.lr_sens[iter]
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
                    + self.reg_param_C[iter]
                    * (
                        sensitivity_map
                        - self._sens_model(self.backward_operator(masked_kspace, dim=self._spatial_dims))
                    )
                )
            )
            sensitivity_map = sensitivity_map - step_sensitivity_map

            sensitivity_map_norm = torch.sqrt(((sensitivity_map ** 2).sum(self._complex_dim)).sum(self._coil_dim))
            sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._complex_dim).unsqueeze(self._coil_dim)

            sensitivity_map = T.safe_divide(sensitivity_map, sensitivity_map_norm)

            input_kspace = self.forward_operator(input_image, dim=tuple([d - 1 for d in self._spatial_dims]))

            step_image = (
                2
                * self.lr_image[iter]
                * (
                    self._backward_operator(
                        self._forward_operator(input_image, sampling_mask, sensitivity_map) - masked_kspace,
                        sampling_mask,
                        sensitivity_map,
                    )
                    + self.reg_param_I[iter] * (input_image - self._image_model(input_image))
                    + self.reg_param_F[iter]
                    * (
                        input_image
                        - self.backward_operator(
                            self._kspace_model(input_kspace), dim=tuple([d - 1 for d in self._spatial_dims])
                        )
                    )
                )
            )
            step_image_scale = T.modulus(step_image).unsqueeze(self._coil_dim).amax(dim=self._spatial_dims)
            input_image = input_image / step_image_scale.view(-1, 1, 1, 1)

            input_image = input_image - step_image

        return self.conv_out(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
