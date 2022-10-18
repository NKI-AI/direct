# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn

import direct.data.transforms as T
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d


class IterDualNet(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iter: int = 10,
        use_norm_unet: bool = False,
        no_parameter_sharing: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ):

        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter

        self.no_parameter_sharing = no_parameter_sharing
        unet_architecture = NormUnetModel2d if use_norm_unet else UnetModel2d

        self.image_block_list = nn.ModuleList()
        self.kspace_block_list = nn.ModuleList()
        self.image_block_grad_list = nn.ModuleList()
        self.kspace_block_grad_list = nn.ModuleList()
        for _ in range(self.num_iter if self.no_parameter_sharing else 1):
            self.image_block_list.append(
                unet_architecture(
                    in_channels=2,
                    out_channels=2,
                    num_filters=kwargs.get("image_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("image_unet_dropout", 0.0),
                )
            )
            self.image_block_grad_list.append(
                unet_architecture(
                    in_channels=2,
                    out_channels=2,
                    num_filters=kwargs.get("image_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("image_unet_dropout", 0.0),
                )
            )

            self.kspace_block_list.append(
                unet_architecture(
                    in_channels=2,
                    out_channels=2,
                    num_filters=kwargs.get("kspace_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
                )
            )
            self.kspace_block_grad_list.append(
                unet_architecture(
                    in_channels=2,
                    out_channels=2,
                    num_filters=kwargs.get("kspace_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
                )
            )
        self.compute_per_coil = compute_per_coil
        self.lr = nn.Parameter(torch.ones(num_iter))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def _image_model(self, image: torch.Tensor, step: int) -> torch.Tensor:
        image = image.permute(0, 3, 1, 2)
        block_idx = step if self.no_parameter_sharing else 0

        return self.image_block_list[block_idx](image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace: torch.Tensor, step: int) -> torch.Tensor:
        block_idx = step if self.no_parameter_sharing else 0
        if self.compute_per_coil:
            kspace = self._compute_model_per_coil(
                self.kspace_block_list[block_idx], kspace.permute(0, 1, 4, 2, 3)
            ).permute(0, 1, 3, 4, 2)
        else:
            kspace = self.kspace_block_list[block_idx](kspace.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        return kspace

    def _compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        return torch.stack(output, dim=self._coil_dim)

    def _forward_operator(
        self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:

        return torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )

    def _backward_operator(
        self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:

        return T.reduce_operator(
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

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`IterDualNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        image = T.reduce_operator(
            self.backward_operator(masked_kspace, self._spatial_dims), sensitivity_map, self._coil_dim
        )

        for step in range(self.num_iter):
            kspace_model_in = (
                self._forward_operator(image, sampling_mask, sensitivity_map)
                if self.compute_per_coil
                else self.forward_operator(image, dim=(1, 2))
            )
            kspace_model_out = self._kspace_model(kspace_model_in, step)
            kspace_model_out = (
                self._backward_operator(kspace_model_out, sampling_mask, sensitivity_map)
                if self.compute_per_coil
                else self.backward_operator(kspace_model_out, dim=(1, 2))
            )
            kspace_model_out = self.kspace_block_grad_list[step if self.no_parameter_sharing else 0](
                (image - kspace_model_out).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)

            img_model_out = self._image_model(image, step)

            img_model_out = self.image_block_grad_list[step if self.no_parameter_sharing else 0](
                (image - img_model_out).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)

            dc_out = self._backward_operator(
                self._forward_operator(image, sampling_mask, sensitivity_map) - masked_kspace,
                sampling_mask,
                sensitivity_map,
            )
            # image = (1 - self.lr[step] * (self.reg_param_I + self.reg_param_F)) * image + self.lr[
            #     step
            # ] * (self.reg_param_I * img_model_out + self.reg_param_F * kspace_model_out - dc_out)
            image = image - self.lr[step] * (img_model_out + kspace_model_out)
        return image


class IterDualNetSSL(IterDualNet):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iter: int = 10,
        use_norm_unet: bool = False,
        no_parameter_sharing: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ):

        super().__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            num_iter=num_iter,
            use_norm_unet=use_norm_unet,
            no_parameter_sharing=no_parameter_sharing,
            compute_per_coil=compute_per_coil,
            **kwargs,
        )
