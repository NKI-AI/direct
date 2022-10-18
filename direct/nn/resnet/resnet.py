# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from direct.data.transforms import reduce_operator
from direct.nn.resnet.conj import ConjGrad


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, scale: Optional[float] = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x.clone())))
        if self.scale:
            out = self.scale * out
        return x + out


class ResNet(nn.Module):
    """Simple residual network."""

    def __init__(
        self,
        hidden_channels: int,
        in_channels: int = 2,
        out_channels: Optional[int] = None,
        num_blocks: int = 15,
        batchnorm: bool = True,
        scale: Optional[float] = 0.1,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.resblocks = []
        for _ in range(num_blocks):
            self.resblocks.append(
                ResNetBlock(in_channels=hidden_channels, hidden_channels=hidden_channels, scale=scale)
            )
            if batchnorm:
                self.resblocks.append(nn.BatchNorm2d(num_features=hidden_channels))

        self.resblocks = nn.Sequential(*self.resblocks)
        if out_channels is None:
            out_channels = in_channels
        self.conv_out = nn.Sequential(
            *[
                nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            ]
        )

    def forward(
        self,
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`ResNet`.

        Parameters
        ----------
        input_image: torch.Tensor
            Masked k-space of shape (N, in_channels, height, width).

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        return self.conv_out(self.conv_in(input_image) + self.resblocks(self.conv_in(input_image)))


class ResNetConjGrad(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps: int,
        resnet_hidden_channels: int,
        resnet_num_blocks: int = 15,
        resenet_batchnorm: bool = True,
        resenet_scale: Optional[float] = 0.1,
        image_init: str = "sense",
        no_parameter_sharing: bool = True,
        cg_param_update_type: str = "PRB",
        cg_iters: int = 10,
        **kwargs,
    ):

        super().__init__()
        self.num_steps = num_steps
        self.resnets = nn.ModuleList()

        self.no_parameter_sharing = no_parameter_sharing
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.resnets.append(
                ResNet(
                    in_channels=2,
                    hidden_channels=resnet_hidden_channels,
                    num_blocks=resnet_num_blocks,
                    batchnorm=resenet_batchnorm,
                    scale=resenet_scale,
                )
            )
        self.learning_rate = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.normal_(self.learning_rate, 0, 1.0)
        self.mu = nn.Parameter(torch.ones(1), requires_grad=True)
        self.conj_grad = ConjGrad(forward_operator, backward_operator, cg_param_update_type, cg_iters)

        assert image_init in ["sense", "zero_filled"], (
            f"Unknown image_initialization. Expected 'sense' or 'zero_filled'. " f"Got {image_init}."
        )
        self.image_init = image_init

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`MRIResNetConjGrad`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.
        sampling_mask: torch.Tensor

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        x = init_image(
            self.image_init,
            self.backward_operator,
            masked_kspace,
            self._coil_dim,
            self._spatial_dims,
            sensitivity_map if self.image_init == "sense" else None,
        )
        z = x.clone()
        for i in range(self.num_steps):
            z = z - self.learning_rate[i] * (
                self.mu * (z - x)
                + self.resnets[i if self.no_parameter_sharing else 0](z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            )
            x = self.conj_grad(x, masked_kspace, z, sensitivity_map, sampling_mask, self.mu)

        return x


def init_image(
    image_init: str,
    backward_operator: Callable,
    kspace: torch.Tensor,
    coil_dim: int,
    spatial_dims: Tuple[int, int],
    sensitivity_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if image_init == "sense":
        image = reduce_operator(
            coil_data=backward_operator(kspace.clone(), dim=spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=coil_dim,
        )
    else:
        image = backward_operator(kspace, dim=spatial_dims).sum(coil_dim)

    return image
