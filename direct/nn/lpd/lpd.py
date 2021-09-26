# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import direct.data.transforms as T

import torch
import torch.nn as nn


class DualNet(nn.Module):
    def __init__(self, num_dual, **kwargs):

        super(DualNet, self).__init__()

        if kwargs.get("dual_architectue") is None:
            n_hidden = kwargs.get("n_hidden")
            if n_hidden is None:
                raise ValueError("Missing argument n_hidden.")
            self.dual_block = nn.Sequential(
                *[
                    nn.Conv2d(2 * (num_dual + 2), n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, 2 * num_dual, kernel_size=3, padding=1),
                ]
            )
        else:
            self.dual_block = kwargs.get("dual_architectue")

    def forward(self, h, forward_f, g):

        inp = torch.cat([h, forward_f, g], dim=-1).permute(0, 1, 4, 2, 3)

        batch, coil, complex, height, width = inp.size()

        inp = inp.reshape(batch * coil, complex, height, width)

        return self.dual_block(inp).reshape(batch, coil, -1, height, width).permute(0, 1, 3, 4, 2)


class PrimalNet(nn.Module):
    def __init__(self, num_primal, **kwargs):

        super(PrimalNet, self).__init__()

        if kwargs.get("primal_architectue") is None:
            n_hidden = kwargs.get("n_hidden")
            if n_hidden is None:
                raise ValueError("Missing argument n_hidden.")
            self.primal_block = nn.Sequential(
                *[
                    nn.Conv2d(2 * (num_primal + 1), n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, 2 * num_primal, kernel_size=3, padding=1),
                ]
            )
        else:
            self.primal_block = kwargs.get("primal_architectue")

    def forward(self, f, backward_h):

        inp = torch.cat([f, backward_h], dim=-1).permute(0, 3, 1, 2)

        return self.primal_block(inp).permute(0, 2, 3, 1)


class LPDNet(nn.Module):
    """
    Learned Primal Dual implementation as in https://arxiv.org/abs/1707.06474.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iters: int,
        num_primal: int,
        num_dual: int,
        n_hidden: int = 32,
    ):
        """

        :param forward_operator: Callable
                    Forward operator.
        :param backward_operator: Callable
                    Backward operator.
        :param num_iters: int
                    Number of unrolled iterations.
        :param num_primal: int
                    Number of primal networks.
        :param num_dual: int
                    Number of dual networks.
        :param n_hidden: int
                    Number of convolutional hidden channels.
        """

        super(LPDNet, self).__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iters = num_iters
        self.num_primal = num_primal
        self.num_dual = num_dual
        self.n_hidden = n_hidden

        self.primal_net = nn.ModuleList([PrimalNet(num_primal, n_hidden=n_hidden) for _ in range(num_iters)])
        self.dual_net = nn.ModuleList([DualNet(num_dual, n_hidden=n_hidden) for _ in range(num_iters)])

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

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
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:

        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)

        if scaling_factor is not None:
            masked_kspace = masked_kspace * scaling_factor
            input_image = input_image * scaling_factor

        kspace_buffer = torch.cat([masked_kspace] * self.num_dual, self._complex_dim).to(masked_kspace.device)
        image_buffer = torch.cat([input_image] * self.num_primal, self._complex_dim).to(masked_kspace.device)

        for iter in range(self.num_iters):

            # Dual
            f_2 = image_buffer[..., 2:4].clone()
            kspace_buffer = self.dual_net[iter](
                kspace_buffer, self._forward_operator(f_2, sampling_mask, sensitivity_map), masked_kspace
            )

            # Primal
            h_1 = kspace_buffer[..., 0:2].clone()

            image_buffer = self.primal_net[iter](
                image_buffer, self._backward_operator(h_1, sampling_mask, sensitivity_map)
            )

        return image_buffer[..., 0:2]
