# coding=utf-8
# Copyright (c) DIRECT Contributors

import torch
import torch.nn as nn


class MultiCoil(nn.Module):
    """
    To be used with multicoil k-space data.
    """

    def __init__(self, model: nn.Module, coil_dim: int = 1, coiltobatch: bool = False):
        """

        :param model: nn.Module,
                Any nn.Module that works with 4D data (N, H, W, C). Typically a convolutional-like model.
        :param coil_dim: int,
                Coil dimension. Default: 1.
        :param coiltobatch: bool,
                If True batch and coil dimensions are merged when forwarded by the model and unmerged when outputted.
                Otherwise, input is forwarded to the model per coil.
        """
        super(MultiCoil, self).__init__()

        self.model = model
        self.coiltobatch = coiltobatch
        self._coil_dim = coil_dim

    def _compute_model_per_coil(self, data: torch.Tensor) -> torch.Tensor:

        output = []

        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.model(subselected_data))
        output = torch.stack(output, dim=self._coil_dim)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.coiltobatch:
            x = x.clone()
            batch, coil, height, width, channels = x.size()

            x = x.reshape(batch * coil, height, width, channels).permute(0, 3, 1, 2).contiguous()
            x = self.model(x).permute(0, 2, 3, 1)
            x = x.reshape(batch, coil, height, width, -1)
        else:
            x = self._compute_model_per_coil(x).contiguous()

        return x
