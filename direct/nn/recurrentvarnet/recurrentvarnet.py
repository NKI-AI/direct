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

"""direct.nn.recurrentvarnet.recurrentvarnet module."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import complex_multiplication, conjugate, expand_operator, reduce_operator
from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU
from direct.nn.types import InitType
from direct.types import FFTOperator


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [#]_.

    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock
    of the RecurrentVarNet.

    References:
        .. [#] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
            the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
            http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """Inits :class:`RecurrentInit`.

        Args:
            in_channels: Input channels.
            out_channels: Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
            channels: Channels :math:`n_d` in the convolutional layers of initializer.
            dilations: Dilations :math:`p` of the convolutional layers of the initializer.
            depth: RecurrentVarNet Block number of layers :math:`n_l`.
            multiscale_depth: ``1`` Number of feature layers to aggregate for the output, if ``1``, multi-scale context aggregation is
                disabled.

        Returns:
            ``None``.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes initialization for recurrent unit given input `x`.

        Args:
            x: Initialization for RecurrentInit.

        Returns:
            Initial recurrent hidden state from input `x`.
        """

        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class RecurrentVarNet(nn.Module):
    """Recurrent Variational Network implementation as presented in [#]_.

    References:
        .. [#] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
            the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
            http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        in_channels: int = COMPLEX_SIZE,
        num_steps: int = 15,
        recurrent_hidden_channels: int = 64,
        recurrent_num_layers: int = 4,
        no_parameter_sharing: bool = True,
        learned_initializer: bool = False,
        initializer_initialization: InitType | None = None,
        initializer_channels: tuple[int, ...] | None = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] | None = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        normalized: bool = False,
        **kwargs,
    ):
        """Inits :class:`RecurrentVarNet`.

        Args:
            forward_operator: Forward Operator.
            backward_operator: Backward Operator.
            num_steps: Number of iterations :math:`T`.
            in_channels: Input channel number. Default is ``2 for complex data``.
            recurrent_hidden_channels: Hidden channels number for the recurrent unit of the RecurrentVarNet Blocks. Default is
                ``64``.
            recurrent_num_layers: Number of layers for the recurrent unit of the RecurrentVarNet Block (:math:`n_l`). Default is
                ``4``.
            no_parameter_sharing: If ``False``, the same :class:`RecurrentVarNetBlock` is used for all num_steps. Default is
                ``True``.
            learned_initializer: If ``True`` an RSI module is used. Default is ``False``.
            initializer_initialization: Type of initialization for the RSI module. Can be either ``'sense'``, ``'zero-filled'`` or
                ``'input-image'``. Default is ``None``.
            initializer_channels: Channels :math:`n_d` in the convolutional layers of the RSI module. Default is ``(``32``, ``32``, ``64``,
                ``64``)``.
            initializer_dilations: Dilations :math:`p` of the convolutional layers of the RSI module. Default is ``(``1``, ``1``, ``2``,
                ``4``)``.
            initializer_multiscale: RSI module number of feature layers to aggregate for the output, if ``1``, multi-scale context
                aggregation is disabled. Default is ``1``.
            normalized: If ``True``, :class:`NormConv2dGRU` will be used as a regularizer in the :class:`RecurrentVarNetBlocks`.
                Default is ``False``.

        Returns:
            ``None``.
        """
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.initializer: nn.Module | None = None
        if (
            learned_initializer
            and initializer_initialization is not None
            and initializer_channels is not None
            and initializer_dilations is not None
        ):
            if initializer_initialization not in [
                "sense",
                "input_image",
                "zero_filled",
            ]:
                raise ValueError(
                    f"Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {initializer_initialization}."
                )
            self.initializer_initialization = initializer_initialization
            self.initializer = RecurrentInit(
                in_channels,
                recurrent_hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=recurrent_num_layers,
                multiscale_depth=initializer_multiscale,
            )
        self.num_steps = num_steps
        self.no_parameter_sharing = no_parameter_sharing
        self.block_list = nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    in_channels=in_channels,
                    hidden_channels=recurrent_hidden_channels,
                    num_layers=recurrent_num_layers,
                    normalized=normalized,
                )
            )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:

        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Args:
            kspace: k-space of shape ``(N, coil, height, width, complex=2)``.
            sensitivity_map: Sensitivity map of shape ``(N, coil, height, width, complex=2)``.

        Returns:
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
        input_image = complex_multiplication(
            conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)

        return input_image

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`RecurrentVarNet`.

        Args:
            masked_kspace: Masked k-space of shape ``(N, coil, height, width, complex=2)``.
            sampling_mask: Sampling mask of shape ``(N, 1, height, width, 1)``.
            sensitivity_map: Coil sensitivities of shape ``(N, coil, height, width, complex=2)``.

        Returns:
            k-space prediction.
        """

        previous_state: torch.Tensor | None = None

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                ).unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"].unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims)

            previous_state = self.initializer(
                self.forward_operator(initializer_input_image, dim=self._spatial_dims)
                .sum(self._coil_dim)
                .permute(0, 3, 1, 2)
            )

        kspace_prediction = masked_kspace.clone()

        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            kspace_prediction, previous_state = block(
                kspace_prediction,
                masked_kspace,
                sampling_mask,
                sensitivity_map,
                previous_state,
                self._coil_dim,
                self._spatial_dims,
            )

        return kspace_prediction


class RecurrentVarNetBlock(nn.Module):
    r"""Recurrent Variational Network Block :math:`\mathcal{H}_{\theta_{t}}` as presented in [#]_.

    References:
        .. [#] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
            the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
            http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        normalized: bool = False,
    ):
        """Inits RecurrentVarNetBlock.

        Args:
            forward_operator: Forward Fourier Transform.
            backward_operator: Backward Fourier Transform.
            in_channels: int, Input channel number. Default is ``2 for complex data``.
            hidden_channels: int, Hidden channels. Default is ``64``.
            num_layers: int, Number of layers of :math:`n_l` recurrent unit. Default is ``4``.
            normalized: If ``True``, :class:`NormConv2dGRU` will be used as a regularizer. Default is ``False``.

        Returns:
            ``None``.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))  # :math:`\alpha_t`
        regularizer_params = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "replication_padding": True,
        }
        # Recurrent Unit of RecurrentVarNet Block :math:`\mathcal{H}_{\theta_t}`
        self.regularizer = (
            NormConv2dGRU(**regularizer_params) if normalized else Conv2dGRU(**regularizer_params)  # type: ignore
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: None | torch.Tensor,
        coil_dim: int = 1,
        spatial_dims: tuple[int, int] = (2, 3),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass of RecurrentVarNetBlock.

        Args:
            current_kspace: Current k-space prediction of shape ``(N, coil, height, width, complex=2)``.
            masked_kspace: Masked k-space of shape ``(N, coil, height, width, complex=2)``.
            sampling_mask: Sampling mask of shape ``(N, 1, height, width, 1)``.
            sensitivity_map: Coil sensitivities of shape ``(N, coil, height, width, complex=2)``.
            hidden_state: Recurrent unit hidden state of shape ``(N, hidden_channels, height, width, num_layers)`` if not ``None``.
                Optional.
            coil_dim: Coil dimension. Default is ``1``.
            spatial_dims: Spatial dimensions. Default is ``(2, 3)``.

        Returns:
            New k-space prediction of shape ``(N, coil, height, width, complex=2)``.
            Next hidden state of shape ``(N, hidden_channels, height, width, num_layers)``.
        """

        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )

        recurrent_term = reduce_operator(
            self.backward_operator(current_kspace, dim=spatial_dims),
            sensitivity_map,
            dim=coil_dim,
        ).permute(0, 3, 1, 2)

        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)  # :math:`w_t`, :math:`h_{t+1}`
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = self.forward_operator(
            expand_operator(recurrent_term, sensitivity_map, dim=coil_dim),
            dim=spatial_dims,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state  # type: ignore
