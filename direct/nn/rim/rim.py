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
"""direct.nn.rim.rim module."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from direct.data import transforms as T
from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU
from direct.nn.types import InitType
from direct.types import FFTOperator
from direct.utils.asserts import assert_positive_integer


class MRILogLikelihood(nn.Module):
    r"""Defines the MRI loglikelihood assuming one noise vector for the complex images for all coils:

    .. math::
         \frac{1}{\sigma^2} \sum_{i}^{N_c} {S}_i^{\text{H}} \mathcal{F}^{-1} P^{*} (P \mathcal{F} S_i x_{\tau} - y_{\tau})

    for each time step :math:`\tau`.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
    ):
        """Inits :class:`MRILogLikelihood`.

        Args:
            forward_operator: Forward Operator.
            backward_operator: Backward Operator.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        input_image,
        masked_kspace,
        sensitivity_map,
        sampling_mask,
        loglikelihood_scaling=None,
    ) -> torch.Tensor:
        """Performs forward pass of :class:`MRILogLikelihood`.

        Args:
            input_image: Initial or previous iteration of image with complex first of shape (N, complex, height, width).
            masked_kspace: Masked k-space of shape (N, coil, height, width, complex).
            sensitivity_map: Sensitivity Map of shape (N, coil, height, width, complex).
            sampling_mask: Sampling mask.
            loglikelihood_scaling: Multiplier for loglikelihood, for instance for the k-space noise, of shape (1,).

        Returns:
            The MRI Loglikelihood.
        """

        input_image = input_image.permute(0, 2, 3, 1)  # shape (N, height, width, complex)

        if loglikelihood_scaling is None:
            loglikelihood_scaling = torch.tensor([1.0], dtype=masked_kspace.dtype).to(masked_kspace.device)
        loglikelihood_scaling = loglikelihood_scaling.reshape(
            -1, *(torch.ones(len(sensitivity_map.shape) - 1).int())
        )  # shape (1, 1, 1, 1, 1)

        # We multiply by the loglikelihood_scaling here to prevent fp16 information loss,
        # as this value is typically <<1, and the operators are linear.

        mul = loglikelihood_scaling * T.complex_multiplication(
            sensitivity_map,
            input_image.unsqueeze(1),  # (N, 1, height, width, complex)
        )  # shape (N, coil, height, width, complex)

        mr_forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            self.forward_operator(mul, dim=self._spatial_dims),
        )  # shape (N, coil, height, width, complex)

        error = mr_forward - loglikelihood_scaling * torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            masked_kspace,
        )  # shape (N, coil, height, width, complex)

        mr_backward = self.backward_operator(error, dim=self._spatial_dims)  # shape (N, coil, height, width, complex)

        if sensitivity_map is not None:
            out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum(self._coil_dim)
        else:
            out = mr_backward.sum(self._coil_dim)
        # out has shape (N, complex=2, height, width)

        out = out.permute(0, 3, 1, 2)  # complex first: shape (N, height, width, complex=2)

        return out


class RIMInit(nn.Module):
    """Learned initializer for RIM, based on multi-scale context aggregation with dilated convolutions, that replaces

    zero initializer for the RIM hidden vector. Inspired by [#]_.

    References:
        .. [#] Yu, Fisher, and Vladlen Koltun. “Multi-Scale Context Aggregation by Dilated Convolutions.” ArXiv:1511.07122 [Cs], Apr. 2016. arXiv.org, http://arxiv.org/abs/1511.07122.
    """

    def __init__(
        self,
        x_ch: int,
        out_ch: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """Inits :class:`RIMInit`.

        Args:
            x_ch: Input channels.
            out_ch: Number of hidden channels in the RIM.
            channels: Channels in the convolutional layers of initializer. Typical it could be e.g. (32, 32, 64, 64).
            dilations: Dilations of the convolutional layers of the initializer. Typically it could be e.g. (1, 1, 2, 4).
            depth: RIM depth
            multiscale_depth: 1 Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is
                disabled.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = x_ch
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_ch, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: X.

        Returns:
            The result.
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


class RIM(nn.Module):
    """Recurrent Inference Machine Module as in [#]_.

    References:
        .. [#] Putzky, Patrick, and Max Welling. “Recurrent Inference Machines for Solving Inverse Problems.” ArXiv:1706.04008 [Cs], June 2017. arXiv.org, http://arxiv.org/abs/1706.04008.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        hidden_channels: int,
        x_channels: int = 2,
        length: int = 8,
        depth: int = 1,
        no_parameter_sharing: bool = True,
        instance_norm: bool = False,
        dense_connect: bool = False,
        skip_connections: bool = True,
        replication_padding: bool = True,
        image_initialization: InitType = InitType.ZERO_FILLED,
        learned_initializer: bool = False,
        initializer_channels: tuple[int, ...] | None = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] | None = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        normalized: bool = False,
        **kwargs,
    ):
        """Inits :class:`RIM`.

        Args:
            forward_operator: Forward Operator.
            backward_operator: Backward Operator.
            hidden_channels: Number of hidden channels in recurrent unit of RIM.
            x_channels: Number of input channels. Default is ``2 (complex data)``.
            length: Number of time-steps. Default is ``8``.
            depth: Number of layers of recurrent unit of RIM. Default is ``1``.
            no_parameter_sharing: If False, a single recurrent unit will be used for each time-step. Default is ``True``.
            instance_norm: If True, instance normalization is applied in the recurrent unit of RIM. Default is ``False``.
            dense_connect: Use dense connection in the recurrent unit of RIM. Default is ``False``.
            skip_connections: If True, the previous prediction is added to the next. Default is ``True``.
            replication_padding: Replication padding for the recurrent unit of RIM. Defaul: True.
            image_initialization: Input image initialization for RIM. Can be InitType.SENSE, InitType.INPUT_KSPACE,
                InitType.INPUT_IMAGE or InitType.ZERO_FILLED. Default is ``InitType.ZERO_FILLED``.
            learned_initializer: If True, an initializer is trained to learn image initialization. Default is ``False``.
            initializer_channels: Number of channels for learned_initializer. If "learned_initializer=False" this is ignored.
                Default is ``(32, 32, 64, 64)``.
            initializer_dilations: Number of dilations for learned_initializer. Must have the same length as
                "initialize_channels". If "learned_initializer=False" this is ignored. Default is ``(1, 1, 2, 4)``.
            initializer_multiscale: Number of initializer multiscale. If "learned_initializer=False" this is ignored. Default is
                ``1``.
            normalized: If True, :class:`NormConv2dGRU` will be used instead of :class:`Conv2dGRU`. Default is ``False``.
        """
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "steps",
                "sensitivity_map_model",
                "model_name",
                "z_reduction_frequency",
                "kspace_context",
                "scale_loglikelihood",
                "whiten_input",  # should be passed!
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        assert_positive_integer(x_channels, hidden_channels, length, depth)
        # assert_bool(no_parameter_sharing, instance_norm, dense_connect, skip_connections, replication_padding)

        self.initializer: nn.Module | None = None
        if learned_initializer and initializer_channels is not None and initializer_dilations is not None:
            # List is because of a omegaconf bug.
            self.initializer = RIMInit(
                x_channels,
                hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=depth,
                multiscale_depth=initializer_multiscale,
            )

        allowed = {
            InitType.SENSE,
            InitType.INPUT_KSPACE,
            InitType.INPUT_IMAGE,
            InitType.ZERO_FILLED,
        }
        if isinstance(image_initialization, InitType):
            parsed = image_initialization
        elif isinstance(image_initialization, str):
            parsed = InitType.from_str(image_initialization)
        else:
            parsed = None
        if parsed not in allowed:
            raise ValueError(
                "Unknown image_initialization. Expected InitType.SENSE, InitType.INPUT_KSPACE, "
                f"InitType.INPUT_IMAGE or InitType.ZERO_FILLED. Got {image_initialization!r}."
            )
        self.image_initialization = parsed

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.grad_likelihood = MRILogLikelihood(forward_operator, backward_operator)

        self.skip_connections = skip_connections

        self.x_channels = x_channels
        self.hidden_channels = hidden_channels

        self.cell_list = nn.ModuleList()
        self.no_parameter_sharing = no_parameter_sharing

        conv_unit_params = {
            "in_channels": x_channels * 2,  # double channels as input is concatenated image and gradient
            "out_channels": x_channels,
            "hidden_channels": hidden_channels,
            "num_layers": depth,
            "instance_norm": instance_norm,
            "dense_connect": dense_connect,
            "replication_padding": replication_padding,
        }
        for _ in range(length if no_parameter_sharing else 1):
            self.cell_list.append(
                NormConv2dGRU(**conv_unit_params) if normalized else Conv2dGRU(**conv_unit_params)  # type: ignore
            )

        self.length = length
        self.depth = depth

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        # kspace is of shape: (N, coil, height, width, complex)
        # sensitivity_map is of shape (N, coil, height, width, complex)

        """Compute sense init.

        Args:
            kspace: Kspace.
            sensitivity_map: Sensitivity map.

        Returns:
            The result.
        """
        input_image = T.complex_multiplication(
            T.conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )  # shape (N, coil, height, width, complex=2)

        input_image = input_image.sum(self._coil_dim)

        # shape (N, height, width, complex=2)
        return input_image

    def forward(
        self,
        input_image: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor | None = None,
        previous_state: torch.Tensor | None = None,
        loglikelihood_scaling: torch.Tensor | None = None,
        **kwargs,
    ):
        """Performs forward pass of :class:`RIM`.

        Args:
            input_image: Initial or intermediate guess of input. Has shape (N, height, width, complex=2).
            masked_kspace: Masked k-space of shape (N, coil, height, width, complex=2).
            sensitivity_map: Sensitivity map of shape (N, coil, height, width, complex=2).
            sampling_mask: Sampling mask of shape (N, 1, height, width, 1).
            previous_state: Previous state.
            loglikelihood_scaling: Float tensor of shape (1,).

        Returns:
            The result.
        """
        if input_image is None:
            if self.image_initialization == InitType.SENSE:
                input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                )
            elif self.image_initialization == InitType.INPUT_KSPACE:
                if "initial_kspace" not in kwargs:
                    raise ValueError(
                        f"`'initial_kspace` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = self.compute_sense_init(
                    kspace=kwargs["initial_kspace"],
                    sensitivity_map=sensitivity_map,
                )
            elif self.image_initialization == InitType.INPUT_IMAGE:
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = kwargs["initial_image"]

            elif self.image_initialization == InitType.ZERO_FILLED:
                input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
            else:
                raise ValueError(
                    "Unknown image_initialization. Expected InitType.SENSE, InitType.INPUT_KSPACE, "
                    f"InitType.INPUT_IMAGE or InitType.ZERO_FILLED. Got {self.image_initialization}."
                )
        # Provide an initialization for the first hidden state.
        if (self.initializer is not None) and (previous_state is None):
            previous_state = self.initializer(
                input_image.permute(0, 3, 1, 2)
            )  # shape (N, hidden_channels, height, width, depth)
        # TODO: This has to be made contiguous

        input_image = input_image.permute(0, 3, 1, 2).contiguous()  # shape (N, complex=2, height, width)

        batch_size = input_image.size(0)
        spatial_shape = [input_image.size(self._spatial_dims[0]), input_image.size(self._spatial_dims[1])]
        # Initialize zero state for RIM
        state_size = [batch_size, self.hidden_channels] + list(spatial_shape) + [self.depth]
        if previous_state is None:
            # shape (N, hidden_channels, height, width, depth)
            previous_state = torch.zeros(*state_size, dtype=input_image.dtype).to(input_image.device)

        cell_outputs = []
        intermediate_image = input_image  # shape (N, complex=2, height, width)

        for cell_idx in range(self.length):
            cell = self.cell_list[cell_idx] if self.no_parameter_sharing else self.cell_list[0]

            grad_loglikelihood = self.grad_likelihood(
                intermediate_image,
                masked_kspace,
                sensitivity_map,
                sampling_mask,
                loglikelihood_scaling,
            )  # shape (N, complex=2, height, width)

            if grad_loglikelihood.abs().max() > 150.0:
                warnings.warn(
                    f"Very large values for the gradient loglikelihood ({grad_loglikelihood.abs().max()}). "
                    f"Might cause difficulties."
                )

            cell_input = torch.cat(
                [intermediate_image, grad_loglikelihood],
                dim=1,
            )  # shape (N, complex=4, height, width)

            cell_output, previous_state = cell(cell_input, previous_state)
            # shapes (N, complex=2, height, width), (N, hidden_channels, height, width, depth)

            if self.skip_connections:
                # shape (N, complex=2, height, width)
                intermediate_image = intermediate_image + cell_output
            else:
                # shape (N, complex=2, height, width)
                intermediate_image = cell_output

            if not self.training:
                # If not training, memory can be significantly reduced by clearing the previous cell.
                cell_output.set_()
                grad_loglikelihood.set_()
                del cell_output, grad_loglikelihood

            # Only save intermediate reconstructions at training step
            if self.training or cell_idx == (self.length - 1):
                cell_outputs.append(intermediate_image)  # type: ignore

        return cell_outputs, previous_state
