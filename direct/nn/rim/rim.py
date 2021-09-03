# coding=utf-8
# Copyright (c) DIRECT Contributors

import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.data import transforms as T
from direct.utils.asserts import assert_positive_integer


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU Cell to be used with RIM (Recurrent Inference Machines).
    """

    def __init__(
        self,
        x_channels: int,
        hidden_channels,
        depth=2,
        gru_kernel_size=1,
        ortho_init: bool = True,
        instance_norm: bool = False,
        dense_connect=0,
        replication_padding=False,
    ):
        super().__init__()
        self.depth = depth
        self.x_channels = x_channels
        self.hidden_channels = hidden_channels
        self.instance_norm = instance_norm
        self.dense_connect = dense_connect
        self.repl_pad = replication_padding

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks of RIM cell
        for idx in range(depth + 1):
            in_ch = x_channels + 2 if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < depth else x_channels
            pad = 0 if replication_padding else (2 if idx == 0 else 1)
            block = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=pad,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks of RIM cell
        for idx in range(depth):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                block = []
                if instance_norm:
                    block.append(nn.InstanceNorm2d(2 * hidden_channels))
                block.append(
                    nn.Conv2d(
                        2 * hidden_channels,
                        hidden_channels,
                        gru_kernel_size,
                        padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*block))

        if ortho_init:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        cell_input : torch.Tensor
            Reconstruction input
        previous_state : torch.Tensor
            Tensor of previous stats.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
        """

        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        for idx in range(self.depth):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1))
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.depth](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.depth](cell_input)

        return out, torch.stack(new_states, dim=-1)


class MRILogLikelihood(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
    ):
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        # TODO UGLY
        self.ndim = 2

        self._coil_dim = 1
        self._spatial_dims = (2, 3) if self.ndim == 2 else (2, 3, 4)

    def forward(
        self,
        input_image,
        masked_kspace,
        sensitivity_map,
        sampling_mask,
        loglikelihood_scaling=None,
    ) -> torch.Tensor:
        r"""
        Defines the MRI loglikelihood assuming one noise vector for the complex images for all coils.
        $$ \frac{1}{\sigma^2} \sum_{i}^{\text{num coils}}
            {S}_i^\{text{H}} \mathcal{F}^{-1} P^T (P \mathcal{F} S_i x_\tau - y_\tau)$$
        for each time step $\tau$

        Parameters
        ----------
        input_image : torch.tensor
            Initial or previous iteration of image with complex first
            of shape (batch, complex, [slice,] height, width).
        masked_kspace : torch.tensor
            Masked k-space of shape (batch, coil, [slice,] height, width, complex).
        sensitivity_map : torch.tensor
            Sensitivity Map of shape (batch, coil, [slice,] height, width, complex).
        sampling_mask : torch.tensor
        loglikelihood_scaling : torch.tensor
            Multiplier for loglikelihood, for instance for the k-space noise, of shape (1,).

        Returns
        -------
        torch.Tensor
        """
        if input_image.ndim == 5:
            self.ndim = 3

        input_image = input_image.permute(
            (0, 2, 3, 1) if self.ndim == 2 else (0, 2, 3, 4, 1)
        )  # shape (batch, [slice,] height, width, complex)

        loglikelihood_scaling = loglikelihood_scaling.reshape(
            list(torch.ones(len(sensitivity_map.shape)).int())
        )  # shape (1, 1, 1, [1,] 1, 1)

        # We multiply by the loglikelihood_scaling here to prevent fp16 information loss,
        # as this value is typically <<1, and the operators are linear.

        mul = loglikelihood_scaling * T.complex_multiplication(
            sensitivity_map, input_image.unsqueeze(1)  # (batch, 1, [slice,] height, width, complex)
        )  # shape (batch, coil, [slice,] height, width, complex)

        mr_forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            self.forward_operator(mul, dim=self._spatial_dims),
        )  # shape (batch, coil, [slice],  height, width, complex)

        error = mr_forward - loglikelihood_scaling * torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            masked_kspace,
        )  # shape (batch, coil, [slice],  height, width, complex)

        mr_backward = self.backward_operator(
            error, dim=self._spatial_dims
        )  # shape (batch, coil, [slice],  height, width, complex)

        if sensitivity_map is not None:
            out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum(self._coil_dim)
        else:
            out = mr_backward.sum(self._coil_dim)
        # out has shape (batch, complex=2, [slice], height, width)

        out = (
            out.permute(0, 3, 1, 2) if self.ndim == 2 else out.permute(0, 4, 1, 2, 3)
        )  # complex first: shape (batch, [slice], height, width, complex=2)

        return out


class RIMInit(nn.Module):
    def __init__(
        self,
        x_ch: int,
        out_ch: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """
        Learned initializer for RIM, based on multi-scale context aggregation with dilated convolutions, that replaces
        zero initializer for the RIM hidden vector.

        Inspired by "Multi-Scale Context Aggregation by Dilated Convolutions" (https://arxiv.org/abs/1511.07122)

        Parameters
        ----------
        x_ch : int
            Input channels.
        out_ch : int
            Number of hidden channels in the RIM.
        channels : tuple
            Channels in the convolutional layers of initializer. Typical it could be e.g. (32, 32, 64, 64).
        dilations: tuple
            Dilations of the convolutional layers of the initializer. Typically it could be e.g. (1, 1, 2, 4).
        depth : int
            RIM depth
        multiscale_depth : 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.

        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = x_ch
        for (curr_channels, curr_dilations) in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for idx in range(depth):
            block = [nn.Conv2d(tch, out_ch, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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
    """
    Recurrent Inference Machine Module as in https://arxiv.org/abs/1706.04008.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        hidden_channels: int,
        x_channels: int = 2,
        length: int = 8,
        depth: int = 1,
        no_parameter_sharing: bool = True,
        instance_norm: bool = False,
        dense_connect: bool = False,
        skip_connections: bool = True,
        replication_padding: bool = True,
        image_initialization: str = "zero_filled",
        learned_initializer: bool = False,
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        **kwargs,
    ):
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

        self.initializer: Optional[nn.Module] = None
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

        self.image_initialization = image_initialization

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.grad_likelihood = MRILogLikelihood(forward_operator, backward_operator)

        self.skip_connections = skip_connections

        self.x_channels = x_channels
        self.hidden_channels = hidden_channels

        self.cell_list = nn.ModuleList()
        self.no_parameter_sharing = no_parameter_sharing
        for _ in range(length if no_parameter_sharing else 1):
            self.cell_list.append(
                ConvGRUCell(
                    x_channels,
                    hidden_channels,
                    depth=depth,
                    instance_norm=instance_norm,
                    dense_connect=dense_connect,
                    replication_padding=replication_padding,
                )
            )

        self.length = length
        self.depth = depth

    def compute_sense_init(self, kspace, sensitivity_map, spatial_dims=(2, 3), coil_dim=1):
        # kspace is of shape: (batch, coil, [slice,] height, width, complex)
        # sensitivity_map is of shape (batch, coil, [slice,] height, width, complex)

        input_image = T.complex_multiplication(
            T.conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=spatial_dims),
        )  # shape (batch, coil, [slice,] height, width, complex=2)

        input_image = input_image.sum(coil_dim)

        # shape (batch, [slice,] height, width, complex=2)
        return input_image

    def forward(
        self,
        input_image: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
        loglikelihood_scaling: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        input_image : torch.Tensor
            Initial or intermediate guess of input. Has shape (batch, [slice,] height, width, complex=2).
        masked_kspace : torch.Tensor
            Kspace masked by the sampling mask.
        sensitivity_map : torch.Tensor
            Coil sensitivities.
        sampling_mask : torch.Tensor
            Sampling mask.
        previous_state : torch.Tensor
        loglikelihood_scaling : torch.Tensor
            Float tensor of shape (1,).

        Returns
        -------
        torch.Tensor
        """

        if input_image is None:
            if self.image_initialization == "sense":
                input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                    spatial_dims=(3, 4) if masked_kspace.ndim == 6 else (2, 3),
                )
            elif self.image_initialization == "input_kspace":
                if "initial_kspace" not in kwargs:
                    raise ValueError(
                        f"`'initial_kspace` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = self.compute_sense_init(
                    kspace=kwargs["initial_kspace"],
                    sensitivity_map=sensitivity_map,
                    spatial_dims=(3, 4) if kwargs["initial_kspace"].ndim == 6 else (2, 3),
                )
            elif self.image_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = kwargs["initial_image"]

            elif self.image_initialization == "zero_filled":
                coil_dim = 1
                input_image = self.backward_operator(masked_kspace).sum(coil_dim)
            else:
                raise ValueError(
                    f"Unknown image_initialization. Expected `sense`, `input_kspace`, `'input_image` or `zero_filled`. "
                    f"Got {self.image_initialization}."
                )

        # Provide an initialization for the first hidden state.
        if (self.initializer is not None) and (previous_state is None):
            previous_state = self.initializer(
                input_image.permute((0, 4, 1, 2, 3) if input_image.ndim == 5 else (0, 3, 1, 2))
            )  # permute to (batch, complex, [slice], height, width),
        # TODO: This has to be made contiguous
        # TODO(gy): Do 3D data pass from here? If not remove if statements below and [slice,] from comments.

        input_image = input_image.permute(
            (0, 4, 1, 2, 3) if input_image.ndim == 5 else (0, 3, 1, 2)
        ).contiguous()  # shape (batch, , complex=2, [slice,] height, width)

        batch_size = input_image.size(0)
        spatial_shape = (
            [input_image.size(-3), input_image.size(-2), input_image.size(-1)]
            if input_image.ndim == 5
            else [input_image.size(-2), input_image.size(-1)]
        )

        # Initialize zero state for RIM
        state_size = [batch_size, self.hidden_channels] + list(spatial_shape) + [self.depth]
        if previous_state is None:
            # shape (batch, hidden_channels, [slice,] height, width, depth)
            previous_state = torch.zeros(*state_size, dtype=input_image.dtype).to(input_image.device)

        cell_outputs = []
        intermediate_image = input_image  # shape (batch, , complex=2, [slice,] height, width)

        for cell_idx in range(self.length):
            cell = self.cell_list[cell_idx] if self.no_parameter_sharing else self.cell_list[0]

            grad_loglikelihood = self.grad_likelihood(
                intermediate_image,
                masked_kspace,
                sensitivity_map,
                sampling_mask,
                loglikelihood_scaling,
            )  # shape (batch, , complex=2, [slice,] height, width)

            if grad_loglikelihood.abs().max() > 150.0:
                warnings.warn(
                    f"Very large values for the gradient loglikelihood ({grad_loglikelihood.abs().max()}). "
                    f"Might cause difficulties."
                )

            cell_input = torch.cat(
                [intermediate_image, grad_loglikelihood],
                dim=1,
            )  # shape (batch, , complex=4, [slice,] height, width)

            cell_output, previous_state = cell(cell_input, previous_state)
            # shapes (batch, complex=2, [slice,] height, width), (batch, hidden_channels, [slice,] height, width, depth)

            if self.skip_connections:
                # shape (batch, complex=2, [slice,] height, width)
                intermediate_image = intermediate_image + cell_output
            else:
                # shape (batch, complex=2, [slice,] height, width)
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
