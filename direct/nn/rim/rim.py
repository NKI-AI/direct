# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from typing import Optional

from direct.utils.asserts import assert_positive_integer


class ConvGRUCell(nn.Module):
    """
    RIMCell
    """

    def __init__(
        self,
        x_channels: int,
        hidden_channels,
        depth=2,
        gru_kernel_size=1,
        ortho_init=True,
        instance_norm=False,
        dense_connection=0,
        replication_padding=False,
    ):
        super().__init__()
        self.depth = depth
        self.x_channels = x_channels
        self.hidden_channels = hidden_channels
        self.instance_norm = instance_norm
        self.dense_connection = dense_connection
        self.repl_pad = replication_padding

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks of RIM cell
        for idx in range(depth + 1):
            in_ch = (
                x_channels + 2
                if idx == 0
                else (1 + min(idx, dense_connection)) * hidden_channels
            )
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
            for reset_gate, update_gate, out_gate in zip(
                self.reset_gates, self.update_gates, self.out_gates
            ):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(self, cell_input, previous_state):
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

        new_states = []
        conv_skip = []
        for idx in range(self.depth):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](
                        torch.cat(
                            [*conv_skip[-self.dense_connection :], cell_input], dim=1
                        )
                    ),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connection > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat(
                [cell_input, previous_state[:, :, :, :, idx]], dim=1
            )

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](
                    torch.cat(
                        [cell_input, previous_state[:, :, :, :, idx] * reset], dim=1
                    )
                )
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.depth](
                torch.cat([*conv_skip[-self.dense_connection :], cell_input], dim=1)
            )
        else:
            out = self.conv_blocks[self.depth](cell_input)
        return out, torch.stack(new_states, dim=-1)


class RIM(nn.Module):
    def __init__(
        self,
        x_channels: int,
        num_hidden_channels: int,
        grad_likelihood: nn.Module,
        length: int = 8,
        depth: int = 1,
        no_sharing: bool = True,
        instance_norm: bool = False,
        dense_connection: bool = False,
        replication_padding: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert_positive_integer(x_channels, num_hidden_channels, length, depth)

        self.x_channels = x_channels
        self.num_hidden_channels = num_hidden_channels

        self.cell_list = nn.ModuleList()
        self.no_sharing = no_sharing
        for _ in range(length if no_sharing else 1):
            self.cell_list.append(
                ConvGRUCell(
                    x_channels,
                    num_hidden_channels,
                    depth=depth,
                    instance_norm=instance_norm,
                    dense_connection=dense_connection,
                    replication_padding=replication_padding,
                )
            )
        self.length = length
        self.grad_likelihood = grad_likelihood
        self.depth = depth

    def forward(
        self,
        input_image: torch.Tensor,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        previous_state: Optional[torch.Tensor] = None,
        loglikelihood_scaling: Optional[float] = None,
        **kwargs,
    ):

        """

        Parameters
        ----------
        input_image : torch.Tensor
            Initial or intermediate guess of input.
        masked_kspace : torch.Tensor
            Kspace masked by the sampling mask.
        sensitivity_map : torch.Tensor
            Coil sensitivities.
        sampling_mask : torch.Tensor
            Sampling mask.
        previous_state : torch.Tensor
        loglikelihood_scaling : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        # TODO: This has to be made contiguous
        input_image = input_image.align_to("batch", "complex", "height", "width").contiguous()  # type: ignore

        batch_size = input_image.size("batch")
        spatial_shape = [input_image.size("height"), input_image.size("width")]
        # Initialize zero state for RIM
        state_size = (
            [batch_size, self.num_hidden_channels] + list(spatial_shape) + [self.depth]
        )
        if previous_state is None:
            previous_state = torch.zeros(*state_size, dtype=input_image.dtype).to(
                input_image.device
            )

        cell_outputs = []
        intermediate_image = input_image
        for cell_idx in range(self.length):
            cell = self.cell_list[cell_idx] if self.no_sharing else self.cell_list[0]

            grad_loglikelihood = self.grad_likelihood(
                intermediate_image,
                masked_kspace,
                sensitivity_map,
                sampling_mask,
                loglikelihood_scaling,
            )

            if grad_loglikelihood.abs().max() > 150.0:
                warnings.warn(
                    f"Very large values for the gradient loglikelihood ({grad_loglikelihood.abs().max()}). "
                    f"Might cause difficulties."
                )

            cell_input = torch.cat(
                [intermediate_image.rename(None), grad_loglikelihood.rename(None)],
                dim=1,
            )
            cell_output, previous_state = cell(cell_input, previous_state)
            intermediate_image = intermediate_image + cell_output
            if not self.training:
                # If not training, memory can be significantly reduced by clearing the previous cell.
                cell_output.set_()
                grad_loglikelihood.rename(
                    None
                ).set_()  # TODO: Fix when named tensors have this support.
                del cell_output, grad_loglikelihood
            # Only save intermediate reconstructions at training step
            if self.training or cell_idx == (self.length - 1):
                cell_outputs.append(intermediate_image.refine_names("batch", "complex", "height", "width"))  # type: ignore
        return cell_outputs, previous_state
