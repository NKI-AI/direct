# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.data import transforms as T


class MRILogLikelihood(nn.Module):
    def __init__(self, forward_operator, backward_operator):
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        # TODO UGLY
        self.ndim = 2

    def forward(
        self,
        input_image,
        masked_kspace,
        sensitivity_map,
        sampling_mask,
        loglikelihood_scaling=None,
    ):
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
        loglikelihood_scaling : float
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

        coil_dim = 1
        # TODO(gy): Is if statement needed? Do 3D data pass from here?
        spatial_dims = (2, 3) if mul.ndim == 5 else (2, 3, 4)

        mr_forward = torch.where(
            sampling_mask == 0,
            torch.Tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            self.forward_operator(mul, dim=spatial_dims),
        )  # shape (batch, coil, [slice],  height, width, complex)

        error = mr_forward - loglikelihood_scaling * torch.where(
            sampling_mask == 0,
            torch.Tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            masked_kspace,
        )  # shape (batch, coil, [slice],  height, width, complex)

        mr_backward = self.backward_operator(
            error, dim=spatial_dims
        )  # shape (batch, coil, [slice],  height, width, complex)

        if sensitivity_map is not None:
            out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum(coil_dim)
        else:
            out = mr_backward.sum(coil_dim)
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

    def forward(self, x):

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


class MRIReconstruction(nn.Module):
    def __init__(
        self,
        rim_model,
        forward_operator,
        backward_operator,
        x_ch,
        hidden_channels: int = 16,
        length: int = 8,
        depth: int = 1,
        no_parameter_sharing: bool = False,
        instance_norm: bool = False,
        dense_connect: bool = False,
        replication_padding: bool = True,
        image_initialization: str = "zero_filled",
        learned_initializer: bool = False,
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        **kwargs,
    ):
        # TODO: Code quality
        # BODY: Constructor can be called with **kwargs as much as possible. Is currently already done for some variables.
        """
        MRI Reconstruction model based on RIM
        """
        super().__init__()

        # Some other keys are possible. Check here if these are actually relevant for MRI Recon.
        # TODO: Expand this to a larger class
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

        self.model = rim_model(
            x_ch,
            hidden_channels,
            MRILogLikelihood(forward_operator, backward_operator),
            length=length,
            depth=depth,
            no_sharing=no_parameter_sharing,
            instance_norm=instance_norm,
            dense_connection=dense_connect,
            replication_padding=replication_padding,
            **kwargs,
        )
        self.initializer: Optional[nn.Module] = None
        if learned_initializer and initializer_channels is not None and initializer_dilations is not None:
            # List is because of a omegaconf bug.
            self.initializer = RIMInit(
                x_ch,
                hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=depth,
                multiscale_depth=initializer_multiscale,
            )

        self.image_initialization = image_initialization

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

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
        input_image,
        masked_kspace,
        sampling_mask,
        sensitivity_map=None,
        hidden_state=None,
        loglikelihood_scaling=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        input_image
            initial reconstruction by fft or previous rim step
        masked_kspace
            masked k_space
        sensitivity_map
        sampling_mask
        hidden_state
        loglikelihood_scaling

        Returns
        -------

        """
        # Provide input image for the first image
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
        if (self.initializer is not None) and (hidden_state is None):
            hidden_state = self.initializer(
                input_image.permute((0, 4, 1, 2, 3) if input_image.ndim == 5 else (0, 3, 1, 2))
            )  # permute to (batch, complex, [slice], height, width),

        return self.model(
            input_image=input_image,
            masked_kspace=masked_kspace,
            sensitivity_map=sensitivity_map,
            sampling_mask=sampling_mask,
            previous_state=hidden_state,
            loglikelihood_scaling=loglikelihood_scaling,
            **kwargs,
        )
