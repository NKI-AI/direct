# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any, Iterable

from direct.data import transforms as T


class MRILogLikelihood(nn.Module):
    def __init__(self, forward_operator, backward_operator):
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        # TODO UGLY
        self.ndim = 2

    # TODO(jt): This is a commonality, split off.
    @property
    def names_image_complex_last(self):
        if self.ndim == 2:
            return ["batch", "height", "width", "complex"]
        if self.ndim == 3:
            return ["batch", "slice", "height", "width", "complex"]
        raise NotImplementedError

    @property
    def names_data_complex_last(self):
        if self.ndim == 2:
            return ["batch", "coil", "height", "width", "complex"]
        if self.ndim == 3:
            return ["batch", "coil", "slice", "height", "width", "complex"]
        raise NotImplementedError

    @property
    def names_image_complex_channel(self):
        if self.ndim == 2:
            return ["batch", "complex", "height", "width"]
        if self.ndim == 3:
            return ["batch", "complex", "slice", "height", "width"]
        raise NotImplementedError

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
            Initial or previous iteration of image.
        masked_kspace : torch.tensor
            Masked k-space.
        sensitivity_map : torch.tensor
        sampling_mask : torch.tensor
        loglikelihood_scaling : float
            Multiplier for loglikelihood, for instance for the k-space noise.

        Returns
        -------
        torch.Tensor
        """
        if "slice" in input_image.names:
            self.ndim = 3

        input_image = input_image.align_to(*self.names_image_complex_last)
        sensitivity_map = sensitivity_map.align_to(*self.names_data_complex_last)
        masked_kspace = masked_kspace.align_to(*self.names_data_complex_last)

        loglikelihood_scaling = loglikelihood_scaling.align_to(*self.names_data_complex_last)

        # We multiply by the loglikelihood_scaling here to prevent fp16 information loss,
        # as this value is typically <<1, and the operators are linear.
        # input_image is a named tensor with names ('batch', 'coil', 'height', 'width', 'complex')
        mul = loglikelihood_scaling.align_as(sensitivity_map) * T.complex_multiplication(
            sensitivity_map, input_image.align_as(sensitivity_map)
        )

        # TODO: Named tensor: this needs a fix once this exists.
        mul_names = mul.names
        mr_forward = torch.where(
            sampling_mask.rename(None) == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            self.forward_operator(mul).rename(None),
        )

        error = mr_forward - loglikelihood_scaling * torch.where(
            sampling_mask.rename(None) == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            masked_kspace.rename(None),
        )

        error = error.refine_names(*mul_names)
        mr_backward = self.backward_operator(error)

        if sensitivity_map is not None:
            out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum("coil")
        else:
            out = mr_backward.sum("coil")

        return out.align_to(*self.names_image_complex_channel)  # noqa


class RIMInit(nn.Module):
    def __init__(
        self,
        x_ch,
        out_ch,
        channels=(32, 32, 64, 64),
        dilations=(1, 1, 2, 4),
        depth=2,
        multiscale_depth=1,
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
            Channels in the convolutional layers of initializer.
        dilations: tuple
            Dilations of the convolutional layers of the initializer.
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
        # TODO: Named tensor
        names = x.names
        x = x.rename(None)
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
        initializer_channels: Iterable[Any] = (32, 32, 64, 64),
        initializer_dilations: Iterable[Any] = (1, 1, 2, 4),
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

        if not learned_initializer:
            self.initializer = None
        else:
            # List is because of a omegaconf bug.
            self.initializer = RIMInit(
                x_ch,
                hidden_channels,
                channels=list(initializer_channels),
                dilations=list(initializer_dilations),
                depth=depth,
                multiscale_depth=initializer_multiscale,
            )

        self.image_initialization = image_initialization

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

    def compute_sense_init(self, kspace, sensitivity_map):
        input_image = T.complex_multiplication(
            T.conjugate(sensitivity_map),
            self.backward_operator(kspace),
        ).sum("coil")
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
                input_image = self.compute_sense_init(masked_kspace, sensitivity_map)
            elif self.image_initialization == "input_kspace":
                if "initial_kspace" not in kwargs:
                    raise ValueError(
                        f"`'initial_kspace` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = self.compute_sense_init(kwargs["initial_kspace"], sensitivity_map)
            elif self.image_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = kwargs["initial_image"]

            elif self.image_initialization == "zero_filled":
                input_image = self.backward_operator(masked_kspace).sum("coil")
            else:
                raise ValueError(
                    f"Unknown image_initialization. Expected `sense`, `input_kspace`, `'input_image` or `zero_filled`. "
                    f"Got {self.image_initialization}."
                )

        # Provide an initialization for the first hidden state.
        if (self.initializer is not None) and (hidden_state is None):
            hidden_state = self.initializer(
                input_image.align_to("batch", "complex", "height", "width"),
            )

        return self.model(
            input_image=input_image,
            masked_kspace=masked_kspace,
            sensitivity_map=sensitivity_map,
            sampling_mask=sampling_mask,
            previous_state=hidden_state,
            loglikelihood_scaling=loglikelihood_scaling,
            **kwargs,
        )
