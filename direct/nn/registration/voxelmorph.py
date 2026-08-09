# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""VoxelMorph building blocks for nonlinear image registration.

Provides spatial transformers, velocity-field integration, and dense VoxelMorph
networks used by :mod:`direct.nn.registration.registration`.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from direct.types import DirectEnum


class GridSampleMode(DirectEnum):
    """Interpolation modes supported by :func:`torch.nn.functional.grid_sample`."""

    BILINEAR = "bilinear"
    NEAREST = "nearest"
    BICUBIC = "bicubic"


class SpatialTransformer(nn.Module):
    """N-D spatial transformer that warps a source volume with a displacement field."""

    def __init__(
        self,
        size: Sequence[int],
        mode: GridSampleMode = GridSampleMode.BILINEAR,
    ) -> None:
        """Inits :class:`SpatialTransformer`.

        Parameters
        ----------
        size : Sequence[int]
            Spatial size of the sampling grid, e.g. ``(height, width)`` or
            ``(depth, height, width)``.
        mode : GridSampleMode
            Interpolation mode passed to :func:`torch.nn.functional.grid_sample`.
            Default: :attr:`GridSampleMode.BILINEAR`.
        """
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp ``src`` with the displacement field ``flow``.

        Parameters
        ----------
        src : torch.Tensor
            Source image of shape ``(batch, channels, *spatial)``.
        flow : torch.Tensor
            Displacement field of shape ``(batch, ndims, *spatial)``.

        Returns
        -------
        torch.Tensor
            Warped source with the same shape as ``src``.
        """
        # create sampling grid
        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(src.device)
        # new locations
        new_locs = grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrate a stationary velocity field via scaling and squaring."""

    def __init__(self, inshape: Sequence[int], nsteps: int) -> None:
        """Inits :class:`VecInt`.

        Parameters
        ----------
        inshape : Sequence[int]
            Spatial shape of the velocity field.
        nsteps : int
            Number of scaling-and-squaring steps (must be ``>= 0``).
        """
        super().__init__()

        assert nsteps >= 0, f"nsteps should be >= 0, found: {nsteps}"
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Integrate the velocity field ``vec``.

        Parameters
        ----------
        vec : torch.Tensor
            Velocity field of shape ``(batch, ndims, *spatial)``.

        Returns
        -------
        torch.Tensor
            Integrated displacement field with the same shape as ``vec``.
        """
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """Resize a displacement / velocity field and rescale its magnitudes accordingly."""

    def __init__(self, vel_resize: float, ndims: int) -> None:
        """Inits :class:`ResizeTransform`.

        Parameters
        ----------
        vel_resize : float
            Resize factor applied to the field. Values ``< 1`` downsize;
            values ``> 1`` upsize.
        ndims : int
            Spatial dimensionality (``2`` or ``3``).
        """
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = "linear"
        if ndims == 2:
            self.mode = "bi" + self.mode
        elif ndims == 3:
            self.mode = "tri" + self.mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and rescale the transform field ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Transform field of shape ``(batch, ndims, *spatial)``.

        Returns
        -------
        torch.Tensor
            Resized and rescaled transform field.
        """
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class VoxelmorphUnet(nn.Module):
    """A U-Net architecture for VoxelMorph.

    Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels.
    """

    def __init__(
        self,
        inshape: tuple[int, ...] | None = None,
        infeats: int | None = None,
        nb_features: int | None = None,
        nb_levels: int | None = None,
        max_pool: int | list[int] = 2,
        nb_conv_per_level: int = 1,
        half_res: bool = False,
    ) -> None:
        """Inits :class:`VoxelmorphUnet`.

        Parameters
        ----------
        inshape : tuple of int, optional
            Input spatial shape, e.g. ``(192, 192, 192)``.
        infeats : int, optional
            Number of input feature channels.
        nb_features : int, optional
            Base number of U-Net convolutional features. Encoder / decoder
            channel counts are derived from this value and ``nb_levels``.
        nb_levels : int, optional
            Number of levels in the U-Net. Only used when ``nb_features`` is set.
        max_pool : int or list of int
            Max-pooling kernel size(s) per level. Default: ``2``.
        nb_conv_per_level : int
            Number of convolutions per U-Net level. Default: ``1``.
        half_res : bool
            If ``True``, skip the last decoder upsampling. Default: ``False``.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], f"ndims should be one of 1, 2, or 3. found: {ndims}"

        # cache some parameters
        self.half_res = half_res

        enc_nf = [nb_features * (2**i) for i in range(nb_levels)]
        dec_nf = enc_nf[::-1] + [nb_features]

        enc_nf = [nb_features * (2**i) for i in range(nb_levels)]
        dec_nf = enc_nf[::-1] + [nb_features]

        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, f"MaxPool{ndims}d")
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`VoxelmorphUnet`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, infeats, *spatial)``.

        Returns
        -------
        torch.Tensor
            Output feature map of shape ``(batch, final_nf, *spatial)``.
        """
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                # Ensure the dimensions of x and x_history[-1] match for concatenation
                diff_dims = [hx - ux for hx, ux in zip(x_history[-1].shape[2:], x.shape[2:])]
                pad = [(d // 2, d - d // 2) for d in diff_dims]
                pad = [p for sublist in reversed(pad) for p in sublist]  # flatten and reverse
                x = F.pad(x, pad)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(nn.Module):
    """VoxelMorph network for unsupervised nonlinear registration between two images."""

    def __init__(
        self,
        inshape: Sequence[int],
        nb_unet_features: int = 8,
        nb_unet_levels: int = 4,
        nb_unet_conv_per_level: int = 1,
        warp_num_integration_steps: int = 1,
        int_downsize: int = 2,
        src_feats: int = 1,
        trg_feats: int = 1,
        **kwargs: object,
    ) -> None:
        """Inits :class:`VxmDense`.

        Parameters
        ----------
        inshape : Sequence[int]
            Spatial shape of the input images, e.g. ``(height, width)``.
        nb_unet_features : int
            Base number of U-Net features. Default: ``8``.
        nb_unet_levels : int
            Number of U-Net levels. Default: ``4``.
        nb_unet_conv_per_level : int
            Convolutions per U-Net level. Default: ``1``.
        warp_num_integration_steps : int
            Scaling-and-squaring steps for diffeomorphic integration.
            If ``0``, the flow is used directly without integration. Default: ``1``.
        int_downsize : int
            Downsampling factor applied before integration. Default: ``2``.
        src_feats : int
            Number of moving-image channels. Default: ``1``.
        trg_feats : int
            Number of reference-image channels. Default: ``1``.
        **kwargs : object
            Ignored; accepted for config-compatibility.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], f"ndims should be one of 1, 2, or 3. found: {ndims}"

        # configure core unet model
        self.unet_model = VoxelmorphUnet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            nb_conv_per_level=nb_unet_conv_per_level,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, f"Conv{ndims}d")
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if warp_num_integration_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if warp_num_integration_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, warp_num_integration_steps) if warp_num_integration_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Register a moving image sequence to a reference frame.

        Parameters
        ----------
        moving_image : torch.Tensor
            Moving images of shape ``(batch, seq_len, height, width)``.
        reference_image : torch.Tensor
            Reference image of shape ``(batch, height, width)``.

        Returns
        -------
        registered_image : torch.Tensor
            Warped moving images of shape ``(batch, seq_len, height, width)``.
        displacement_field : torch.Tensor
            Predicted displacement fields of shape
            ``(batch, seq_len, 2, height, width)``.
        """
        _, seq_len, _, _ = moving_image.shape

        displacement_field = []
        registered_image = []
        for t in range(seq_len):
            # concatenate inputs and propagate unet
            x = torch.cat([moving_image[:, t : t + 1], reference_image.unsqueeze(1)], dim=1)
            x = self.unet_model(x)

            # transform into flow field
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field
            if self.resize:
                pos_flow = self.resize(pos_flow)

            # integrate to produce diffeomorphic warp
            if self.integrate:
                pos_flow = self.integrate(pos_flow)

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)

            displacement_field.append(pos_flow)
            registered_image.append(self.transformer(moving_image[:, t : t + 1], pos_flow))

        displacement_field = torch.stack(displacement_field, dim=1)
        registered_image = torch.cat(registered_image, dim=1)

        return registered_image, displacement_field


class ConvBlock(nn.Module):
    """Convolutional block followed by a leaky ReLU, used inside the VoxelMorph U-Net."""

    def __init__(self, ndims: int, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Inits :class:`ConvBlock`.

        Parameters
        ----------
        ndims : int
            Spatial dimensionality (``1``, ``2``, or ``3``).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int
            Convolution stride. Default: ``1``.
        """
        super().__init__()

        Conv = getattr(nn, f"Conv{ndims}d")
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and leaky ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """
        out = self.main(x)
        out = self.activation(out)
        return out
