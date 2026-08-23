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

"""Implementation of the MEDL network [#]_ for MRI reconstruction.

Expansion to 3D is supported.

References:
    .. [#] Qiao, X., Huang, Y., Li, W.: MEDL‐Net: A model‐based neural network for MRI reconstruction with enhanced deep
        learned regularizers. Magnetic Resonance in Med. 89, 2062–2075 (2023). https://doi.org/10.1002/mrm.29575
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import apply_mask, expand_operator, reduce_operator
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.nn.unet.unet_3d import NormUnetModel3d, UnetModel3d
from direct.types import DirectEnum


class MEDLType(DirectEnum):
    """Supported MEDL network dimensionalities."""

    TWO_DIMENSIONAL = "2D"
    THREE_DIMENSIONAL = "3D"


class GD(nn.Module):
    """Gradient descent block for MEDL.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        medl_type: Type of MEDL network. Can be either MEDLType.TWO_DIMENSIONAL or MEDLType.THREE_DIMENSIONAL. Default is
            ``MEDLType.TWO_DIMENSIONAL``.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        medl_type: MEDLType = MEDLType.TWO_DIMENSIONAL,
    ) -> None:
        """Inits :class:`GD`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            medl_type: Type of MEDL network. Can be either MEDLType.TWO_DIMENSIONAL or MEDLType.THREE_DIMENSIONAL. Default is
                ``MEDLType.TWO_DIMENSIONAL``.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._complex_dim = -1

        if medl_type == MEDLType.TWO_DIMENSIONAL:
            self._spatial_dims = (2, 3)
        else:
            self._spatial_dims = (3, 4)

        self.lambda_step = nn.Parameter(torch.tensor([0.5]))

    def _forward_operator(
        self,
        image: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward operator of :class:`GD`.

        This will apply the expand operator, compute the k-space by applying the forward Fourier transform,
        and apply the sampling mask.

        Args:
            image: Image tensor of shape (batch, [time/slice,] height, width, [complex=2]).
            sampling_mask: Sampling mask tensor of shape (batch, [time/slice or 1, height, width, 1).
            sensitivity_map: Sensitivity map tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).

        Returns:
            k-space tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).
        """
        return apply_mask(
            self.forward_operator(
                expand_operator(image, sensitivity_map, dim=self._coil_dim),  # ty: ignore[invalid-argument-type]
                dim=self._spatial_dims,  # ty: ignore[unknown-argument]
            ),
            sampling_mask,
            return_mask=False,
        )

    def _backward_operator(
        self,
        kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Backward operator of :class:`GD`.

        This will apply the sampling mask, compute the image by applying the adjoint Fourier transform,
        and apply the reduce operator using the sensitivity map.

        Args:
            kspace: k-space tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).
            sampling_mask: Sampling mask tensor of shape (batch, [time/slice or 1,] height, width, 1).
            sensitivity_map: Sensitivity map tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).

        Returns:
            Image tensor of shape (batch, [time/slice,] height, width, [complex=2]).
        """
        return reduce_operator(
            self.backward_operator(
                apply_mask(kspace, sampling_mask, return_mask=False),  # ty: ignore[invalid-argument-type]
                dim=self._spatial_dims,  # ty: ignore[unknown-argument]
            ),
            sensitivity_map,
            dim=self._coil_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`GD`.

        Args:
            x: Image tensor of shape (batch, [time/slice,] height, width, [complex=2]).
            masked_kspace: Masked k-space tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).
            sampling_mask: Sampling mask tensor of shape (batch, [time/slice or 1,] height, width, 1).
            sensitivity_map: Sensitivity map tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).

        Returns:
            Image tensor of shape (batch, [time/slice,] height, width, [complex=2]).
        """

        Ax = self._forward_operator(x, sampling_mask, sensitivity_map)
        ATAx_y = self._backward_operator(Ax - masked_kspace, sampling_mask, sensitivity_map)
        r = x - self.lambda_step * ATAx_y

        return r


class VarBlock(nn.Module):
    """Varitaional block for MEDL.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        iterations: Number of iterations for the block. Default is ``3``.
        unet_num_filters: Number of filters in the U-Net. Default is ``18``.
        unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
        unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
        unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
        medl_type: Type of MEDL network. Can be either MEDLType.TWO_DIMENSIONAL or MEDLType.THREE_DIMENSIONAL.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        iterations: int = 3,
        unet_num_filters: int = 18,
        unet_num_pool_layers: int = 4,
        unet_dropout: float = 0.0,
        unet_norm: bool = False,
        medl_type: MEDLType = MEDLType.TWO_DIMENSIONAL,
    ) -> None:
        """Inits :class:`VarBlock`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            iterations: Number of iterations for the block. Default is ``3``.
            unet_num_filters: Number of filters in the U-Net. Default is ``18``.
            unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
            unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
            unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
            medl_type: Type of MEDL network. Can be either MEDLType.TWO_DIMENSIONAL or MEDLType.THREE_DIMENSIONAL.
        """
        super().__init__()
        self.iterations = iterations
        self.cnn = nn.ModuleList()
        self.gd_blocks = nn.ModuleList()

        if medl_type == MEDLType.TWO_DIMENSIONAL:
            unet = UnetModel2d if not unet_norm else NormUnetModel2d
        else:
            unet = UnetModel3d if not unet_norm else NormUnetModel3d

        for i in range(self.iterations):
            self.cnn.append(
                unet(
                    in_channels=4 + i * COMPLEX_SIZE,
                    out_channels=COMPLEX_SIZE,
                    num_filters=unet_num_filters,
                    num_pool_layers=unet_num_pool_layers,
                    dropout_probability=unet_dropout,
                )
            )
            self.gd_blocks.append(GD(forward_operator, backward_operator, medl_type))

        self.reg = unet(
            in_channels=COMPLEX_SIZE,
            out_channels=COMPLEX_SIZE,
            num_filters=unet_num_filters,
            num_pool_layers=unet_num_pool_layers,
            dropout_probability=unet_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`VarBlock`.

        Args:
            x: Current image tensor of shape (batch, [time/slice,] height, width, [complex=2]).
            masked_kspace: Masked k-space tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).
            sampling_mask: Sampling mask tensor of shape (batch, [time/slice or 1,] height, width, 1).
            sensitivity_map: Sensitivity map tensor of shape (batch, coil, [time/slice,] height, width, [complex=2]).

        Returns:
            Image tensor of shape (batch, [time/slice,] height, width, [complex=2]).
        """

        gds = []
        current_x = x

        for i in range(self.iterations):
            x = self.gd_blocks[i](x, masked_kspace, sampling_mask, sensitivity_map)
            gds.append(x)
            cnn_out = self.cnn[i](
                torch.cat((current_x, *gds), dim=-1).permute((0, 4, 1, 2, 3) if x.dim() == 5 else (0, 3, 1, 2))
            )
            x = x + cnn_out.permute((0, 2, 3, 4, 1) if x.dim() == 5 else (0, 2, 3, 1))

        out = self.reg(x.permute((0, 4, 1, 2, 3) if x.dim() == 5 else (0, 3, 1, 2)))
        return out.permute((0, 2, 3, 4, 1) if x.dim() == 5 else (0, 2, 3, 1))


class MEDL(nn.Module):
    """Model-based neural network for MRI reconstruction with enhanced deep learned regularizers.

    Adapted from the original implementation in [#]_.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        iterations: Number of iterations for each Variational Block gradient descent. Default is ``4``.
        num_layers: Number of layers in the MEDL network. Must be equal to the length of iterations if iterations is a
            tuple. Default is ``3``.
        unet_num_filters: Number of filters in the U-Net. Default is ``18``.
        unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
        unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
        unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
        medl_type: Type of MEDL network. Can be either MEDLType.TWO_DIMENSIONAL or MEDLType.THREE_DIMENSIONAL.

    References:
        .. [#] Qiao, X., Huang, Y., Li, W.: MEDL‐Net: A model‐based neural network for MRI reconstruction with enhanced deep
            learned regularizers. Magnetic Resonance in Med. 89, 2062–2075 (2023). https://doi.org/10.1002/mrm.29575
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        iterations: int | tuple[int, ...] = 4,
        num_layers: int = 3,
        unet_num_filters: int = 18,
        unet_num_pool_layers: int = 4,
        unet_dropout: float = 0.0,
        unet_norm: bool = False,
        medl_type: MEDLType = MEDLType.TWO_DIMENSIONAL,
        **kwargs,
    ) -> None:
        """Inits :class:`MEDL`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            iterations: Number of iterations for each Variational Block gradient descent. Default is ``4``.
            num_layers: Number of layers in the MEDL network. Must be equal to the length of iterations if iterations is a
                tuple. Default is ``3``.
            unet_num_filters: Number of filters in the U-Net. Default is ``18``.
            unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
            unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
            unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
            medl_type: Type of MEDL network. Can be either MEDLType.TWO_DIMENSIONAL or MEDLType.THREE_DIMENSIONAL.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key != "model_name":
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.iterations = iterations
        self.blocks = nn.ModuleList()

        if isinstance(iterations, int):
            iterations = [iterations] * num_layers  # ty: ignore[invalid-assignment]
        else:
            if len(iterations) != num_layers:
                raise ValueError(
                    f"Number of iterations must be equal to the number of layers."
                    f"Received {len(iterations)} iterations and {num_layers} layers."
                )

        for i in range(num_layers):
            self.blocks.append(
                VarBlock(
                    forward_operator,
                    backward_operator,
                    iterations=iterations[i],  # ty: ignore[not-subscriptable]
                    unet_num_filters=unet_num_filters,
                    unet_num_pool_layers=unet_num_pool_layers,
                    unet_dropout=unet_dropout,
                    unet_norm=unet_norm,
                    medl_type=medl_type,
                )
            )

        if medl_type == MEDLType.TWO_DIMENSIONAL:
            self._spatial_dims = (2, 3)
        else:
            self._spatial_dims = (3, 4)

        self._coil_dim = 1
        self.backward_operator = backward_operator

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Computes forward pass of :class:`MEDL`.

        Args:
            masked_kspace: Masked k-space of shape (N, coil, [slice/time,] height, width, complex=2).
            sampling_mask: Sampling mask of shape (N, 1, [1 or slice/time,] height, width, 1).
            sensitivity_map: Sensitivity map of shape (N, coil, [slice/time,] height, width, complex=2).

        Returns:
            List of output images each of shape (N, [slice/time,] height, width, complex=2).
        """
        x = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),  # ty: ignore[invalid-argument-type, unknown-argument]
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        )  # SENSE reconstruction

        out = []
        for block in self.blocks:
            x = block(x, masked_kspace, sampling_mask, sensitivity_map) + x
            out.append(x)

        return out


class MEDL2D(MEDL):
    """MEDL network for 2D MRI reconstruction.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        iterations: Number of iterations for each Variational Block gradient descent. Default is ``4``.
        num_layers: Number of layers in the MEDL network. Must be equal to the length of iterations if iterations is a
            tuple. Default is ``3``.
        unet_num_filters: Number of filters in the U-Net. Default is ``18``.
        unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
        unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
        unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        iterations: int | tuple[int, ...] = 4,
        num_layers: int = 3,
        unet_num_filters: int = 18,
        unet_num_pool_layers: int = 4,
        unet_dropout: float = 0.0,
        unet_norm: bool = False,
        **kwargs,
    ) -> None:
        """Inits :class:`MEDL2D`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            iterations: Number of iterations for each Variational Block gradient descent. Default is ``4``.
            num_layers: Number of layers in the MEDL network. Must be equal to the length of iterations if iterations is a
                tuple. Default is ``3``.
            unet_num_filters: Number of filters in the U-Net. Default is ``18``.
            unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
            unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
            unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
        """
        super().__init__(
            forward_operator,
            backward_operator,
            iterations=iterations,
            num_layers=num_layers,
            unet_num_filters=unet_num_filters,
            unet_num_pool_layers=unet_num_pool_layers,
            unet_dropout=unet_dropout,
            unet_norm=unet_norm,
            medl_type=MEDLType.TWO_DIMENSIONAL,
            **kwargs,
        )


class MEDL3D(MEDL):
    """MEDL network for 3D MRI reconstruction.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        iterations: Number of iterations for each Variational Block gradient descent. Default is ``4``.
        num_layers: Number of layers in the MEDL network. Must be equal to the length of iterations if iterations is a
            tuple. Default is ``3``.
        unet_num_filters: Number of filters in the U-Net. Default is ``18``.
        unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
        unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
        unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        iterations: int | tuple[int, ...] = 4,
        num_layers: int = 3,
        unet_num_filters: int = 18,
        unet_num_pool_layers: int = 4,
        unet_dropout: float = 0.0,
        unet_norm: bool = False,
        **kwargs,
    ) -> None:
        """Inits :class:`MEDL3D`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            iterations: Number of iterations for each Variational Block gradient descent. Default is ``4``.
            num_layers: Number of layers in the MEDL network. Must be equal to the length of iterations if iterations is a
                tuple. Default is ``3``.
            unet_num_filters: Number of filters in the U-Net. Default is ``18``.
            unet_num_pool_layers: Number of pooling layers in the U-Net. Default is ``4``.
            unet_dropout: Dropout probability in the U-Net. Default is ``0.0``.
            unet_norm: Whether to use normalization in the U-Net. Default is ``False``.
        """
        super().__init__(
            forward_operator,
            backward_operator,
            iterations=iterations,
            num_layers=num_layers,
            unet_num_filters=unet_num_filters,
            unet_num_pool_layers=unet_num_pool_layers,
            unet_dropout=unet_dropout,
            unet_norm=unet_norm,
            medl_type=MEDLType.THREE_DIMENSIONAL,
            **kwargs,
        )
