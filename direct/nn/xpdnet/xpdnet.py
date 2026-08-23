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
"""direct.nn.xpdnet.xpdnet module."""

from __future__ import annotations

import torch
from torch import nn

from direct.nn.conv.conv import Conv2d
from direct.nn.conv.modulated import ModConvActivation, ModConvType, ModulationParams
from direct.nn.crossdomain.crossdomain import CrossDomainNetwork
from direct.nn.crossdomain.multicoil import MultiCoil
from direct.nn.didn.didn import DIDN
from direct.nn.mwcnn.mwcnn import MWCNN
from direct.types import FFTOperator


class XPDNetPrimalBlock(nn.Module):
    """Primal image block: MWCNN feature extractor followed by a channel projection."""

    def __init__(
        self,
        mwcnn: MWCNN,
        out_conv: nn.Conv2d,
        conv_modulation: ModConvType = ModConvType.NONE,
    ) -> None:
        """Initialize the instance.

        Args:
            mwcnn: Mwcnn.
            out_conv: Out conv.
            conv_modulation: Conv modulation.

        Returns:
            ``None``.
        """
        super().__init__()
        self.mwcnn = mwcnn
        self.out_conv = out_conv
        self.conv_modulation = conv_modulation

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            x: X.
            y: Y.

        Returns:
            The result.
        """
        if self.conv_modulation != ModConvType.NONE:
            x = self.mwcnn(x, y)
        else:
            x = self.mwcnn(x)
        return self.out_conv(x)


class XPDNet(CrossDomainNetwork):
    """XPDNet as implemented in [#]_.

    References:
        .. [#] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI Challenge.” ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        num_primal: int = 5,
        num_dual: int = 1,
        num_iter: int = 10,
        use_primal_only: bool = True,
        image_model_architecture: str = "MWCNN",
        kspace_model_architecture: str | None = None,
        normalize: bool = False,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        **kwargs,
    ):
        """Inits :class:`XPDNet`.

        Args:
            forward_operator: Forward Operator.
            backward_operator: Backward Operator.
            num_primal: Number of primal networks.
            num_dual: Number of dual networks.
            num_iter: Number of unrolled iterations.
            use_primal_only: If set to ``True`` no dual-kspace model is used. Default is ``True``.
            image_model_architecture: Primal-image model architecture. Currently only implemented for MWCNN. Default is
                ``'MWCNN'``.
            kspace_model_architecture: Dual-kspace model architecture. Currently only implemented for CONV and DIDN.
            normalize: Normalize input. Default is ``False``.
            conv_modulation: Modulation type for convolutional sub-networks.
            aux_in_features: Number of auxiliary conditioning features.
            fc_hidden_features: Hidden features in the modulation MLP.
            fc_groups: Modulation MLP groups. Default is ``1``.
            fc_activation: Modulation MLP activation. Default is ``SIGMOID``.
            num_weights: Number of weight bases for SUM modulation.
            kwargs: Keyword arguments for model architectures.

        Returns:
            ``None``.
        """
        modulation_params = ModulationParams(
            modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        if use_primal_only:
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "CONV":
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        Conv2d(
                            2 * (num_dual + num_primal + 1),
                            2 * num_dual,
                            kwargs.get("dual_conv_hidden_channels", 16),
                            kwargs.get("dual_conv_n_convs", 4),
                            batchnorm=kwargs.get("dual_conv_batchnorm", False),
                            modulation_params=modulation_params,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        DIDN(
                            in_channels=2 * (num_dual + num_primal + 1),
                            out_channels=2 * num_dual,
                            hidden_channels=kwargs.get("dual_didn_hidden_channels", 16),
                            num_dubs=kwargs.get("dual_didn_num_dubs", 6),
                            num_convs_recon=kwargs.get("dual_didn_num_convs_recon", 9),
                            modulation_params=modulation_params,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )

        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )
        if image_model_architecture == "MWCNN":
            image_model_list = nn.ModuleList(
                [
                    XPDNetPrimalBlock(
                        MWCNN(
                            input_channels=2 * (num_primal + num_dual),
                            first_conv_hidden_channels=kwargs.get("mwcnn_hidden_channels", 32),
                            num_scales=kwargs.get("mwcnn_num_scales", 4),
                            bias=kwargs.get("mwcnn_bias", False),
                            batchnorm=kwargs.get("mwcnn_batchnorm", False),
                            modulation_params=modulation_params,
                        ),
                        nn.Conv2d(
                            2 * (num_primal + num_dual),
                            2 * num_primal,
                            kernel_size=3,
                            padding=1,
                        ),
                        conv_modulation=conv_modulation,
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN'."
                f"Got {image_model_architecture}."
            )
        super().__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
            normalize_image=normalize,
            conv_modulation=conv_modulation,
        )
