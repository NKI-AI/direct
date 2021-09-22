# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch.nn as nn

from direct.nn.mwcnn.mwcnn import MWCNN
from direct.nn.crossdomain.crossdomain import CrossDomainNetwork


class XPDNet(CrossDomainNetwork):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_primal: int = 5,
        num_dual: int = 1,
        num_iter: int = 10,
        use_primal_only: bool = True,
        image_model_architecture: str = "MWCNN",
        **kwargs,
    ):
        if image_model_architecture == "MWCNN":
            image_model_list = nn.ModuleList(
                [
                    nn.Sequential(
                        MWCNN(
                            input_channels=2 * (num_primal + 1),
                            first_conv_hidden_channels=kwargs.get("mwcnn_hidden_channels"),
                            num_scales=kwargs.get("mwcnn_num_scales"),
                            bias=kwargs.get("mwcnn_bias"),
                            batchnorm=kwargs.get("mwcnn_batchnorm"),
                        ),
                        nn.Conv2d(2 * (num_primal + 1), 2 * (num_primal), kernel_size=3, padding=1),
                    )
                    for i in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with 'MWCNN' as an image correction architecture."
                f"Got {image_model_architecture}."
            )

        if use_primal_only:
            kspace_model_list = None
        else:
            raise NotImplementedError("XPDNet is not yet implemented with use_primal_only set to True.")

        super(XPDNet, self).__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
        )
