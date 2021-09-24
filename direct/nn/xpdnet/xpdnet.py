# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import torch.nn as nn

from direct.nn.mwcnn.mwcnn import MWCNN
from direct.nn.crossdomain.crossdomain import CrossDomainNetwork


class KspaceConv(nn.Module):
    """
    Simple 2D convolutional model to be used with k-space data. Batch and coil dimensions are merged.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, n_convs=3):
        super(KspaceConv, self).__init__()

        self.conv = []

        for i in range(n_convs):

            self.conv.append(
                nn.Conv2d(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels if i != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if i != n_convs - 1:
                self.conv.append(nn.PReLU())

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = x.clone()
        batch, coil, height, width, complex = x.size()
        x = x.reshape(batch * coil, height, width, complex).permute(0, 3, 1, 2)
        x = self.conv(x).permute(0, 2, 3, 1)

        return x.reshape(batch, coil, height, width, -1)


class XPDNet(CrossDomainNetwork):
    """
    XPDNet as implemented in https://arxiv.org/abs/2010.07290.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_primal: int = 5,
        num_dual: int = 1,
        num_iter: int = 10,
        use_primal_only: bool = True,
        image_model_architecture: str = "MWCNN",
        kspace_model_architecture: Optional[str] = None,
        **kwargs,
    ):
        """

        :param forward_operator: Callable
                    Forward Operator.
        :param backward_operator: Callable
                    Backward Operator.
        :param num_primal: int
                    Number of primal networks.
        :param num_dual: int
                    Number of dual networks.
        :param num_iter: int
                    Number of unrolled iterations.
        :param use_primal_only: bool
                    If set to True no dual-kspace model is used. Default: True.
        :param image_model_architecture: str
                    Primal-image model architecture. Currently only implemented for MWCNN. Default: 'MWCNN'.
        :param kspace_model_architecture: str
                    Dual-kspace model architecture. Currently only implemented for KspaceConv.
                    If use_primal_only == True this is omitted. Default: None.
        :param kwargs: str
                Keyword arguments for model architectures.
        """
        if use_primal_only:
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "conv":
            kspace_model_list = nn.ModuleList(
                [
                    KspaceConv(
                        2 * (num_dual + num_primal + 1),
                        2 * num_dual,
                        kwargs.get("kspace_conv_hidden_channels", 64),
                        kwargs.get("kspace_conv_n_convs", 4),
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError("XPDNet is currently implemented for kspace_model_architecture == 'conv'.")

        if image_model_architecture == "MWCNN":
            image_model_list = nn.ModuleList(
                [
                    nn.Sequential(
                        MWCNN(
                            input_channels=2 * (num_primal + num_dual),
                            first_conv_hidden_channels=kwargs.get("mwcnn_hidden_channels"),
                            num_scales=kwargs.get("mwcnn_num_scales"),
                            bias=kwargs.get("mwcnn_bias"),
                            batchnorm=kwargs.get("mwcnn_batchnorm"),
                        ),
                        nn.Conv2d(2 * (num_primal + num_dual), 2 * (num_primal), kernel_size=3, padding=1),
                    )
                    for i in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN'."
                f"Got {image_model_architecture}."
            )

        super(XPDNet, self).__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
        )
