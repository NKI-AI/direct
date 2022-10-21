# coding=utf-8
# Copyright (c) DIRECT Contributors

from enum import Enum

import torch.nn as nn

from direct.nn.conv.conv import Conv2d
from direct.nn.didn.didn import DIDN
from direct.nn.resnet.resnet import ResNet
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d


class ModelName(str, Enum):
    unet = "unet"
    normunet = "normunet"
    resnet = "resnet"
    didn = "didn"
    conv = "conv"


def _build_model(
    model_architecture_name: ModelName, in_channels: int = 2, out_channels: int = 2, **kwargs
) -> nn.Module:
    model_kwargs = {"in_channels": in_channels, "out_channels": out_channels}
    if model_architecture_name in ["unet", "normunet"]:
        model_architecture = UnetModel2d if model_architecture_name == "unet" else NormUnetModel2d
        model_kwargs.update(
            {
                "num_filters": kwargs.get("unet_num_filters", 32),
                "num_pool_layers": kwargs.get("unet_num_pool_layers", 4),
                "dropout_probability": kwargs.get("unet_dropout", 0.0),
            }
        )
    elif model_architecture_name == "resnet":
        model_architecture = ResNet
        model_kwargs.update(
            {
                "in_channels": in_channels,
                "hidden_channels": kwargs.get("resnet_hidden_channels", 64),
                "num_blocks": kwargs.get("resnet_num_blocks", 15),
                "batchnorm": kwargs.get("resnet_batchnorm", True),
                "scale": kwargs.get("resnet_scale", 0.1),
            }
        )
    elif model_architecture_name == "didn":
        model_architecture = DIDN
        model_kwargs.update(
            {
                "hidden_channels": kwargs.get("didn_hidden_channels", 16),
                "num_dubs": kwargs.get("didn_num_dubs", 6),
                "num_convs_recon": kwargs.get("didn_num_convs_recon", 9),
            }
        )
    else:
        model_architecture = Conv2d
        model_kwargs.update(
            {
                "hidden_channels": kwargs.get("conv_hidden_channels", 64),
                "n_convs": kwargs.get("conv_n_convs", 15),
                "activation": nn.PReLU() if kwargs.get("conv_activation", "prelu") == "prelu" else nn.ReLU(),
                "batchnorm": kwargs.get("conv_batchnorm", False),
            }
        )

    return model_architecture, model_kwargs
