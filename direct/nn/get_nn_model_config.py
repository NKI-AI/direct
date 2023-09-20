# coding=utf-8
# Copyright (c) DIRECT Contributors

from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.nn.conv.conv import Conv2d
from direct.nn.didn.didn import DIDN
from direct.nn.resnet.resnet import ResNet
from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType, UFormerModel
from direct.nn.transformers.vision_transformers import VisionTransformerModel
from direct.nn.types import ActivationType, ModelName
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d


def _get_activation(activation: ActivationType):
    return (
        nn.PReLU()
        if activation == ActivationType.prelu
        else nn.ReLU()
        if activation == ActivationType.relu
        else nn.LeakyReLU()
    )


def _get_model_config(
    model_architecture_name: ModelName, in_channels: int = COMPLEX_SIZE, out_channels: int = COMPLEX_SIZE, **kwargs
) -> nn.Module:
    model_kwargs = {"in_channels": in_channels, "out_channels": out_channels}
    if model_architecture_name in ["unet", "normunet"]:
        model_architecture = UnetModel2d if model_architecture_name == "unet" else NormUnetModel2d
        model_kwargs.update(
            {
                "num_filters": kwargs.get("unet_num_filters", 32),
                "num_pool_layers": kwargs.get("unet_num_pool_layers", 4),
                "dropout_probability": kwargs.get("unet_dropout", 0.0),
                "cwn_conv": kwargs.get("cwn_conv", False),
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
    elif model_architecture_name == "uformer":
        model_architecture = UFormerModel
        model_kwargs.update(
            {
                "patch_size": kwargs.get("patch_size", 256),
                "embedding_dim": kwargs.get("embedding_dim", 32),
                "encoder_depths": kwargs.get("encoder_depths", (2, 2, 2, 2)),
                "encoder_num_heads": kwargs.get("encoder_num_heads", (1, 2, 4, 8)),
                "bottleneck_depth": kwargs.get("bottleneck_depth", 2),
                "bottleneck_num_heads": kwargs.get("bottleneck_num_heads", 16),
                "win_size": kwargs.get("win_size", 8),
                "mlp_ratio": kwargs.get("mlp_ratio", 4.0),
                "qkv_bias": kwargs.get("qkv_bias", True),
                "qk_scale": kwargs.get("qk_scale", None),
                "drop_rate": kwargs.get("drop_rate", 0.0),
                "attn_drop_rate": kwargs.get("attn_drop_rate", 0.0),
                "drop_path_rate": kwargs.get("drop_path_rate", 0.1),
                "patch_norm": kwargs.get("patch_norm", True),
                "token_projection": kwargs.get("token_projection", AttentionTokenProjectionType.linear),
                "token_mlp": kwargs.get("token_mlp", LeWinTransformerMLPTokenType.leff),
                "shift_flag": kwargs.get("shift_flag", True),
                "modulator": kwargs.get("modulator", False),
                "cross_modulator": kwargs.get("cross_modulator", False),
                "normalized": kwargs.get("normalized", True),
            }
        )
    elif model_architecture_name == "vision_transformer":
        model_architecture = VisionTransformerModel
        model_kwargs.update(
            {
                "average_img_size": kwargs.get("average_img_size", 320),
                "patch_size": kwargs.get("patch_size", 10),
                "embedding_dim": kwargs.get("embedding_dim", 64),
                "depth": kwargs.get("depth", 8),
                "num_heads": kwargs.get("num_heads", 9),
                "mlp_ratio": kwargs.get("mlp_ratio", 4.0),
                "qkv_bias": kwargs.get("qkv_bias", True),
                "qk_scale": kwargs.get("qk_scale", None),
                "drop_rate": kwargs.get("drop_rate", 0.0),
                "normalized": kwargs.get("normalized", True),
                "attn_drop_rate": kwargs.get("attn_drop_rate", 0.0),
                "dropout_path_rate": kwargs.get("dropout_path_rate", 0.0),
                "gpsa_interval": kwargs.get("gpsa_interval", (-1, -1)),
                "locality_strength": kwargs.get("locality_strength", 1.0),
                "use_pos_embedding": kwargs.get("use_pos_embedding", True),
            }
        )
    else:
        model_architecture = Conv2d
        model_kwargs.update(
            {
                "hidden_channels": kwargs.get("conv_hidden_channels", 64),
                "n_convs": kwargs.get("conv_n_convs", 15),
                "activation": _get_activation(kwargs.get("conv_activation", ActivationType.relu)),
                "batchnorm": kwargs.get("conv_batchnorm", False),
            }
        )

    return model_architecture, model_kwargs
