# coding=utf-8
# Copyright (c) DIRECT Contributors

from torch import nn

from direct.nn.types import ActivationType


def get_activation_from_type(act_type: ActivationType) -> nn.Module:
    """Returns activation (nn.Module) from input.

    Parameters
    ----------
    act_type : ActivationType
        Activation type.

    Returns
    -------
    nn.Module
    """
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "leaky_relu":
        return nn.LeakyReLU()
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "prelu":
        return nn.PReLU()
    return nn.Tanh()
