# Copyright (c) DIRECT Contributors

"""direct.nn.types module."""

from direct.types import DirectEnum


class ActivationType(DirectEnum):
    RELU = "relu"
    PRELU = "prelu"
    LEAKYRELU = "leaky_relu"


class ModelName(DirectEnum):
    UNET = "unet"
    NORMUNET = "normunet"
    RESNET = "resnet"
    DIDN = "didn"
    CONV = "conv"


class InitType(DirectEnum):
    INPUTIMAGE = "input_image"
    SENSE = "sense"
    ZEROFILLED = "zero_filled"
    ZEROS = "zeros"
