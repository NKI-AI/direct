# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.types module."""

from direct.types import DirectEnum


class PolicySamplingDimension(DirectEnum):
    ONE_D = "1D"
    TWO_D = "2D"


class PolicySamplingType(DirectEnum):
    STATIC = "static"
    DYNAMIC_2D = "dynamic_2d"
    DYNAMIC_2D_NON_UNIFORM = "dynamic_2d_non_uniform"
    MULTISLICE_2D = "multislice_2d"
    MULTISLICE_2D_NON_UNIFORM = "multislice_2d_non_uniform"
