# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.types module."""

from direct.types import DirectEnum


class PolicySamplingDimension(DirectEnum):
    ONE_D = "1D"
    TWO_D = "2D"


class PolicySamplingType(DirectEnum):
    DYNAMIC = "dynamic"
    NON_DYNAMIC = "non_dynamic"
