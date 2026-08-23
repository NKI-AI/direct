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
"""direct.types module."""

import pathlib
from collections.abc import Sequence
from enum import Enum
from typing import Protocol, Self

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig
from torch import nn
from torch.amp import GradScaler

DictOrDictConfig = dict | DictConfig
Number = float | int
IntOrTuple = int | tuple
PathOrString = pathlib.Path | str
FileOrUrl = PathOrString
HasStateDict = nn.Module | torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler | GradScaler
TensorOrNone = None | torch.Tensor
TensorOrNdarray = torch.Tensor | np.ndarray


# fmt: off
class FFTOperator(Protocol):

    """Protocol satisfied by :func:`direct.data.transforms.fft2` and :func:`ifft2`.

    Spelling out the operator's signature lets static type checkers reason
    about the ``dim`` / ``centered`` / ``normalized`` keyword arguments that
    every engine forwards.
    """
# fmt: on

    def __call__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        data: torch.Tensor,
        dim: Sequence[int] = ...,
        centered: bool = ...,
        normalized: bool = ...,
        complex_input: bool = ...,
    ) -> torch.Tensor:
        """Apply the Fourier operator to ``data``.

        Args:
            data: Data.
            dim: Dim.
            centered: Centered.
            normalized: Normalized.
            complex_input: Complex input.

        Returns:
            The result.
        """
        ...


class DirectEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str) -> Self | None:
        """From str.

        Args:
            value: Value.

        Returns:
            The result.
        """
        statuses = cls.__members__.keys()
        for st in statuses:
            if st.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        """Return whether this object equals ``other``.

        Args:
            other: Other.

        Returns:
            The result.
        """
        if isinstance(other, Enum):
            _other = str(other.value)
        else:
            _other = str(other)
        return bool(self.value.lower() == _other.lower())

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        """Return the hash of this object.

        Returns:
            The result.
        """
        return hash(self.value.lower())


class KspaceKey(DirectEnum):
    """KspaceKey."""
    ACS_KSPACE = "acs_kspace"
    KSPACE = "kspace"
    MASKED_KSPACE = "masked_kspace"
    REFERENCE_KSPACE = "reference_kspace"


class TransformKey(DirectEnum):
    # K-space keys
    """TransformKey."""
    ACS_KSPACE = "acs_kspace"
    KSPACE = "kspace"
    MASKED_KSPACE = "masked_kspace"
    # Mask keys
    SAMPLING_MASK = "sampling_mask"
    ACS_MASK = "acs_mask"
    PADDING = "padding"
    # Image keys
    TARGET = "target"
    # Other keys
    SENSITIVITY_MAP = "sensitivity_map"
    SCALING_FACTOR = "scaling_factor"
    # Registration keys
    DISPLACEMENT_FIELD = "displacement_field"
    REFERENCE_IMAGE = "reference_image"
    REFERENCE_KSPACE = "reference_kspace"
    MOVING_IMAGE = "moving_image"
    WARPED_IMAGE = "warped_image"
    # Other keys
    ACCELERATION = "acceleration"
    CENTER_FRACTION = "center_fraction"


class MaskFuncMode(DirectEnum):
    """MaskFuncMode."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    MULTISLICE = "multislice"


class RangeMode(DirectEnum):
    """RangeMode."""
    DISCRETE = "discrete"
    UNIFORM = "uniform"
    LINEAR = "linear"


class IntegerListOrTupleStringMeta(type):
    """Metaclass for the :class:`IntegerListOrTupleString` class.

    Returns:
        ``True`` if the instance is a valid representation of IntegerListOrTupleString, ``False`` otherwise.
    """

    def __instancecheck__(cls, instance):
        """Check if the given instance is a valid representation of an IntegerListOrTupleString.

        Args:
            cls: The class being checked, i.e., IntegerListOrTupleStringMeta.
            instance: The instance being checked.

        Returns:
            ``True`` if the instance is a valid representation of IntegerListOrTupleString, ``False`` otherwise.
        """
        if isinstance(instance, str):
            try:
                assert (instance.startswith("[") and instance.endswith("]")) or (
                    instance.startswith("(") and instance.endswith(")")
                )
                elements = instance.strip()[1:-1].split(",")
                integers = [int(element) for element in elements]
                return all(isinstance(num, int) for num in integers)
            except (AssertionError, ValueError, AttributeError):
                pass
        return False


class IntegerListOrTupleString(metaclass=IntegerListOrTupleStringMeta):
    """IntegerListOrTupleString class represents a list or tuple of integers based on a string representation.

    Examples:
        s1 = "[1, 2, 45, -1, 0]"
        print``(isinstance(s1, IntegerListOrTupleString)``)  # ``True``
        print``(IntegerListOrTupleString(s1)``)  # [1, 2, 45, -1, 0]
        print``(type(IntegerListOrTupleString(s1)``))  # <class ``'list'``>
        print``(type(IntegerListOrTupleString(s1)``[0]))  # <class ``'int'``>

        s2 = "(10, -9, 20)"
        print``(isinstance(s2, IntegerListOrTupleString)``)  # ``True``
        print``(IntegerListOrTupleString(s2)``)  # (10, -9, 20)
        print``(type(IntegerListOrTupleString(s2)``))  # <class ``'tuple'``>
        print``(type(IntegerListOrTupleString(s2)``[0]))  # <class ``'int'``>

        s3 = "[a, we, 2]"
        print``(isinstance(s3, IntegerListOrTupleString)``)  # ``False``

        s4 = "(1, 2, 3]"
        print``(isinstance(s4 IntegerListOrTupleString)``)  # ``False``
    """

    def __new__(cls, string):
        """Create a new instance of IntegerListOrTupleString based on the given string representation.

        Args:
            string: The string representation of the integer list or tuple.

        Returns:
            A new instance of IntegerListOrTupleString.
        """
        list_or_tuple = list if string.startswith("[") else tuple
        string = string.strip()[1:-1]  # Remove outer brackets
        elements = string.split(",")
        integers = [int(element) for element in elements]
        return list_or_tuple(integers)
