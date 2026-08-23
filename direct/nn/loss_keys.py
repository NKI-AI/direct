# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""Helpers for key-based loss lookup from model/data dictionaries."""

from collections.abc import Callable
from typing import Any

# Default source keys used when partitioning reconstruction vs registration losses.
RECONSTRUCTION_SOURCE_KEYS = frozenset({"output_image", "output_kspace"})
REGISTRATION_SOURCE_KEYS = frozenset({"registered_image", "registered_target", "displacement_field", "output_image"})


class KeyedLossFn:
    """Callable loss wrapper that stores which dict keys to compare."""

    def __init__(self, fn: Callable[..., Any], source_key: str, target_key: str | None) -> None:
        """Inits :class:`KeyedLossFn`.

        Args:
            fn: Underlying loss function.
            source_key: Dictionary key for the model output tensor.
            target_key: str | None Dictionary key for the target tensor, or ``None`` for unsupervised losses.
        """
        self.fn = fn
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped loss function."""
        return self.fn(*args, **kwargs)


def resolve_loss_keys(
    function_name: str,
    source_key: str | None = None,
    target_key: str | None = None,
) -> tuple[str, str | None]:
    """Infer ``(source_key, target_key)`` from a loss name when not set explicitly.

    Naming conventions match historical routing in :meth:`MRIModelEngine.compute_loss_on_data`:

    * names containing ``kspace`` → ``output_kspace`` vs ``kspace``
    * names containing ``displacement_field`` → ``displacement_field`` vs ``displacement_field``
    * otherwise → ``output_image`` vs ``target``

    Args:
        function_name: Loss function name used for heuristic key resolution.
        source_key: Explicit source key. When ``None``, inferred from ``function_name``.
        target_key: Explicit target key. When ``None``, inferred from ``function_name``.

    Returns:
        Resolved ``(source_key, target_key)`` pair.
    """
    name = str(function_name)
    if source_key is None:
        if "kspace" in name:
            source_key = "output_kspace"
        elif "displacement_field" in name:
            source_key = "displacement_field"
        else:
            source_key = "output_image"
    if target_key is None:
        if "kspace" in name:
            target_key = "kspace"
        elif "displacement_field" in name:
            target_key = "displacement_field"
        else:
            target_key = "target"
    return source_key, target_key


def loss_source_key(loss_fn: Callable[..., Any] | KeyedLossFn, name: str) -> str:
    """Return the source key for a loss callable.

    Args:
        loss_fn: Loss function or keyed wrapper.
        name: Loss name used when ``loss_fn`` is not a :class:`KeyedLossFn`.

    Returns:
        Source dictionary key for the loss output tensor.
    """
    if isinstance(loss_fn, KeyedLossFn):
        return loss_fn.source_key
    return resolve_loss_keys(name)[0]
