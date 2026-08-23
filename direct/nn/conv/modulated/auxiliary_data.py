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
"""Utilities for building auxiliary conditioning vectors for modulated convolutions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from direct.nn.adain.adain import NormType
from direct.nn.conv.modulated.modulated_conv import ModConvType

__all__ = [
    "AUXILIARY_FEATURE_REGISTRY",
    "DEFAULT_AUXILIARY_FEATURE_NAMES",
    "AuxiliaryFeature",
    "ModulationConfig",
    "prepare_auxiliary_data",
    "register_auxiliary_feature",
    "resolve_auxiliary_features",
]


@dataclass(frozen=True)
class AuxiliaryFeature:
    """Metadata for one auxiliary conditioning channel.

    Args:
        key: Key in the batch dictionary.
        log_scale: Multiplier applied to the feature before ``log`` when ``log_aux`` is enabled.
    """

    key: str
    log_scale: float = 1.0


AUXILIARY_FEATURE_REGISTRY: dict[str, AuxiliaryFeature] = {
    "acceleration": AuxiliaryFeature("acceleration"),
    "center_fraction": AuxiliaryFeature("center_fraction", log_scale=100.0),
    "field_strength": AuxiliaryFeature("field_strength"),
}

DEFAULT_AUXILIARY_FEATURE_NAMES: tuple[str, ...] = tuple(AUXILIARY_FEATURE_REGISTRY.keys())


class ModulationConfig(Protocol):
    """Minimal model configuration required to build auxiliary data."""

    conv_modulation: ModConvType
    aux_in_features: int
    log_aux: bool
    auxiliary_features: tuple[str, ...] | None


def register_auxiliary_feature(feature: AuxiliaryFeature) -> None:
    """Register a custom auxiliary feature for use in ``auxiliary_features`` configs.

    Args:
        feature: Feature.

    Returns:
        ``None``.
    """
    AUXILIARY_FEATURE_REGISTRY[feature.key] = feature


def resolve_auxiliary_features(
    feature_names: Sequence[str] | None,
    aux_in_features: int,
) -> tuple[AuxiliaryFeature, ...]:
    """Resolve auxiliary feature names from config into feature metadata.

    Args:
        feature_names: Explicit ordered list of feature keys. When ``None``, the first ``aux_in_features`` entries from
            :data:`DEFAULT_AUXILIARY_FEATURE_NAMES` are used.
        aux_in_features: Expected number of auxiliary channels. Must match the resolved list length.

    Returns:
        Resolved feature metadata in request order.

    Raises:
        If feature names are unknown, or the list length does not match ``aux_in_features``.
    """
    if aux_in_features <= 0:
        raise ValueError(f"aux_in_features must be positive, got {aux_in_features}.")

    if feature_names is None:
        if aux_in_features > len(DEFAULT_AUXILIARY_FEATURE_NAMES):
            raise ValueError(
                f"aux_in_features={aux_in_features} exceeds the number of default auxiliary "
                f"features ({len(DEFAULT_AUXILIARY_FEATURE_NAMES)}): "
                f"{list(DEFAULT_AUXILIARY_FEATURE_NAMES)}. "
                f"Set auxiliary_features explicitly in the model config."
            )
        names = DEFAULT_AUXILIARY_FEATURE_NAMES[:aux_in_features]
    else:
        names = tuple(feature_names)
        if len(names) != aux_in_features:
            raise ValueError(
                f"auxiliary_features has length {len(names)} ({list(names)}) but aux_in_features={aux_in_features}."
            )

    unknown = sorted(set(names) - AUXILIARY_FEATURE_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown auxiliary feature(s): {unknown}. Known features: {sorted(AUXILIARY_FEATURE_REGISTRY)}."
        )

    return tuple(AUXILIARY_FEATURE_REGISTRY[name] for name in names)


def _needs_auxiliary_data(cfg: Any) -> bool:
    """Return whether the model config requires auxiliary conditioning vectors.

    Args:
        cfg: Cfg.

    Returns:
        The result.
    """
    if not hasattr(cfg, "conv_modulation"):
        return False

    conv_modulation = cfg.conv_modulation
    if isinstance(conv_modulation, str):
        conv_modulation = ModConvType.from_str(conv_modulation) or ModConvType.NONE

    if conv_modulation != ModConvType.NONE:
        return True

    for attr in ("image_unet_norm_type", "unet_norm_type"):
        norm_type = getattr(cfg, attr, None)
        if norm_type is None:
            continue
        if isinstance(norm_type, str) and norm_type.upper() == "ADAIN":
            return True
        if getattr(norm_type, "name", "").upper() == "ADAIN":
            return True
        if norm_type == NormType.ADAIN:
            return True

    return False


def prepare_auxiliary_data(
    data: Mapping[str, Any],
    cfg: ModulationConfig | None,
    *,
    features: Sequence[AuxiliaryFeature] | None = None,
) -> torch.Tensor | None:
    """Build an auxiliary conditioning vector for modulated models.

    Args:
        data: Batch dictionary containing the auxiliary feature tensors.
        cfg: Model configuration with modulation settings. Uses ``cfg.auxiliary_features`` when ``features`` is not
            provided.
        features: Explicit feature list, mainly for testing. Overrides ``cfg.auxiliary_features``.

    Returns:
        Tensor of shape ``(batch_size, aux_in_features)``, or ``None`` when modulation is disabled.

    Raises:
        If auxiliary configuration is invalid or a feature tensor has an unexpected shape.
        If a required auxiliary feature is missing from ``data``.
    """
    if cfg is None or not _needs_auxiliary_data(cfg):
        return None

    log_aux = getattr(cfg, "log_aux", False)
    auxiliary_features = getattr(cfg, "auxiliary_features", None)
    aux_in_features = getattr(cfg, "aux_in_features", None)
    if aux_in_features is None:
        return None

    selected_features = (
        tuple(features) if features is not None else resolve_auxiliary_features(auxiliary_features, aux_in_features)
    )

    components = [_prepare_feature(data, feature, log_aux=log_aux) for feature in selected_features]
    auxiliary_data = torch.cat(components, dim=1)

    if log_aux:
        auxiliary_data = auxiliary_data.log()

    return auxiliary_data


def _prepare_feature(data: Mapping[str, Any], feature: AuxiliaryFeature, *, log_aux: bool) -> torch.Tensor:
    """Prepare feature.

    Args:
        data: Data.
        feature: Feature.
        log_aux: Log aux.

    Returns:
        The result.

    Raises:
        KeyError: If the operation cannot be completed.
    """
    if feature.key not in data:
        raise KeyError(
            f"Missing auxiliary feature '{feature.key}' required for modulation. Available keys: {sorted(data.keys())}."
        )

    tensor = _to_aux_column(data[feature.key])
    if log_aux and feature.log_scale != 1.0:
        tensor = tensor * feature.log_scale
    return tensor


def _to_aux_column(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a scalar auxiliary value to shape ``(batch_size, 1)``.

    Args:
        tensor: Tensor.

    Returns:
        The result.
    """
    tensor = tensor.float()

    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 2 and tensor.shape[-1] == 1:
        return tensor

    raise ValueError(
        f"Expected auxiliary feature with shape (batch_size,) or (batch_size, 1), got {tuple(tensor.shape)}."
    )
