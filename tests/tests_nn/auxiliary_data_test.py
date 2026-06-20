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
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from direct.nn.conv.modulated import (
    AuxiliaryFeature,
    ModConvType,
    prepare_auxiliary_data,
    register_auxiliary_feature,
    resolve_auxiliary_features,
)


@dataclass
class _Cfg:
    conv_modulation: ModConvType
    aux_in_features: int
    log_aux: bool
    auxiliary_features: Optional[tuple[str, ...]] = None


def _batch_data() -> dict[str, torch.Tensor]:
    return {
        "acceleration": torch.tensor([4.0, 8.0]),
        "center_fraction": torch.tensor([0.08, 0.04]),
        "field_strength": torch.tensor([3.0, 1.5]),
    }


def test_prepare_auxiliary_data_returns_none_without_modulation_config():
    cfg = object()
    assert prepare_auxiliary_data(_batch_data(), cfg) is None  # type: ignore[arg-type]


def test_prepare_auxiliary_data_returns_none_without_modulation():
    cfg = _Cfg(conv_modulation=ModConvType.NONE, aux_in_features=2, log_aux=False)
    assert prepare_auxiliary_data(_batch_data(), cfg) is None


def test_prepare_auxiliary_data_for_adain_without_modulation():
    @dataclass
    class _AdaINCfg:
        conv_modulation: ModConvType = ModConvType.NONE
        aux_in_features: int = 2
        log_aux: bool = True
        auxiliary_features: Optional[tuple[str, ...]] = None
        image_unet_norm_type: str = "ADAIN"

    auxiliary_data = prepare_auxiliary_data(_batch_data(), _AdaINCfg())
    assert auxiliary_data is not None
    assert auxiliary_data.shape == (2, 2)


def test_prepare_auxiliary_data_two_features_without_log():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=2, log_aux=False)
    auxiliary_data = prepare_auxiliary_data(_batch_data(), cfg)

    assert auxiliary_data is not None
    assert auxiliary_data.shape == (2, 2)
    torch.testing.assert_close(
        auxiliary_data,
        torch.tensor([[4.0, 0.08], [8.0, 0.04]]),
    )


def test_prepare_auxiliary_data_two_features_with_log():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=2, log_aux=True)
    auxiliary_data = prepare_auxiliary_data(_batch_data(), cfg)

    assert auxiliary_data is not None
    torch.testing.assert_close(
        auxiliary_data,
        torch.log(torch.tensor([[4.0, 8.0], [8.0, 4.0]])),
    )


def test_prepare_auxiliary_data_three_features():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=3, log_aux=False)
    auxiliary_data = prepare_auxiliary_data(_batch_data(), cfg)

    assert auxiliary_data is not None
    assert auxiliary_data.shape == (2, 3)
    torch.testing.assert_close(
        auxiliary_data,
        torch.tensor([[4.0, 0.08, 3.0], [8.0, 0.04, 1.5]]),
    )


def test_prepare_auxiliary_data_explicit_feature_list():
    cfg = _Cfg(
        conv_modulation=ModConvType.FEATURES,
        aux_in_features=2,
        log_aux=False,
        auxiliary_features=("field_strength", "acceleration"),
    )
    auxiliary_data = prepare_auxiliary_data(_batch_data(), cfg)

    assert auxiliary_data is not None
    torch.testing.assert_close(
        auxiliary_data,
        torch.tensor([[3.0, 4.0], [1.5, 8.0]]),
    )


def test_prepare_auxiliary_data_missing_feature_raises():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=2, log_aux=False)
    data = {"acceleration": torch.tensor([4.0])}

    with pytest.raises(KeyError, match="center_fraction"):
        prepare_auxiliary_data(data, cfg)


def test_prepare_auxiliary_data_invalid_aux_in_features_raises():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=4, log_aux=False)

    with pytest.raises(ValueError, match="exceeds the number of default"):
        prepare_auxiliary_data(_batch_data(), cfg)


def test_resolve_auxiliary_features_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown auxiliary feature"):
        resolve_auxiliary_features(("acceleration", "unknown_feature"), aux_in_features=2)


def test_resolve_auxiliary_features_length_mismatch_raises():
    with pytest.raises(ValueError, match="auxiliary_features has length"):
        resolve_auxiliary_features(("acceleration",), aux_in_features=2)


def test_resolve_auxiliary_features_non_positive_raises():
    with pytest.raises(ValueError, match="aux_in_features must be positive"):
        resolve_auxiliary_features(None, aux_in_features=0)


def test_register_auxiliary_feature():
    feature = AuxiliaryFeature("custom_feature")
    register_auxiliary_feature(feature)
    resolved = resolve_auxiliary_features(("custom_feature",), aux_in_features=1)
    assert resolved[0].key == "custom_feature"


def test_prepare_auxiliary_data_string_modulation():
    cfg = _Cfg(conv_modulation="features", aux_in_features=2, log_aux=False)  # type: ignore[arg-type]
    auxiliary_data = prepare_auxiliary_data(_batch_data(), cfg)
    assert auxiliary_data is not None
    assert auxiliary_data.shape == (2, 2)


def test_prepare_auxiliary_data_missing_aux_in_features():
    class _CfgNoAux:
        conv_modulation = ModConvType.FEATURES
        aux_in_features = None
        log_aux = False
        auxiliary_features = None

    assert prepare_auxiliary_data(_batch_data(), _CfgNoAux()) is None


def test_prepare_auxiliary_data_scalar_feature():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=1, log_aux=False)
    data = {"acceleration": torch.tensor(4.0)}
    auxiliary_data = prepare_auxiliary_data(data, cfg)
    assert auxiliary_data is not None
    assert auxiliary_data.shape == (1, 1)


def test_prepare_auxiliary_data_column_feature():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=1, log_aux=False)
    data = {"acceleration": torch.tensor([[4.0], [8.0]])}
    auxiliary_data = prepare_auxiliary_data(data, cfg)
    assert auxiliary_data is not None
    torch.testing.assert_close(auxiliary_data, torch.tensor([[4.0], [8.0]]))


def test_prepare_auxiliary_data_invalid_feature_shape_raises():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=1, log_aux=False)
    data = {"acceleration": torch.randn(2, 4, 4)}
    with pytest.raises(ValueError, match="Expected auxiliary feature"):
        prepare_auxiliary_data(data, cfg)


def test_prepare_auxiliary_data_explicit_features_override():
    cfg = _Cfg(conv_modulation=ModConvType.FEATURES, aux_in_features=1, log_aux=False)
    features = (AuxiliaryFeature("acceleration"),)
    auxiliary_data = prepare_auxiliary_data(_batch_data(), cfg, features=features)
    assert auxiliary_data is not None
    torch.testing.assert_close(auxiliary_data, torch.tensor([[4.0], [8.0]]))
