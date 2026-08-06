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

import torch

from direct.config.defaults import FunctionConfig
from direct.nn.loss_keys import KeyedLossFn, resolve_loss_keys
from direct.nn.mri_models import MRIModelEngine


def test_resolve_loss_keys_defaults():
    assert resolve_loss_keys("l1_loss") == ("output_image", "target")
    assert resolve_loss_keys("ssim_loss") == ("output_image", "target")
    assert resolve_loss_keys("kspace_l1_loss") == ("output_kspace", "kspace")
    assert resolve_loss_keys("displacement_field_smooth_loss_l1") == (
        "displacement_field",
        "displacement_field",
    )


def test_resolve_loss_keys_explicit_override():
    assert resolve_loss_keys("l1_loss", source_key="registered_image", target_key="reference_image") == (
        "registered_image",
        "reference_image",
    )


def test_function_config_keys_optional():
    cfg = FunctionConfig(function="l1_loss")
    assert cfg.source_key is None
    assert cfg.target_key is None
    cfg2 = FunctionConfig(function="l1_loss", source_key="registered_image", target_key="reference_image")
    assert cfg2.source_key == "registered_image"
    assert cfg2.target_key == "reference_image"


def test_compute_loss_on_data_uses_outputs_keys():
    """Key-based path: only losses whose source_key is present in outputs are applied."""

    class _DummyEngine(MRIModelEngine):
        def __init__(self):
            # Bypass Engine.__init__; only need compute_loss_on_data.
            pass

        def forward_function(self, data):
            raise NotImplementedError

        def build_loss(self):
            raise NotImplementedError

    engine = _DummyEngine()
    image = torch.ones(1, 8, 8)
    kspace = torch.ones(1, 1, 8, 8, 2)
    target = torch.zeros(1, 8, 8)
    data = {
        "target": target,
        "kspace": torch.zeros_like(kspace),
        "reconstruction_size": None,
    }

    def _l1(source, target, reduction="mean", reconstruction_size=None):
        return (source - target).abs().mean()

    loss_fns = {
        "l1_loss": KeyedLossFn(_l1, "output_image", "target"),
        "kspace_l1_loss": KeyedLossFn(_l1, "output_kspace", "kspace"),
    }
    loss_dict = {k: torch.tensor([0.0]) for k in loss_fns}

    # Only image outputs → only image loss accumulates.
    out = engine.compute_loss_on_data(
        {k: v.clone() for k, v in loss_dict.items()},
        loss_fns,
        data,
        outputs={"output_image": image},
    )
    assert out["l1_loss"].item() > 0
    assert out["kspace_l1_loss"].item() == 0

    # Only k-space outputs → only k-space loss accumulates.
    out = engine.compute_loss_on_data(
        {k: v.clone() for k, v in loss_dict.items()},
        loss_fns,
        data,
        outputs={"output_kspace": kspace},
    )
    assert out["l1_loss"].item() == 0
    assert out["kspace_l1_loss"].item() > 0

    # Legacy kwargs still work.
    out = engine.compute_loss_on_data(
        {k: v.clone() for k, v in loss_dict.items()},
        loss_fns,
        data,
        output_image=image,
        output_kspace=kspace,
    )
    assert out["l1_loss"].item() > 0
    assert out["kspace_l1_loss"].item() > 0
