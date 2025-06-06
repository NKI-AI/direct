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
import pytest
import torch

from direct.data.transforms import fft2, ifft2
from direct.nn.xpdnet.xpdnet import XPDNet


def create_input(shape):
    data = torch.rand(shape).float()

    return data


@pytest.mark.parametrize(
    "shape",
    [
        [3, 3, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "num_iter",
    [2, 3],
)
@pytest.mark.parametrize(
    "num_primal",
    [2, 3],
)
@pytest.mark.parametrize(
    "image_model_architecture",
    ["MWCNN", None],
)
@pytest.mark.parametrize(
    "primal_only, kspace_model_architecture, num_dual",
    [
        [True, None, 1],
        [False, "CONV", 3],
        [False, "DIDN", 2],
        [False, None, 2],
    ],
)
@pytest.mark.parametrize(
    "normalize",
    [True, False],
)
def test_xpdnet(
    shape,
    num_iter,
    num_primal,
    num_dual,
    image_model_architecture,
    kspace_model_architecture,
    primal_only,
    normalize,
):
    kwargs = {
        "forward_operator": fft2,
        "backward_operator": ifft2,
        "num_iter": num_iter,
        "num_primal": num_primal,
        "num_dual": num_dual,
        "image_model_architecture": image_model_architecture,
        "kspace_model_architecture": kspace_model_architecture,
        "use_primal_only": primal_only,
        "normalize": normalize,
    }

    kspace = create_input(shape + [2]).cpu()
    sens = create_input(shape + [2]).cpu()
    mask = create_input([shape[0]] + [1] + shape[2:] + [1]).round().int().cpu()

    if (not image_model_architecture == "MWCNN") or (not primal_only and not kspace_model_architecture):
        with pytest.raises(NotImplementedError):
            model = XPDNet(**kwargs)
            out = model(kspace, mask, sens)
    else:
        model = XPDNet(**kwargs)
        out = model(kspace, mask, sens)
        assert list(out.shape) == [shape[0]] + shape[2:] + [2]
