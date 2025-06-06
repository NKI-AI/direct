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
import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from sklearn.datasets import load_sample_image

from direct.functionals import HFENL1Loss, HFENL2Loss, hfen_l1, hfen_l2

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("kernel_size", [10, 15])
@pytest.mark.parametrize("norm", [True, False])
def test_hfen_l1(image, reduction, kernel_size, norm):
    image = torch.from_numpy(image).unsqueeze(0)

    noise = 0.5 * torch.randn(*image.shape)
    image_noise = image + noise
    hfenl1loss = HFENL1Loss(reduction=reduction, kernel_size=kernel_size, norm=norm).forward(image_noise, image)
    hfenl1metric = hfen_l1(image_noise, image, reduction=reduction, kernel_size=kernel_size, norm=norm)
    assert hfenl1loss == hfenl1metric


@pytest.mark.parametrize("image", [flower, china])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("kernel_size", [10, 15])
@pytest.mark.parametrize("norm", [True, False])
def test_hfen_l2(image, reduction, kernel_size, norm):
    image = torch.from_numpy(image).unsqueeze(0)

    noise = 0.5 * torch.randn(*image.shape)
    image_noise = image + noise
    hfenl2loss = HFENL2Loss(reduction=reduction, kernel_size=kernel_size, norm=norm).forward(image_noise, image)
    hfenl2metric = hfen_l2(image_noise, image, reduction=reduction, kernel_size=kernel_size, norm=norm)
    assert hfenl2loss == hfenl2metric
