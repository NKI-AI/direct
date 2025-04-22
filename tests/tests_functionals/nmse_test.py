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
from skimage.metrics import normalized_root_mse as nrmse
from sklearn.datasets import load_sample_image

from direct.functionals.challenges import fastmri_nmse
from direct.functionals.nmse import NMSELoss, NRMSELoss

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
def test_nmse(image):
    image_batch = []
    image_noise_batch = []
    single_image_nmse = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        image_batch.append(image)
        image_noise_batch.append(image_noise)

    image_batch = np.stack(image_batch)
    image_noise_batch = np.stack(image_noise_batch)

    skimage_nrmse = nrmse(
        image_batch.squeeze(),
        image_noise_batch.squeeze(),
    )
    skimage_nmse = skimage_nrmse**2

    image_batch_torch = torch.tensor(image_batch)
    image_noise_batch_torch = torch.tensor(image_noise_batch)
    # Test fastmri_nmse
    fastmri_nmse_batch = fastmri_nmse(image_batch_torch, image_noise_batch_torch)
    assert np.allclose(fastmri_nmse_batch, skimage_nmse, atol=5e-4)
    # Test NMSE loss
    nmse_batch = NMSELoss().forward(image_noise_batch_torch, image_batch_torch)
    assert np.allclose(nmse_batch, skimage_nmse, atol=5e-4)
    # test NRMSE loss
    nrmse_batch = NRMSELoss().forward(image_noise_batch_torch, image_batch_torch)
    assert np.allclose(nrmse_batch**2, skimage_nmse, atol=5e-4)
