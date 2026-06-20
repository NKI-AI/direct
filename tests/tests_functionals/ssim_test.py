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
"""Tests for the direct.functionals.ssim module."""

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from sklearn.datasets import load_sample_image

from direct.functionals.challenges import calgary_campinas_ssim, fastmri_ssim
from direct.functionals.ssim import SSIM3DLoss, SSIMLoss

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
@pytest.mark.parametrize("data_range_255", [True, False])
@pytest.mark.parametrize("win_size", [7, 11])
@pytest.mark.parametrize("k1, k2", [[0.01, 0.03], [0.05, 0.1]])
def test_ssim(image, data_range_255, win_size, k1, k2):
    image_batch = []
    image_noise_batch = []
    single_image_ssim = []

    for sigma in range(0, 101, 20):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        ssim_skimage = structural_similarity(
            im1=image,
            im2=image_noise,
            win_size=win_size,
            channel_axis=0,
            data_range=255 if data_range_255 else image.max(),
            K1=k1,
            K2=k2,
        )

        image_torch = (torch.from_numpy(image).unsqueeze(0)).float()  # 1, C, H, W
        image_noise_torch = (torch.from_numpy(image_noise).unsqueeze(0)).float()

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        ssim_torch = 1 - SSIMLoss(win_size=win_size, k1=k1, k2=k2).forward(
            image_noise_torch, image_torch, data_range=torch.tensor([255 if data_range_255 else image_torch.max()])
        )

        ssim_torch = ssim_torch.numpy()
        single_image_ssim.append(ssim_torch)
        # Assert that single slice ssim matches
        assert np.allclose(ssim_torch, ssim_skimage, atol=5e-4)

    ssim_skimage_batch = np.mean(single_image_ssim)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    ssim_batch = 1 - SSIMLoss(win_size=win_size, k1=k1, k2=k2).forward(
        input_data=image_noise_batch,
        target_data=image_batch,
        data_range=torch.tensor([255]) if data_range_255 else image_batch.amax((1, 2, 3)),
    )
    # Assert that batch ssim matches
    assert np.allclose(ssim_batch, ssim_skimage_batch, atol=5e-4)


@pytest.mark.parametrize("image", [flower, china])
def test_calgary_campinas_ssim(image):
    image_batch = []
    image_noise_batch = []
    single_image_ssim = []

    for sigma in range(0, 101, 20):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        ssim_skimage = structural_similarity(
            im1=image,
            im2=image_noise,
            channel_axis=0,
            data_range=np.maximum(image.max(), image_noise.max()) - np.minimum(image.min(), image_noise.min()),
        )

        image_torch = (torch.from_numpy(image).unsqueeze(0)).float()  # 1, C, H, W
        image_noise_torch = (torch.from_numpy(image_noise).unsqueeze(0)).float()  # 1, C, H, W

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        single_image_ssim.append(ssim_skimage)
    ssim_skimage_batch = np.mean(single_image_ssim)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)

    calgary_campinas_ssim_batch = calgary_campinas_ssim(image_batch, image_noise_batch)

    assert np.allclose(calgary_campinas_ssim_batch, ssim_skimage_batch, atol=5e-4)


@pytest.mark.parametrize("image", [flower, china])
def test_fastmri_ssim(image):
    image_batch = []
    image_noise_batch = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        image_batch.append(image)
        image_noise_batch.append(image_noise)

    image_batch = np.stack(image_batch)
    image_noise_batch = np.stack(image_noise_batch)

    ssim_skimage_batch = structural_similarity(
        image_batch.squeeze(),
        image_noise_batch.squeeze(),
        channel_axis=0,
        data_range=image_batch.max(),
    )
    image_batch_torch = torch.tensor(image_batch)
    image_noise_batch_torch = torch.tensor(image_noise_batch)

    fastmri_ssim_batch = fastmri_ssim(image_batch_torch, image_noise_batch_torch)

    assert np.allclose(fastmri_ssim_batch, ssim_skimage_batch, atol=5e-4)


@pytest.mark.parametrize("data_range_255", [True, False])
@pytest.mark.parametrize("win_size", [7])
@pytest.mark.parametrize("k1, k2", [[0.01, 0.03], [0.05, 0.1]])
def test_ssim_3de(data_range_255, win_size, k1, k2):
    image = torch.from_numpy(np.concatenate([flower, china] * 4, 0)).unsqueeze(0).unsqueeze(0)
    image_noise_batch = []

    single_image_ssim = []

    for sigma in range(0, 101, 20):
        noise = sigma * torch.randn(*image.shape)
        image_noise = image + noise
        ssim_torch = 1 - SSIM3DLoss(win_size=win_size, k1=k1, k2=k2).forward(
            image_noise, image, data_range=torch.tensor([255 if data_range_255 else image.max()])
        )
        image_noise_batch.append(image_noise)
        single_image_ssim.append(ssim_torch)

    image_batch = torch.cat([image] * len(image_noise_batch), dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    ssim_batch = 1 - SSIM3DLoss(win_size=win_size, k1=k1, k2=k2).forward(
        input_data=image_noise_batch,
        target_data=image_batch,
        data_range=torch.tensor([255]) if data_range_255 else image_batch.amax((1, 2, 3, 4)),
    )
    # Assert that batch ssim matches single ssims
    assert np.allclose(ssim_batch, np.average(single_image_ssim), atol=5e-4)
