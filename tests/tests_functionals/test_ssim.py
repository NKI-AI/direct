# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from sklearn.datasets import load_sample_image

from direct.functionals.ssim import SSIMLoss

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
        X=image_noise_batch,
        Y=image_batch,
        data_range=torch.tensor([255]) if data_range_255 else image_batch.amax((1, 2, 3)),
    )
    # Assert that batch ssim matches
    assert np.allclose(ssim_batch, ssim_skimage_batch, atol=5e-4)
