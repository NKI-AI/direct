# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio
from sklearn.datasets import load_sample_image

from direct.functionals.psnr import PSNRLoss

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
def test_psnr(image):

    image_batch = []
    image_noise_batch = []
    single_image_psnr = []

    for sigma in range(0, 101, 2):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        psnr_skimage = peak_signal_noise_ratio(image_true=image, image_test=image_noise, data_range=image_noise.max())

        image_torch = (torch.from_numpy(image).unsqueeze(0)).float()  # 1, C, H, W
        image_noise_torch = (torch.from_numpy(image_noise).unsqueeze(0)).float()

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        psnr_torch = PSNRLoss(reduction="none").forward(image_noise_torch, image_torch)

        psnr_torch = psnr_torch.numpy().item()
        single_image_psnr.append(psnr_torch)
        print(psnr_torch, psnr_skimage)
        # Assert that single slice psnr matches
        assert np.allclose(psnr_torch, psnr_skimage, atol=5e-4)

    psnr_skimage_batch = np.mean(single_image_psnr)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    psnr_batch = PSNRLoss(reduction="mean").forward(
        image_noise_batch,
        image_batch,
    )
    # Assert that batch psnr matches
    assert np.allclose(psnr_batch, psnr_skimage_batch, atol=5e-4)
