# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio
from sklearn.datasets import load_sample_image

from direct.functionals.challenges import calgary_campinas_psnr, fastmri_psnr
from direct.functionals.psnr import PSNRLoss

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
def test_psnr(image):
    image_batch = []
    image_noise_batch = []
    single_image_psnr = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        psnr_skimage = peak_signal_noise_ratio(image_true=image, image_test=image_noise, data_range=image_noise.max())

        image_torch = (torch.from_numpy(image).unsqueeze(0)).float()  # 1, C, H, W
        image_noise_torch = (torch.from_numpy(image_noise).unsqueeze(0)).float()  # 1, C, H, W

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        psnr_torch = PSNRLoss(reduction="none").forward(image_noise_torch, image_torch)

        psnr_torch = psnr_torch.numpy().item()
        single_image_psnr.append(psnr_skimage)
        # Assert that single slice psnr matches
        assert np.allclose(psnr_torch, psnr_skimage, atol=5e-4)

    psnr_skimage_batch = np.mean(single_image_psnr)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    psnr_batch = PSNRLoss(reduction="mean").forward(
        image_noise_batch,
        image_batch,
    )
    assert np.allclose(psnr_batch, psnr_skimage_batch, atol=5e-4)


@pytest.mark.parametrize("image", [flower, china])
def test_calgary_campinas_psnr(image):
    image_batch = []
    image_noise_batch = []
    single_image_psnr = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        psnr_skimage = peak_signal_noise_ratio(
            image_true=image,
            image_test=image_noise,
            data_range=np.maximum(image.max(), image_noise.max()) - np.minimum(image.min(), image_noise.min()),
        )

        image_torch = (torch.from_numpy(image).unsqueeze(0)).float()  # 1, C, H, W
        image_noise_torch = (torch.from_numpy(image_noise).unsqueeze(0)).float()  # 1, C, H, W

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        single_image_psnr.append(psnr_skimage)

    psnr_skimage_batch = np.mean(single_image_psnr)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)

    calgary_campinas_psnr_batch = calgary_campinas_psnr(image_batch, image_noise_batch)

    assert np.allclose(calgary_campinas_psnr_batch, psnr_skimage_batch, atol=5e-4)


@pytest.mark.parametrize("image", [flower, china])
def test_fastmri_psnr(image):
    image_batch = []
    image_noise_batch = []
    single_image_psnr = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        image_batch.append(image)
        image_noise_batch.append(image_noise)

    image_batch = np.stack(image_batch)
    image_noise_batch = np.stack(image_noise_batch)

    psnr_skimage_batch = peak_signal_noise_ratio(image_batch, image_noise_batch, data_range=image_batch.max())
    image_batch_torch = torch.tensor(image_batch)
    image_noise_batch_torch = torch.tensor(image_noise_batch)

    fastmri_psnr_batch = fastmri_psnr(image_batch_torch, image_noise_batch_torch)

    assert np.allclose(fastmri_psnr_batch, psnr_skimage_batch, atol=5e-4)
