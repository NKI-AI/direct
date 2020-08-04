# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import torch
import pathlib

from skimage.metrics import structural_similarity
from PIL import Image

from direct.functionals.ssim import batch_ssim


def _read_image():
    image = Image.open(pathlib.Path("direct/functionals/tests/images/kodim10.png"))
    image = np.array(image).astype(np.float32)
    return image


def test_ssim():
    image = _read_image()
    image_batch = []
    image_noise_batch = []
    single_image_ssim = []
    N_repeat = 5
    for sigma in range(0, 101, 20):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        for _ in range(N_repeat):
            ssim_skimage = structural_similarity(
                image,
                image_noise,
                win_size=11,
                multichannel=True,
                sigma=1.5,
                data_range=255,
                use_sample_covariance=False,
                gaussian_weights=True,
            )

        image_torch = (
            torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
        )  # 1, C, H, W
        image_noise_torch = (
            torch.from_numpy(image_noise).unsqueeze(0).permute(0, 3, 1, 2)
        )

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        for _ in range(N_repeat):
            ssim_torch = batch_ssim(
                image_noise_torch, image_torch, win_size=11, data_range=255
            )

        ssim_torch = ssim_torch.numpy()
        single_image_ssim.append(ssim_torch)
        assert np.allclose(ssim_torch, ssim_skimage, atol=5e-4)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    ssim_batch = batch_ssim(
        image_noise_batch, image_batch, win_size=11, data_range=255, reduction="none"
    )
    assert np.allclose(ssim_batch, single_image_ssim, atol=5e-4)
