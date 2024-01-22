# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio
from sklearn.datasets import load_sample_image

from direct.functionals.snr import SNRLoss

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_snr(image, reduction):
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    image_noise_batch = []
    single_image_snr = []

    for sigma in range(0, 101, 20):
        noise = sigma * torch.randn(*image.shape)
        image_noise = image + noise
        snr_torch = SNRLoss(reduction=reduction).forward(image_noise, image)
        image_noise_batch.append(image_noise)
        single_image_snr.append(snr_torch)

    image_batch = torch.cat([image] * len(image_noise_batch), dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    snr_batch = SNRLoss(reduction=reduction).forward(image_noise_batch, image_batch)
    # Assert that batch snr matches single snrs
    assert np.allclose(snr_batch, np.average(single_image_snr), atol=5e-4)
