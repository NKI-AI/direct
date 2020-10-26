# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np


def volume_post_processing_func(volume):
    """Processing function for Calgary-Campinas, the challenge uses orthogonally normalized iFFT/FFT."""
    # Only needed to fix a bug in Calgary Campinas training
    volume = volume / np.sqrt(np.prod(volume.shape[1:]))
    return volume
