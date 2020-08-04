# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np
from numpy import ma as ma


def circular_mask(shape, masking_percentage=0.25):
    radius = masking_percentage * min(shape[1:]) / 2
    center = np.asarray(shape[1:]) // 2

    Y, X = np.ogrid[: shape[1], : shape[2]]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
    return mask


def compute_spike_noise(kspace, masking_percentage=0.8, spike_threshold=100):
    shape = kspace.shape
    mask = circular_mask(shape, masking_percentage)
    kspace_abs = np.abs(kspace)

    # Mask out center of kspace
    kspace_abs[mask] = 0

    curr_spike_threshold = spike_threshold * ma.median(
        ma.array(kspace_abs, mask=kspace_abs == 0)
    )
    spike_mask = kspace_abs > curr_spike_threshold
    spike_num = spike_mask.sum(axis=(-1, -2))
    spike_idx = np.where(spike_num > 0)
    return spike_idx[0].tolist(), spike_num.tolist(), spike_mask
