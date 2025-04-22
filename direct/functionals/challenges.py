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
"""Direct metrics for the FastMRI and Calgary-Campinas challenges."""

import numpy as np
import skimage.metrics
import torch

__all__ = (
    "fastmri_ssim",
    "fastmri_psnr",
    "fastmri_nmse",
    "calgary_campinas_ssim",
    "calgary_campinas_psnr",
    "calgary_campinas_vif",
)


def _to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.cpu().numpy()


def fastmri_ssim(gt, target):
    """Compute Structural Similarity Index Measure (SSIM) compatible with the FastMRI challenge."""

    gt = _to_numpy(gt)[:, 0, ...]
    target = _to_numpy(target)[:, 0, ...]
    out = skimage.metrics.structural_similarity(
        gt.transpose(1, 2, 0),
        target.transpose(1, 2, 0),
        channel_axis=-1,
        data_range=gt.max(),
    )
    return torch.from_numpy(np.array(out)).float()


def fastmri_psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR) compatible with the FastMRI challenge."""
    gt = _to_numpy(gt)[:, 0, ...]
    pred = _to_numpy(pred)[:, 0, ...]

    out = skimage.metrics.peak_signal_noise_ratio(image_true=gt, image_test=pred, data_range=gt.max())
    return torch.from_numpy(np.array(out)).float()


def fastmri_nmse(gt, pred):
    """Compute Normalized Mean Square Error metric (NMSE) compatible with the FastMRI challenge."""
    gt = _to_numpy(gt)[:, 0, ...]
    pred = _to_numpy(pred)[:, 0, ...]
    out = np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2
    return torch.from_numpy(np.array(out)).float()


def _calgary_campinas_metric(gt, pred, metric_func):
    """General placeholder for the Calgary-Campinas challenge metrics."""
    # https://github.com/rmsouza01/MC-MRRec-challenge/blob/master/JNotebooks/evaluation-system/extract_challenge_metrics_pre_submisison.ipynb
    gt = _to_numpy(gt)[:, 0, ...]
    pred = _to_numpy(pred)[:, 0, ...]
    gt_max = gt.max(axis=(1, 2), keepdims=True)
    gt = gt / gt_max
    pred = pred / gt_max

    output = []
    for idx in range(gt.shape[0]):
        data_range = np.maximum(gt[idx].max(), pred[idx].max()) - np.minimum(gt[idx].min(), pred[idx].min())
        output.append(metric_func(gt[idx], pred[idx], data_range=data_range))

    return torch.from_numpy(np.asarray(output)).mean()


def calgary_campinas_ssim(gt, pred):
    return _calgary_campinas_metric(gt, pred, skimage.metrics.structural_similarity)


def calgary_campinas_psnr(gt, pred):
    return _calgary_campinas_metric(gt, pred, skimage.metrics.peak_signal_noise_ratio)


def calgary_campinas_vif(gt, pred):
    def vif_func(gt, target, data_range):  # noqa
        from direct.utils.imports import _module_available

        # Calgary Campinas VIF metric requires 'sewar' module. Check if it exists
        if not _module_available("sewar"):
            raise RuntimeError(
                "'sewar' module required for calgary_campinas_vif metric, but not found. "
                "Please use 'pip3 install sewar' and run again."
            )
        else:
            from sewar.full_ref import vifp

            return vifp(gt, target, sigma_nsq=0.4)

    return _calgary_campinas_metric(gt, pred, vif_func)
