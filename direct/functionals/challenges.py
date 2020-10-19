# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import numpy as np

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
    else:
        return tensor.cpu().numpy()


def fastmri_ssim(gt, target):
    from skimage.metrics import structural_similarity

    gt = _to_numpy(gt)[:, 0, ...]
    target = _to_numpy(target)[:, 0, ...]
    out = structural_similarity(
        gt.transpose(1, 2, 0),
        target.transpose(1, 2, 0),
        multichannel=True,
        data_range=gt.max(),
    )
    return torch.from_numpy(np.array(out)).float()


def fastmri_psnr(gt, target):
    gt = _to_numpy(gt)[:, 0, ...]
    target = _to_numpy(target)[:, 0, ...]
    from skimage.measure import compare_psnr

    """ Compute Peak Signal to Noise Ratio metric (PSNR) """

    out = compare_psnr(gt, target, data_range=gt.max())
    return torch.from_numpy(np.array(out)).float()


def fastmri_nmse(gt, target):
    gt = _to_numpy(gt)[:, 0, ...]
    target = _to_numpy(target)[:, 0, ...]
    out = np.linalg.norm(gt - target) ** 2 / np.linalg.norm(gt) ** 2
    return torch.from_numpy(np.array(out)).float()


def _calgary_campinas_metric(gt, target, metric_func):
    gt = _to_numpy(gt)[:, 0, ...]
    target = _to_numpy(target)[:, 0, ...]

    output = []
    for idx in range(target.shape[0]):
        data_range = np.maximum(gt[idx].max(), target[idx].max()) - np.minimum(
            gt[idx].min(), target[idx].min()
        )
        output.append(metric_func(gt[idx], target[idx], data_range=data_range))

    return torch.from_numpy(np.asarray(output)).mean()


def calgary_campinas_ssim(gt, target):
    from skimage.metrics import structural_similarity

    return _calgary_campinas_metric(gt, target, structural_similarity)


def calgary_campinas_psnr(gt, target):
    from skimage.metrics import peak_signal_noise_ratio

    return _calgary_campinas_metric(gt, target, peak_signal_noise_ratio)


def calgary_campinas_vif(gt, target):
    def vif_func(gt, target, data_range):
        from sewar.full_ref import vifp

        return vifp(gt, target, sigma_nsq=0.4)

    return _calgary_campinas_metric(gt, target, vif_func)
