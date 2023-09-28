# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Tests for the direct.ssl.ssl module."""

import numpy as np
import pytest
import torch

from direct.ssl.ssl import *


def create_sample(shape, **kwargs):
    sample = dict()
    sample["kspace"] = torch.rand(*shape).float()
    sample["filename"] = ["filename" + str(_) for _ in np.random.randint(100, 10000, size=shape[0])]
    sample["slice_no"] = [_ for _ in np.random.randint(0, 1000, size=shape[0])]

    sample["sampling_mask"] = torch.rand(shape[0], 1, *shape[2:-1], 1).round().bool()
    sample["sampling_mask"][
        :, :, shape[2] // 2 - 16 : shape[2] // 2 + 16, shape[3] // 2 - 16 : shape[3] // 2 + 16
    ] = True

    sample["acs_mask"] = torch.zeros(shape[0], 1, *shape[2:-1], 1).bool()
    sample["acs_mask"][:, :, shape[2] // 2 - 16 : shape[2] // 2 + 16, shape[3] // 2 - 16 : shape[3] // 2 + 16] = True

    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


@pytest.mark.parametrize(
    "shape",
    [(1, 3, 40, 60)],
)
@pytest.mark.parametrize(
    "ratio",
    [0.2, 0.5],
)
@pytest.mark.parametrize(
    "acs_region",
    [[4, 4], [0, 0], [14, 13]],
)
@pytest.mark.parametrize(
    "keep_acs",
    [True, False],
)
@pytest.mark.parametrize(
    "use_seed",
    [True, False],
)
@pytest.mark.parametrize(
    "std_scale",
    [2.0, 4.0],
)
def test_gaussian_mask_splitter(shape, ratio, acs_region, keep_acs, use_seed, std_scale):
    sample = create_sample(shape + (2,))
    splitter = GaussianMaskSplitterModule(
        ratio=ratio,
        acs_region=acs_region,
        keep_acs=keep_acs,
        use_seed=use_seed,
        kspace_key="kspace",
        std_scale=std_scale,
    )
    sample = splitter(sample)
    if not keep_acs:
        assert torch.allclose(
            sample["theta_sampling_mask"] & sample["lambda_sampling_mask"], torch.zeros_like(sample["sampling_mask"])
        )
    assert torch.allclose(sample["theta_sampling_mask"] | sample["lambda_sampling_mask"], sample["sampling_mask"])
    assert torch.allclose(
        torch.where(
            sample["theta_sampling_mask"], sample["theta_kspace"], sample["lambda_sampling_mask"] * sample["kspace"]
        ),
        (sample["theta_sampling_mask"] | sample["lambda_sampling_mask"]) * sample["kspace"],
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 3, 40, 60)],
)
@pytest.mark.parametrize(
    "ratio",
    [0.2, 0.5],
)
@pytest.mark.parametrize(
    "acs_region",
    [[4, 4], [0, 0], [14, 13]],
)
@pytest.mark.parametrize(
    "keep_acs",
    [True, False],
)
@pytest.mark.parametrize(
    "use_seed",
    [True, False],
)
def test_uniform_mask_splitter(shape, ratio, acs_region, keep_acs, use_seed):
    sample = create_sample(shape + (2,))
    splitter = UniformMaskSplitterModule(
        ratio=ratio, acs_region=acs_region, keep_acs=keep_acs, use_seed=use_seed, kspace_key="kspace"
    )
    sample = splitter(sample)
    if not keep_acs:
        assert torch.allclose(
            sample["theta_sampling_mask"] & sample["lambda_sampling_mask"], torch.zeros_like(sample["sampling_mask"])
        )
    assert torch.allclose(sample["theta_sampling_mask"] | sample["lambda_sampling_mask"], sample["sampling_mask"])
    assert torch.allclose(
        torch.where(
            sample["theta_sampling_mask"], sample["theta_kspace"], sample["lambda_sampling_mask"] * sample["kspace"]
        ),
        (sample["theta_sampling_mask"] | sample["lambda_sampling_mask"]) * sample["kspace"],
    )
