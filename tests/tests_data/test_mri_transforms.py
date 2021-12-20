# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Tests for the direct.data.mri_transforms module"""

import functools

import numpy as np
import pytest
import torch

from direct.data.mri_transforms import (
    Compose,
    ComputeImage,
    CreateSamplingMask,
    CropAndMask,
    DeleteKeys,
    EstimateBodyCoilImage,
    EstimateSensitivityMap,
    Normalize,
    PadCoilDimension,
    ToTensor,
    WhitenData,
)
from direct.data.transforms import ifft2


def create_sample(shape, **kwargs):
    if any(_ is None for _ in shape):
        shape = tuple(_ if _ else np.random.randint(0, 10) for _ in shape)
    sample = dict()
    sample["kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["filename"] = "filename" + str(np.random.randint(100, 10000))
    sample["slice_no"] = np.random.randint(0, 1000)
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


def _mask_func(shape, seed=None, return_acs=False):
    shape = shape[:-1]
    mask = torch.zeros(shape).bool()
    mask[
        shape[0] // 2 - shape[0] // 4 : shape[0] // 2 + shape[0] // 4,
        shape[1] // 2 - shape[1] // 4 : shape[1] // 2 + shape[1] // 4,
    ] = True
    if return_acs:
        return mask.unsqueeze(0).unsqueeze(-1)
    if seed:
        rng = np.random.RandomState()
        rng.seed(seed)
    mask = mask | torch.from_numpy(np.random.rand(*shape)).round().bool()
    return mask.unsqueeze(0).unsqueeze(-1)


@pytest.mark.parametrize(
    "shape",
    [(4, 7, 6), (3, 10, 8)],
)
def test_Compose(shape):
    sample = create_sample(shape + (2,))

    from torchvision.transforms import CenterCrop, RandomVerticalFlip

    transforms = [CenterCrop([_ // 2 for _ in shape[1:]]), RandomVerticalFlip(0.5)]
    transform = Compose(transforms)

    torch.manual_seed(0)
    compose_out = transform(sample["kspace"])
    kspace = sample["kspace"]

    torch.manual_seed(0)
    for t in transforms:
        kspace = t(kspace)

    assert torch.allclose(compose_out, kspace)


@pytest.mark.parametrize(
    "shape",
    [(1, 4, 6), (5, 7, 6), (2, None, None), (3, 4, 6, 4)],
)
@pytest.mark.parametrize(
    "return_acs",
    [True, False],
)
@pytest.mark.parametrize(
    "padding",
    [None, [2, 2]],
)
@pytest.mark.parametrize(
    "use_shape",
    [True, False],
)
def test_CreateSamplingMask(shape, return_acs, padding, use_shape):

    sample = create_sample(shape + (2,))
    if padding:
        sample.update({"padding_right": padding[0], "padding_left": padding[1]})
    transform = CreateSamplingMask(mask_func=_mask_func, shape=shape[1:] if use_shape else None, return_acs=return_acs)
    if padding and len(shape) > 3:
        with pytest.raises(ValueError):
            sample = transform(sample)
    else:
        sample = transform(sample)
        assert "sampling_mask" in sample
        assert tuple(sample["sampling_mask"].shape) == (1,) + sample["kspace"].shape[1:-1] + (1,)
        if return_acs:
            assert "acs_mask" in sample


@pytest.mark.parametrize(
    "shape, crop",
    [[(3, 10, 16), (5, 6)], [(5, 7, 6), None], [(4, 5, 5), None], [(3, 4, 6, 4), (3, 4, 4)]],
)
def test_CropAndMask(shape, crop):

    sample = create_sample(
        shape=shape + (2,),
        sensitivity_map=torch.rand(shape + (2,)),
        sampling_mask=torch.rand(shape[1:]).round().unsqueeze(0).unsqueeze(-1),
        input_image=torch.rand((1,) + shape[1:] + (2,)),
    )

    transform = CropAndMask(crop, image_space_center_crop=True)

    sample = transform(sample)

    assert "masked_kspace" in sample
    assert "target" in sample

    mask = ~(torch.abs(sample["masked_kspace"]).sum(dim=(0, -1)) == 0)
    assert torch.allclose(mask, sample["sampling_mask"].squeeze().bool())


@pytest.mark.parametrize(
    "shape, spatial_dims",
    [
        [(1, 4, 6), (1, 2)],
        [(5, 7, 6), (1, 2)],
        [(4, 5, 5), (1, 2)],
        [(3, 4, 6, 4), (2, 3)],
    ],
)
@pytest.mark.parametrize(
    "type_recon, complex_output, expect_error",
    [
        ["complex", True, False],
        ["sense", True, False],
        ["rss", False, False],
        ["invalid", None, True],
    ],
)
def test_ComputeImage(shape, spatial_dims, type_recon, complex_output, expect_error):
    sample = create_sample(shape=shape + (2,))
    if expect_error:
        with pytest.raises(ValueError):
            transform = ComputeImage("kspace", "target", ifft2, type_reconstruction=type_recon)
    else:
        transform = ComputeImage("kspace", "target", ifft2, type_reconstruction=type_recon)
        if type_recon == "sense":
            with pytest.raises(ValueError):
                sample = transform(sample, coil_dim=0, spatial_dims=spatial_dims)
            sample.update({"sensitivity_map": torch.rand(shape + (2,))})
        sample = transform(sample, coil_dim=0, spatial_dims=spatial_dims)
        assert "target" in sample
        assert sample["target"].shape == (shape[1:] + (2,) if complex_output else shape[1:])


@pytest.mark.parametrize(
    "shape, spatial_dims",
    [
        [(1, 4, 6), (1, 2)],
        [(5, 7, 6), (1, 2)],
        [(4, 5, 5), (1, 2)],
        [(3, 4, 6, 4), (2, 3)],
    ],
)
@pytest.mark.parametrize("use_seed", [True, False])
def test_EstimateBodyCoilImage(shape, spatial_dims, use_seed):

    sample = create_sample(shape=shape + (2,), sensitivity_map=torch.rand(shape + (2,)))
    transform = EstimateBodyCoilImage(
        mask_func=_mask_func, backward_operator=functools.partial(ifft2, dim=spatial_dims), use_seed=use_seed
    )
    sample = transform(sample)
    assert "body_coil_image" in sample
    assert sample["body_coil_image"].shape == shape[1:]


@pytest.mark.parametrize(
    "shape, spatial_dims",
    [
        [(1, 4, 6), (1, 2)],
        [(5, 7, 6), (1, 2)],
        [(4, 5, 5), (1, 2)],
        [(3, 4, 6, 4), (2, 3)],
    ],
)
@pytest.mark.parametrize(
    "type_of_map, gaussian_sigma, expect_error, sense_map_in_sample",
    [
        ["unit", None, False, False],
        ["rss_estimate", 0.5, False, False],
        ["rss_estimate", None, False, False],
        ["rss_estimate", None, False, True],
        ["invalid", None, True, False],
    ],
)
def test_EstimateSensitivityMap(shape, spatial_dims, type_of_map, gaussian_sigma, expect_error, sense_map_in_sample):
    sample = create_sample(
        shape=shape + (2,),
        acs_mask=torch.rand((1,) + shape[1:] + (1,)).round(),
        sampling_mask=torch.rand((1,) + shape[1:] + (1,)).round(),
    )
    if sense_map_in_sample:
        sample.update({"sensitivity_map": torch.rand(shape + (2,))})

    transform = EstimateSensitivityMap(
        kspace_key="kspace",
        backward_operator=functools.partial(ifft2, dim=spatial_dims),
        type_of_map=type_of_map,
        gaussian_sigma=gaussian_sigma,
    )
    if expect_error:
        with pytest.raises(ValueError):
            sample = transform(sample)
    else:
        if shape[0] == 1 or sense_map_in_sample:
            with pytest.warns(None):
                sample = transform(sample)
        else:
            sample = transform(sample)
        assert "sensitivity_map" in sample
        assert sample["sensitivity_map"].shape == shape + (2,)


@pytest.mark.parametrize(
    "shape",
    [(5, 3, 4)],
)
@pytest.mark.parametrize(
    "delete_keys",
    [["kspace"], ["sensitivity_map", "acs_mask", "sampling_mask"]],
)
def test_DeleteKeys(shape, delete_keys):
    sample = create_sample(
        shape=shape + (2,),
        sensitivity_map=torch.rand(shape + (2,)),
        acs_mask=torch.rand((1,) + shape[1:] + (1,)).round(),
        sampling_mask=torch.rand((1,) + shape[1:] + (1,)).round(),
    )
    transform = DeleteKeys(delete_keys)
    sample = transform(sample)
    for key in delete_keys:
        assert key not in sample


@pytest.mark.parametrize(
    "shape, pad_coils",
    [[(3, 10, 16), 5], [(5, 7, 6), 5], [(4, 5, 5), 2], [(4, 5, 5), None], [(3, 4, 6, 4), 4], [(5, 3, 3, 4), 3]],
)
@pytest.mark.parametrize(
    "key",
    ["kspace", "masked_kspace"],
)
def test_PadCoilDimension(shape, pad_coils, key):
    sample = create_sample(shape=shape + (2,))
    transform = PadCoilDimension(pad_coils=pad_coils, key=key)
    if key not in sample:
        kspace = sample["kspace"]
        sample = transform(sample)
        assert torch.all(sample["kspace"] == kspace)
    else:
        if pad_coils and shape[0] > pad_coils:
            with pytest.raises(ValueError):
                sample = transform(sample)
        else:
            kspace = sample["kspace"]
            sample = transform(sample)
            if pad_coils is None or shape[0] == pad_coils:
                assert torch.all(sample["kspace"] == kspace)
            else:
                assert sample["kspace"].shape == (pad_coils,) + shape[1:] + (2,)


@pytest.mark.parametrize(
    "shape",
    [(3, 4), (5, 3, 4)],
)
@pytest.mark.parametrize(
    "normalize_key",
    [None, "masked_kspace", "kspace"],
)
@pytest.mark.parametrize(
    "percentile",
    [None, 0.9],
)
def test_Normalize(shape, normalize_key, percentile):
    sample = create_sample(
        shape=shape + (2,),
        masked_kspace=torch.rand(shape + (2,)),
        sensitivity_map=torch.rand(shape + (2,)),
        sampling_mask=torch.rand(shape[1:]).round().unsqueeze(0).unsqueeze(-1),
    )
    transform = Normalize(normalize_key, percentile)
    sample = transform(sample)

    assert "scaling_diff" in sample
    assert "scaling_factor" in sample


@pytest.mark.parametrize(
    "shape",
    [(3, 4), (5, 3, 4)],
)
def test_WhitenData(shape):
    sample = create_sample(
        shape=shape + (2,),
        input_image=torch.rand(shape[1:] + (2,)),
    )
    transform = WhitenData(key="input_image")

    sample = transform(sample)


@pytest.mark.parametrize(
    "shape",
    [(3, 4), (5, 3, 4)],
)
@pytest.mark.parametrize(
    "key, is_multicoil, is_complex, is_scalar",
    [
        ["sensitivity_map", True, True, False],
        ["acs_mask", False, False, False],
        ["sampling_mask", False, False, False],
        ["initial_kspace", True, True, False],
        ["initial_image", True, False, False],
        ["target", False, False, False],
        ["scaling_factor", False, False, True],
        ["loglikelihood_scaling", False, False, True],
    ],
)
def test_ToTensor(shape, key, is_multicoil, is_complex, is_scalar):
    sample = create_sample(shape, kspace=np.random.randn(*shape) + 1.0j * np.random.randn(*shape))

    if is_scalar:
        key_shape = (1,)
    else:
        key_shape = shape[1:] if not is_multicoil else shape
    sample[key] = np.random.randn(*key_shape)
    if is_complex:
        sample[key] = sample[key] + 1.0j * np.random.randn(*key_shape)
        key_shape += (2,)

    transform = ToTensor()
    if len(shape) - 1 not in [2, 3]:
        with pytest.raises(ValueError):
            sample = transform(sample)
    else:
        sample = transform(sample)
        assert isinstance(sample["kspace"], torch.Tensor)
        assert sample["kspace"].shape == shape + (2,)
        assert sample[key].shape == key_shape
