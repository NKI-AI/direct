# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Tests for the direct.data.mri_transforms module."""

import functools

import numpy as np
import pytest
import torch

from direct.data.mri_transforms import (
    ApplyMask,
    ApplyZeroPadding,
    Compose,
    ComputeImage,
    ComputeScalingFactor,
    ComputeZeroPadding,
    CreateSamplingMask,
    CropKspace,
    DeleteKeys,
    EstimateBodyCoilImage,
    EstimateSensitivityMap,
    Normalize,
    PadCoilDimension,
    RandomFlip,
    RandomFlipType,
    RandomRotation,
    ReconstructionType,
    SensitivityMapType,
    ToTensor,
    WhitenData,
    build_mri_transforms,
)
from direct.data.transforms import fft2, ifft2, modulus
from direct.exceptions import ItemNotFoundException


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
    assert all(repr(t) in repr(transform) for t in transforms)
    torch.manual_seed(0)
    compose_out = transform(sample["kspace"])
    kspace = sample["kspace"]

    torch.manual_seed(0)
    for t in transforms:
        kspace = t(kspace)
    assert torch.allclose(compose_out, kspace)


@pytest.mark.parametrize(
    "shape",
    [(5, 7, 6), (3, 4, 6, 4)],
)
def test_ComputeZeroPadding(shape):

    sample = create_sample(shape + (2,))

    pad_shape = [1 for _ in range(len(sample["kspace"].shape))]
    pad_shape[1:-1] = sample["kspace"].shape[1:-1]
    padding = torch.from_numpy(np.random.randn(*pad_shape)).round().bool()
    sample["kspace"] = (~padding) * sample["kspace"]

    transform = ComputeZeroPadding()
    sample = transform(sample)

    assert torch.allclose(sample["padding"], padding)


@pytest.mark.parametrize(
    "shape",
    [(5, 7, 6), (3, 4, 6, 4)],
)
def test_ApplyZeroPadding(shape):

    sample = create_sample(shape + (2,))
    pad_shape = [1 for _ in range(len(sample["kspace"].shape))]
    pad_shape[1:-1] = sample["kspace"].shape[1:-1]
    padding = torch.from_numpy(np.random.randn(*pad_shape)).round().bool()
    sample.update({"padding": padding})

    kspace = sample["kspace"]
    transform = ApplyZeroPadding()
    sample = transform(sample)

    assert torch.allclose(sample["kspace"], (~padding) * kspace)


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
    [None, True],
)
@pytest.mark.parametrize(
    "use_shape",
    [True, False],
)
def test_CreateSamplingMask(shape, return_acs, padding, use_shape):

    sample = create_sample(shape + (2,))
    if padding:
        pad_shape = [1 for _ in range(len(sample["kspace"].shape))]
        pad_shape[1:-1] = sample["kspace"].shape[1:-1]
        sample.update({"padding": torch.from_numpy(np.random.randn(*pad_shape))})
    transform = CreateSamplingMask(
        mask_func=_mask_func,
        shape=shape[1:] if use_shape else None,
        return_acs=return_acs,
    )
    sample = transform(sample)
    assert "sampling_mask" in sample
    assert tuple(sample["sampling_mask"].shape) == (1,) + sample["kspace"].shape[1:-1] + (1,)
    if return_acs:
        assert "acs_mask" in sample


@pytest.mark.parametrize(
    "shape",
    [(4, 32, 32), (3, 10, 16)],
)
def test_ApplyMask(shape):
    sample = create_sample(shape=shape + (2,))
    transform = ApplyMask()
    # Check error raise when sampling mask not present in sample
    with pytest.raises(ValueError):
        sample = transform(sample)
    sample.update({"sampling_mask": torch.rand(shape[1:]).round().unsqueeze(0).unsqueeze(-1)})
    sample = transform(sample)
    assert "masked_kspace" in sample

    mask = ~(torch.abs(sample["masked_kspace"]).sum(dim=(0, -1)) == 0)
    assert torch.allclose(mask, sample["sampling_mask"].squeeze().bool())


@pytest.mark.parametrize(
    "shape",
    [(3, 10, 16)],
)
@pytest.mark.parametrize(
    "crop",
    [(5, 6), "reconstruction_size", None, "invalid_key"],
)
@pytest.mark.parametrize(
    "image_space_center_crop",
    [True, False],
)
@pytest.mark.parametrize(
    "random_crop_sampler_type, random_crop_sampler_gaussian_sigma",
    [
        ["uniform", None],
        ["gaussian", None],
        ["gaussian", [1.0, 2.0]],
    ],
)
@pytest.mark.parametrize(
    "random_crop_sampler_use_seed",
    [True, False],
)
def test_CropKspace(
    shape,
    crop,
    image_space_center_crop,
    random_crop_sampler_type,
    random_crop_sampler_use_seed,
    random_crop_sampler_gaussian_sigma,
):

    sample = create_sample(
        shape=shape + (2,),
        sensitivity_map=torch.rand(shape + (2,)),
        sampling_mask=torch.rand(shape[1:]).round().unsqueeze(0).unsqueeze(-1),
        input_image=torch.rand((1,) + shape[1:] + (2,)),
    )
    args = {
        "crop": crop,
        "image_space_center_crop": image_space_center_crop,
        "random_crop_sampler_type": random_crop_sampler_type,
        "random_crop_sampler_use_seed": random_crop_sampler_use_seed,
        "random_crop_sampler_gaussian_sigma": random_crop_sampler_gaussian_sigma,
    }
    crop_shape = crop
    if crop is None:
        with pytest.raises(ValueError):
            transform = CropKspace(**args)
    elif crop == "invalid_key":
        with pytest.raises(AssertionError):
            transform = CropKspace(**args)
            sample = transform(sample)
    else:
        if crop == "reconstruction_size":
            crop_shape = tuple((d // 2 for d in shape[1:]))
            sample.update({"reconstruction_size": crop_shape})

        transform = CropKspace(**args)

        sample = transform(sample)
        assert sample["kspace"].shape == (shape[0],) + crop_shape + (2,)


@pytest.mark.parametrize(
    "shape",
    [(3, 10, 16)],
)
@pytest.mark.parametrize(
    "type",
    [RandomFlipType.horizontal, RandomFlipType.vertical, RandomFlipType.random],
)
def test_random_flip(shape, type):
    sample = create_sample(shape=shape + (2,))
    kspace = sample["kspace"]
    image = modulus(ifft2(kspace, dim=(1, 2))).numpy()
    transform = RandomFlip(fft2, ifft2, type, p=1)
    sample = transform(sample)
    flipped_image = modulus(ifft2(sample["kspace"], dim=(1, 2))).numpy()
    if type == "horizontal":
        assert np.allclose(np.flip(image, 2), flipped_image, 0.0001)
    elif type == "vertical":
        assert np.allclose(np.flip(image, 1), flipped_image, 0.0001)
    else:
        assert np.allclose(np.flip(image, 1), flipped_image, 0.0001) | np.allclose(
            np.flip(image, 2), flipped_image, 0.0001
        )


@pytest.mark.parametrize(
    "shape",
    [(3, 10, 16)],
)
@pytest.mark.parametrize(
    "degree",
    [90, -90, 180],
)
def test_random_rotation(shape, degree):
    sample = create_sample(shape=shape + (2,), reconstruction_size=shape[1:] + (1,))
    kspace = sample["kspace"]
    image = modulus(ifft2(kspace, dim=(1, 2))).numpy()
    transform = RandomRotation(fft2, ifft2, degrees=(degree,), p=1)
    sample = transform(sample)
    rot_image = modulus(ifft2(sample["kspace"], dim=(1, 2))).numpy()
    k = degree // 90
    assert np.allclose(np.rot90(image, k=k, axes=(1, 2)), rot_image, 0.0001)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 5, 5),
        (4, 6, 4),
    ],
)
@pytest.mark.parametrize(
    "type_recon, complex_output",
    [
        [ReconstructionType.complex, True],
        [ReconstructionType.complex_mod, False],
        [ReconstructionType.sense, True],
        [ReconstructionType.sense_mod, False],
        [ReconstructionType.rss, False],
    ],
)
def test_ComputeImage(shape, type_recon, complex_output):
    sample = create_sample(shape=shape + (2,))
    transform = ComputeImage("kspace", "target", ifft2, type_reconstruction=type_recon)
    if type_recon in ["sense", "sense_mod"]:
        with pytest.raises(ItemNotFoundException):
            sample = transform(sample.copy())
        sample.update({"sensitivity_map": torch.rand(shape + (2,))})
    sample = transform(sample)
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
        mask_func=_mask_func,
        backward_operator=functools.partial(ifft2, dim=spatial_dims),
        use_seed=use_seed,
    )
    sample = transform(sample)
    assert "body_coil_image" in sample
    assert sample["body_coil_image"].shape == shape[1:]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 54, 46),
        (5, 37, 26),
        (4, 35, 35),
    ],
)
@pytest.mark.parametrize(
    "type_of_map, gaussian_sigma, espirit_iters, expect_error, sense_map_in_sample",
    [
        [SensitivityMapType.unit, None, None, False, False],
        [SensitivityMapType.rss_estimate, 0.5, None, False, False],
        [SensitivityMapType.rss_estimate, None, None, False, False],
        [SensitivityMapType.rss_estimate, None, None, False, True],
        [SensitivityMapType.espirit, None, 5, False, True],
        ["invalid", None, None, True, False],
    ],
)
def test_EstimateSensitivityMap(shape, type_of_map, gaussian_sigma, espirit_iters, expect_error, sense_map_in_sample):
    sample = create_sample(
        shape=shape + (2,),
        acs_mask=torch.rand((1,) + shape[1:] + (1,)).round(),
        sampling_mask=torch.rand((1,) + shape[1:] + (1,)).round(),
    )
    if sense_map_in_sample:
        sample.update({"sensitivity_map": torch.rand(shape + (2,))})
    args = {
        "kspace_key": "kspace",
        "backward_operator": functools.partial(ifft2),
        "type_of_map": type_of_map,
        "gaussian_sigma": gaussian_sigma,
        "espirit_max_iters": espirit_iters,
        "espirit_kernel_size": 3,
    }
    if expect_error:
        with pytest.raises(ValueError):
            transform = EstimateSensitivityMap(**args)
            sample = transform(sample)
    else:
        transform = EstimateSensitivityMap(**args)
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
    [
        [(3, 10, 16), 5],
        [(5, 7, 6), 5],
        [(4, 5, 5), 2],
        [(4, 5, 5), None],
        [(3, 4, 6, 4), 4],
        [(5, 3, 3, 4), 3],
    ],
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
    [None, "masked_kspace", "kspace", "scaling_factor"],
)
@pytest.mark.parametrize(
    "percentile",
    [None, 0.9],
)
@pytest.mark.parametrize(
    "norm_keys",
    [None, ["kspace", "masked_kspace", "target"]],
)
def test_Normalize(shape, normalize_key, percentile, norm_keys):
    sample = create_sample(
        shape=shape + (2,),
        masked_kspace=torch.rand(shape + (2,)),
        sensitivity_map=torch.rand(shape + (2,)),
        sampling_mask=torch.rand(shape[1:]).round().unsqueeze(0).unsqueeze(-1),
        scaling_factor=torch.rand(1),
    )
    transform = Compose(
        [
            ComputeScalingFactor(normalize_key, percentile, "scaling_factor"),
            Normalize("scaling_factor", keys_to_normalize=norm_keys),
        ]
    )
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
    [(5, 3), (5, 3, 4)],
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


@pytest.mark.parametrize(
    "shape, spatial_dims",
    [[(5, 3, 4), (1, 2)], [(5, 4, 5, 6), (2, 3)]],
)
@pytest.mark.parametrize(
    "estimate_body_coil_image",
    [True, False],
)
@pytest.mark.parametrize(
    "image_center_crop",
    [True, False],
)
def test_build_mri_transforms(shape, spatial_dims, estimate_body_coil_image, image_center_crop):
    transform = build_mri_transforms(
        forward_operator=functools.partial(fft2, dim=spatial_dims),
        backward_operator=functools.partial(ifft2, dim=spatial_dims),
        mask_func=_mask_func,
        crop=None,
        crop_type="uniform",
        scaling_key="masked_kspace",
        estimate_body_coil_image=estimate_body_coil_image,
        image_center_crop=image_center_crop,
    )
    sample = create_sample(shape, kspace=np.random.randn(*shape) + 1.0j * np.random.randn(*shape))

    sample = transform(sample)

    assert all(
        key in sample.keys()
        for key in ["sampling_mask", "sensitivity_map", "target", "masked_kspace", "scaling_diff", "scaling_factor"]
    )
    assert sample["masked_kspace"].shape == shape + (2,)
    assert sample["sensitivity_map"].shape == shape + (2,)
    assert sample["sampling_mask"].shape == (1,) + shape[1:] + (1,)
    assert sample["target"].shape == shape[1:]
