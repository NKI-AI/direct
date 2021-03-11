# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import numpy as np
import warnings
import functools
import torch.nn as nn

from typing import Dict, Any, Callable, Optional, Iterable

from direct.data import transforms as T
from direct.utils import DirectTransform


import logging

logger = logging.getLogger(__name__)


class Compose(DirectTransform):
    """Compose several transformations together, for instance ClipAndScale and a flip.
    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
    additional dependencies.
    """

    def __init__(self, transforms: Iterable) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def __repr__(self):
        repr_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            repr_string += "\n"
            repr_string += f"    {transform}"
        repr_string += "\n)"
        return repr_string


# TODO: Flip augmentation
class RandomFlip(DirectTransform):
    def __call__(self):
        raise NotImplementedError


class CreateSamplingMask(DirectTransform):
    def __init__(self, mask_func, shape=None, use_seed=True, return_acs=False):
        super().__init__()
        self.mask_func = mask_func
        self.shape = shape
        self.use_seed = use_seed
        self.return_acs = return_acs

    def __call__(self, sample):
        if not self.shape:
            shape = sample["kspace"].shape[1:]
        elif any(_ is None for _ in self.shape):  # Allow None as values.
            kspace_shape = list(sample["kspace"].shape[1:-1])
            shape = tuple([_ if _ else kspace_shape[idx] for idx, _ in enumerate(self.shape)]) + (2,)
        else:
            shape = self.shape + (2,)

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        mask = self.mask_func(shape, seed, return_acs=False)

        sampling_mask = mask.refine_names(*sample["kspace"].names)

        if sample.get("padding_left", 0) > 0 or sample.get("padding_right", 0) > 0:
            if sampling_mask.names[2] != "width":
                raise NotImplementedError(
                    "Currently only support for the `width` axis" f" to be at the 2th position when padding."
                )

            if sample["kspace"].shape[2] != shape[-2]:
                raise ValueError(
                    "When padding in left or right is present, " f"you cannot crop in the phase-encoding direction!"
                )

            padding_left = sample["padding_left"]
            padding_right = sample["padding_right"]

            sampling_mask[:, :, :padding_left, :] = 0
            sampling_mask[:, :, padding_right:, :] = 0

        sample["sampling_mask"] = sampling_mask.rename(None)

        if self.return_acs:
            kspace_shape = sample["kspace"].shape[1:]
            sample["acs_mask"] = self.mask_func(kspace_shape, seed, return_acs=True)

        return sample


class CropAndMask(DirectTransform):
    """
    Data Transformer for training RIM models.
    """

    def __init__(
        self,
        crop,
        use_seed=True,
        forward_operator=T.fft2,
        backward_operator=T.ifft2,
        image_space_center_crop=False,
        random_crop_sampler_type="uniform",
    ):
        """
        Parameters
        ----------
        crop : tuple or None
            Size to crop input_image to.
        mask_func : direct.common.subsample.MaskFunc
            A function which creates a mask of the appropriate shape.
        use_seed : bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time.
        forward_operator : callable
            The __call__ operator, e.g. some form of FFT (centered or uncentered).
        backward_operator : callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        image_space_center_crop : bool
            If set, the crop in the data will be taken in the center
        random_crop_sampler_type : str
            If "uniform" the random cropping will be done by uniformly sampling `crop`, as opposed to `gaussian` which
            will sample from a gaussian distribution.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.use_seed = use_seed
        self.image_space_center_crop = image_space_center_crop

        self.crop = crop
        self.crop_func = None
        if self.crop:
            if self.image_space_center_crop:
                self.crop_func = T.complex_center_crop
            else:
                self.crop_func = functools.partial(T.complex_random_crop, sampler=self.random_crop_sampler_type)

        self.random_crop_sampler_type = random_crop_sampler_type

        self.forward = forward_operator
        self.backward_operator = backward_operator

        self.image_space_center_crop = image_space_center_crop

    def __call__(self, sample: Dict[str, Any]):
        """

        Parameters
        ----------
        sample: dict

        Returns
        -------
        data dictionary
        """
        kspace = sample["kspace"]

        # Image-space croppable objects
        croppable_images = ["sensitivity_map", "input_image"]
        sensitivity_map = sample.get("sensitivity_map", None)
        sampling_mask = sample["sampling_mask"]
        backprojected_kspace = self.backward_operator(kspace)

        # TODO: Also create a kspace-like crop function
        if self.crop:
            cropped_output = self.crop_func(
                [
                    backprojected_kspace,
                    *[sample[_] for _ in croppable_images if _ in sample],
                ],
                self.crop,
                contiguous=True,
            )
            backprojected_kspace = cropped_output[0]
            for idx, key in enumerate(croppable_images):
                sample[key] = cropped_output[1 + idx]

            # Compute new k-space for the cropped input_image
            kspace = self.forward_operator(backprojected_kspace)

        masked_kspace, sampling_mask = T.apply_mask(kspace, sampling_mask)

        sample["target"] = T.root_sum_of_squares(backprojected_kspace, dim="coil")
        sample["masked_kspace"] = masked_kspace
        sample["sampling_mask"] = sampling_mask
        sample["kspace"] = kspace  # The cropped kspace

        if sensitivity_map is not None:
            sample["sensitivity_map"] = sensitivity_map

        return sample


class ComputeImage(DirectTransform):
    def __init__(self, kspace_key, target_key, backward_operator, type_reconstruction="complex"):
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        self.target_key = target_key

        self.type_reconstruction = type_reconstruction

        if type_reconstruction.lower() not in ["complex", "sense", "rss"]:
            raise ValueError(
                f"Only `complex`, `rss` and `sense` are possible choices for `reconstruction_type`. "
                f"Got {self.type_reconstruction}."
            )

    def __call__(self, sample):
        kspace_data = sample[self.kspace_key]

        # Get complex-valued data solution
        image = self.backward_operator(kspace_data)

        if self.type_reconstruction == "complex":
            sample[self.target_key] = image.sum("coil")
        elif self.type_reconstruction.lower() == "rss":
            sample[self.target_key] = T.root_sum_of_squares(image, dim="coil")
        elif self.type_reconstruction == "sense":
            if "sensitivity_map" not in sample:
                raise ValueError("Sensitivity map is required for SENSE reconstruction.")
            raise NotImplementedError("SENSE is not implemented.")

        return sample


class EstimateBodyCoilImage(DirectTransform):
    def __init__(self, mask_func, backward_operator, use_seed=True):
        super().__init__()
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.backward_operator = backward_operator

    def __call__(self, sample):
        kspace = sample["kspace"]
        # We need to create an ACS mask based on the shape of this kspace, as it can be cropped.

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        kspace_shape = sample["kspace"].shape[1:]
        acs_mask = self.mask_func(kspace_shape, seed, return_acs=True)

        kspace = acs_mask * kspace + 0.0
        acs_image = self.backward_operator(kspace)

        sample["body_coil_image"] = T.root_sum_of_squares(acs_image, dim="coil")
        return sample


class EstimateSensitivityMap(DirectTransform):
    def __init__(
        self,
        kspace_key: str,
        backward_operator: Callable = T.ifft2,
        type_of_map: Optional[str] = "unit",
        gaussian_sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        self.type_of_map = type_of_map
        self.gaussian_sigma = gaussian_sigma

    def estimate_acs_image(self, sample):
        kspace_data = sample[self.kspace_key]

        if kspace_data.shape[0] == 1:
            warnings.warn(
                "`Single-coil data, skipping estimation of sensitivity map. "
                f"This warning will be displayed only once."
            )
            return sample

        if "sensitivity_map" in sample:
            warnings.warn(
                "`sensitivity_map` is given, but will be overwritten. " f"This warning will be displayed only once."
            )

        if self.gaussian_sigma == 0 or not self.gaussian_sigma:
            kspace_acs = kspace_data * sample["acs_mask"] + 0.0  # + 0.0 removes the sign of zeros.
        else:
            gaussian_mask = torch.linspace(-1, 1, kspace_data.size("width"), dtype=kspace_data.dtype).refine_names(
                "width"
            )
            gaussian_mask = torch.exp(-((gaussian_mask / self.gaussian_sigma) ** 2))
            kspace_acs = kspace_data * sample["acs_mask"] * gaussian_mask.align_as(kspace_data) + 0.0

        # Get complex-valued data solution
        acs_image = self.backward_operator(kspace_acs)
        return acs_image

    def __call__(self, sample):
        if self.type_of_map == "unit":
            kspace = sample["kspace"]
            sensitivity_map = torch.zeros(kspace.shape).float()
            # TODO(jt): Named variant, this assumes the complex channel is last.
            if not kspace.names[-1] == "complex":
                raise NotImplementedError("Assuming last channel is complex.")
            sensitivity_map[..., 0] = 1.0
            sample["sensitivity_map"] = sensitivity_map.refine_names(*kspace.names).to(kspace.device)

        elif self.type_of_map == "rss_estimate":
            acs_image = self.estimate_acs_image(sample)
            acs_image_rss = T.root_sum_of_squares(acs_image, dim="coil").align_as(acs_image)
            sample["sensitivity_map"] = T.safe_divide(acs_image, acs_image_rss)
        else:
            raise ValueError(f"Expected type of map to be either `unit` or `rss_estimate`. Got {self.type_of_map}.")

        return sample


class DeleteKeys(DirectTransform):
    """
    Remove keys from the sample.
    """

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                del sample[key]

        return sample


class PadCoilDimension(DirectTransform):
    """
    Pad the coils by zeros to a given number of coils.
    Useful if you want to collate volumes with different coil dimension.
    """

    def __init__(self, pad_coils: Optional[int] = None, key: str = "masked_kspace"):
        """
        Parameters
        ----------
        pad_coils : int
            Number of coils to pad to.
        key: tuple
            Key to pad in sample
        """
        super().__init__()
        self.num_coils = pad_coils
        self.key = key

    def __call__(self, sample):
        if not self.num_coils:
            return sample

        if self.key not in sample:
            return sample

        data = sample[self.key]
        curr_num_coils = data.shape[data.names.index("coil")]
        if curr_num_coils > self.num_coils:
            raise ValueError(
                f"Tried to pad to {self.num_coils} coils, but already have {curr_num_coils} for "
                f"{sample['filename']}."
            )
        elif curr_num_coils == self.num_coils:
            return sample

        shape = data.shape
        coils_dim = data.names.index("coil")
        num_coils = shape[coils_dim]
        padding_data_shape = list(shape).copy()
        padding_data_shape[coils_dim] = max(self.num_coils - num_coils, 0)
        zeros = torch.zeros(padding_data_shape, dtype=data.dtype)

        data_names = data.names
        sample[self.key] = torch.cat([zeros, data.rename(None)], dim=coils_dim).refine_names(*data_names)

        return sample


class Normalize(DirectTransform):
    """
    Normalize the input data either to the percentile or to the maximum.
    """

    def __init__(self, normalize_key="masked_kspace", percentile=0.99):
        """

        Parameters
        ----------
        normalize_key : str
            Key name to compute the data for. If the maximum has to be computed on the ACS, ensure the reconstruction
            on the ACS is available (typically `body_coil_image`).
        percentile : float or None
            Rescale data with the given percentile. If None, the division is done by the maximum.
        """
        super().__init__()
        self.normalize_key = normalize_key
        self.percentile = percentile

        self.other_keys = [
            "masked_kspace",
            "target",
            "kspace",
            "body_coil_image",  # sensitivity_map does not require normalization.
            "initial_image",
            "initial_kspace",
        ]

    def __call__(self, sample):
        if self.normalize_key == "scaling_factor":  # This is a real-valued given number
            scaling_factor = sample["scaling_factor"]
        elif not self.normalize_key:
            scaling_factor = 1.0
        else:
            data = sample[self.normalize_key]

            # Compute the maximum and scale the input
            if self.percentile:
                # TODO: Fix when named tensors allow views.
                tview = -1.0 * T.modulus(data).rename(None).view(-1)
                scaling_factor, _ = torch.kthvalue(tview, int((1 - self.percentile) * tview.size()[0]))
                scaling_factor = -1.0 * scaling_factor
            else:
                scaling_factor = T.modulus(data).max()

        # Normalize data
        if self.normalize_key:
            for key in sample.keys():
                if key != self.normalize_key and key not in self.other_keys:
                    continue
                sample[key] = sample[key] / scaling_factor

        sample["scaling_diff"] = 0.0
        sample["scaling_div"] = scaling_factor
        return sample


class WhitenData(DirectTransform):
    def __init__(self, epsilon=1e-10, key="complex_image"):
        super().__init__()
        self.epsilon = epsilon
        self.key = key

    def complex_whiten(self, complex_image):
        # From: https://github.com/facebookresearch/fastMRI/blob/da1528585061dfbe2e91ebbe99a5d4841a5c3f43/banding_removal/fastmri/data/transforms.py#L464  # noqa
        real = complex_image[:, :, 0]
        imag = complex_image[:, :, 1]

        # Center around mean.
        mean = complex_image.mean()
        centered_complex_image = complex_image - mean

        # Determine covariance between real and imaginary.
        n = real.nelement()
        real_real = (real.mul(real).sum() - real.mean().mul(real.mean())) / n
        real_imag = (real.mul(imag).sum() - real.mean().mul(imag.mean())) / n
        imag_imag = (imag.mul(imag).sum() - imag.mean().mul(imag.mean())) / n
        V = torch.Tensor([[real_real, real_imag], [real_imag, imag_imag]])

        # Remove correlation by rotating around covariance eigenvectors.
        eig_values, eig_vecs = torch.eig(V, eigenvectors=True)

        # Scale by eigenvalues for unit variance.
        std = (eig_values[:, 0] + self.epsilon).sqrt()
        whitened_image = torch.matmul(centered_complex_image, eig_vecs) / std

        return mean, std, whitened_image

    def __call__(self, sample):
        mean, std, whitened_image = self.complex_whiten(sample[self.key])
        sample[self.key] = whitened_image


class DropNames(DirectTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        new_sample = {}

        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and any(v.names):
                new_sample[k + "_names"] = ";".join(v.names)  # collate_fn will do funky things without this.
                v = v.rename(None)
            new_sample[k] = v

        return new_sample


class AddNames(DirectTransform):
    def __init__(self, add_batch_dimension=True):
        super().__init__()
        self.add_batch_dimension = add_batch_dimension

    def __call__(self, sample):
        names = [_[:-6] for _ in sample.keys() if _[-5:] == "names"]
        new_sample = {k: v for k, v in sample.items() if k[-5:] != "names"}

        for name in names:
            if self.add_batch_dimension:
                names = ["batch"] + sample[name + "_names"][0].split(";")

            else:
                names = sample[name + "_names"].split(";")

            new_sample[name] = sample[name].rename(*names)

        return new_sample


class ToTensor(nn.Module):
    def __init__(self):
        # 2D and 3D data
        super().__init__()
        self.names = (["coil", "height", "width"], ["coil", "slice", "height", "width"])

    def __call__(self, sample):
        ndim = sample["kspace"].ndim - 1
        if ndim == 2:
            names = self.names[0]
        elif ndim == 3:
            names = self.names[1]
        else:
            raise ValueError(f"Can only cast 2D and 3D data (+coil) to tensor. Got {ndim}.")

        sample["kspace"] = T.to_tensor(sample["kspace"], names=names).float()
        # Sensitivity maps are not necessarily available in the dataset.
        if "initial_kspace" in sample:
            sample["initial_kspace"] = T.to_tensor(sample["initial_kspace"], names=names).float()
        if "initial_image" in sample:
            sample["initial_image"] = T.to_tensor(sample["initial_image"], names=names[1:]).float()

        if "sensitivity_map" in sample:
            sample["sensitivity_map"] = T.to_tensor(sample["sensitivity_map"], names=names).float()
        if "target" in sample:
            sample["target"] = sample["target"].refine_names(*names)
        if "sampling_mask" in sample:
            sample["sampling_mask"] = torch.from_numpy(sample["sampling_mask"]).byte()
        if "acs_mask" in sample:
            sample["acs_mask"] = torch.from_numpy(sample["acs_mask"])
        if "scaling_div" in sample:
            sample["scaling_div"] = torch.tensor(sample["scaling_div"]).float()
        if "loglikelihood_scaling" in sample:
            sample["loglikelihood_scaling"] = (
                torch.from_numpy(np.asarray(sample["loglikelihood_scaling"])).float().refine_names("coil")
            )

        return sample


def build_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Optional[Callable],
    crop: Optional[int] = None,
    crop_type: Optional[str] = None,
    image_center_crop: bool = False,
    estimate_sensitivity_maps: bool = True,
    estimate_body_coil_image: bool = False,
    sensitivity_maps_gaussian: Optional[float] = None,
    pad_coils: Optional[int] = None,
    scaling_key: str = "scaling_factor",
    use_seed: bool = True,
) -> object:
    """
    Build transforms for MRI.
    - Converts input to (complex-valued) tensor.
    - Adds a sampling mask if `mask_func` is defined.
    - Adds coil sensitivities and / or the body coil_image
    - Crops the input data if needed and masks the fully sampled k-space.
    - Add a target.
    - Normalize input data.
    - Pads the coil dimension.

    Parameters
    ----------
    backward_operator : callable
    forward_operator : callable
    mask_func : callable or none
    crop : int or none
    crop_type : str or None
        Type of cropping, either "gaussian" or "uniform".
    image_center_crop : bool
    estimate_sensitivity_maps : bool
    estimate_body_coil_image : bool
    sensitivity_maps_gaussian : float
        Optional sigma for gaussian weighting of sensitivity map.
    pad_coils : int
        Number of coils to pad data to.
    scaling_key : str
        Key to use to compute scaling factor for.
    use_seed : bool

    Returns
    -------
    object : a transformation object.
    """
    # TODO: Use seed

    mri_transforms = [ToTensor()]
    if mask_func:
        mri_transforms.append(
            CreateSamplingMask(
                mask_func,
                shape=crop,
                use_seed=use_seed,
                return_acs=estimate_sensitivity_maps,
            )
        )

    mri_transforms += [
        EstimateSensitivityMap(
            kspace_key="kspace",
            backward_operator=backward_operator,
            type_of_map="unit" if not estimate_sensitivity_maps else "rss_estimate",
            gaussian_sigma=sensitivity_maps_gaussian,
        ),
        DeleteKeys(keys=["acs_mask"]),
        CropAndMask(
            crop,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            image_space_center_crop=image_center_crop,
            random_crop_sampler_type=crop_type,
        ),
    ]
    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))

    mri_transforms += [
        Normalize(
            normalize_key=scaling_key,
            percentile=0.99,
        ),
        PadCoilDimension(pad_coils=pad_coils, key="masked_kspace"),
        PadCoilDimension(pad_coils=pad_coils, key="sensitivity_map"),
        DeleteKeys(keys=["kspace"]),
        DropNames(),
    ]

    mri_transforms = Compose(mri_transforms)
    return mri_transforms
