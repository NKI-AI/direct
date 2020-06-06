# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import numpy as np
import warnings

from typing import Dict, Any, Callable, Optional

from direct.data import transforms

import logging
logger = logging.getLogger(__name__)


class Compose:
    """Compose several transformations together, for instance ClipAndScale and a flip.
    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
    additional dependencies.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        repr_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            repr_string += '\n'
            repr_string += f'    {transform}'
        repr_string += '\n)'
        return repr_string


# TODO: Flip augmentation
class RandomFlip:
    def __call__(self):
        raise NotImplementedError


class CropAndMask:
    """
    Data Transformer for training RIM models.
    """

    def __init__(self,
                 crop, mask_func, use_seed=True, kspace_crop_probability=0.0, image_space_center_crop=False):
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
        kspace_crop_probability : float
            Probability a crop in k-space will be done rather than input_image space.
        image_space_center_crop : bool
            If set, the crop in the image will be taken in the center.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.mask_func = mask_func
        self.use_seed = use_seed

        self.crop = crop

        self.forward_operator = transforms.fft2
        self.backward_operator = transforms.ifft2
        self.kspace_crop_probability = kspace_crop_probability
        self.image_space_center_crop = image_space_center_crop

    def __central_kspace_crop(self, kspace, masked_kspace, mask, sensitivity_map=None):
        if self.crop is not None:
            kspace, masked_kspace, mask = transforms.complex_center_crop(
                [kspace, masked_kspace, mask], self.crop, contiguous=True)

            if sensitivity_map is not None:
                # TODO: Linear does not work yet in 4D, working with bilinear instead.
                sensitivity_map = torch.nn.functional.interpolate(
                    sensitivity_map.permute(0, 3, 1, 2).rename(None),
                    size=self.crop,
                    mode='bilinear',
                    align_corners=True).permute(0, 2, 3, 1).refine_names(*sensitivity_map.names)

        backprojected_kspace = self.backward_operator(kspace)
        return kspace, masked_kspace, mask, backprojected_kspace, sensitivity_map

    def __random_image_crop(self, kspace, sensitivity_map=None):
        """
        Crop the input_image randomly based on the k-space data. To do this, first the complex-valued input_image needs to be
        computed by a backward transform, and subsequently cropped and again forward transformed to get the new k-space.

        Parameters
        ----------
        kspace : torch.Tensor
        sensitivity_map : torch.Tensor

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
        """
        backprojected_kspace = self.backward_operator(kspace)

        if self.crop is not None:
            crop_func = transforms.complex_center_crop if self.image_space_center_crop \
                else transforms.complex_random_crop
            if sensitivity_map is not None:
                backprojected_kspace, sensitivity_map = crop_func(
                    [backprojected_kspace, sensitivity_map], self.crop, contiguous=True)
            else:
                backprojected_kspace = crop_func(backprojected_kspace, self.crop, contiguous=True)

            # Compute new k-space for the cropped input_image
            kspace = self.forward_operator(backprojected_kspace)

        return kspace, backprojected_kspace, sensitivity_map

    def __call__(self, sample: Dict[str, Any]):
        """

        Parameters
        ----------
        sample: dict

        Returns
        -------
        data dictionary
        """
        kspace = sample['kspace']
        sensitivity_map = sample.get('sensitivity_map', None)
        filename = sample['filename']

        if 'sampling_mask' in sample and self.mask_func is not None:
            warnings.warn(f'`sampling_mask` is passed by the Dataset class, yet `mask_func` is also set. '
                          f'This will be ignored and the `sampling_mask` will be used instead. '
                          f'Be aware of this as it can lead to unexpected results. '
                          f'This warning will be issued only once.')
            raise NotImplementedError('This is required when a mask is present,'
                                      ' but in this case this should be applied differently!')

        seed = None if not self.use_seed else tuple(map(ord, str(filename)))

        if np.random.random() >= self.kspace_crop_probability:
            kspace, backprojected_kspace, sensitivity_map = self.__random_image_crop(kspace, sensitivity_map)
            masked_kspace, sampling_mask = transforms.apply_mask(kspace, self.mask_func, seed)

        else:
            masked_kspace, sampling_mask = transforms.apply_mask(kspace, self.mask_func, seed)
            kspace, masked_kspace, sampling_mask, backprojected_kspace, sensitivity_map = self.__central_kspace_crop(
                kspace, masked_kspace, sampling_mask, sensitivity_map)

        sample['target'] = transforms.root_sum_of_squares(backprojected_kspace, dim='coil')
        del sample['kspace']
        sample['masked_kspace'] = masked_kspace
        sample['sampling_mask'] = sampling_mask

        if sensitivity_map is not None:
            sample['sensitivity_map'] = sensitivity_map

        return sample


class ComputeImage:
    def __init__(self, kspace_key, target_key, type_reconstruction='complex'):
        self.backward_operator = transforms.ifft2
        self.kspace_key = kspace_key
        self.target_key = target_key

        self.type_reconstruction = type_reconstruction

        if not type_reconstruction.lower() in ['complex', 'sense', 'rss']:
            raise ValueError(f'Only `complex`, `rss` and `sense` are possible choices for `reconstruction_type`. '
                             f'Got {self.type_reconstruction}.')

    def __call__(self, sample):
        kspace_data = sample[self.kspace_key]

        # Get complex-valued image solution
        image = self.backward_operator(kspace_data)

        if self.type_reconstruction == 'complex':
            sample[self.target_key] = image.sum('coil')
        elif self.type_reconstruction.lower() == 'rss':
            sample[self.target_key] = transforms.root_sum_of_squares(image, dim='coil')
        elif self.type_reconstruction == 'sense':
            if 'sensitivity_map' not in sample:
                raise ValueError('Sensitivity map is required for SENSE reconstruction.')
            raise NotImplementedError('SENSE is not implemented.')

        return sample


class Normalize:
    """
    Normalize the input data either to the percentile or to the maximum
    """
    # TODO: Central band of kspace
    def __init__(self, normalize_key='masked_kspace', percentile=0.99):
        """

        Parameters
        ----------
        normalize_key : str
            Key name to compute the data for.
        percentile : float or None
            Rescale data with the given percentile. If None, the division is done by the maximum.
        """
        self.normalize_key = normalize_key
        self.percentile = percentile

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : dict

        Returns
        -------
        data dictionary

        TODO: Normalization of the sensitivity map should be done in the data loader.
        """
        data = sample[self.normalize_key]

        # Compute the maximum and scale the input
        if self.percentile:
            # TODO: Fix when named tensors allow views.
            tview = -1.0 * transforms.modulus(data).rename(None).view(-1)
            image_max, _ = torch.kthvalue(tview, int((1 - self.percentile) * tview.size()[0]))
            image_max = -1.0 * image_max
        else:
            image_max = transforms.modulus(data).max()

        # Normalize data
        for key in sample.keys():
            # TODO: Reconsider this.
            if any([_ in key for _ in [self.normalize_key, 'masked_kspace', 'target', 'kspace']]):
                sample[key] = sample[key] / image_max

        sample['scaling_factor'] = image_max

        return sample


class DropNames:
    def __init__(self):
        pass

    def __call__(self, sample):
        new_sample = {}

        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and any(v.names):
                new_sample[k + '_names'] = ';'.join(v.names)  # collate_fn will do funky things without this.
                v = v.rename(None)
            new_sample[k] = v

        return new_sample


class AddNames:
    def __init__(self, add_batch_dimension=True):
        self.add_batch_dimension = add_batch_dimension

    def __call__(self, sample):
        names = [_[:-6] for _ in sample.keys() if _[-5:] == 'names']
        new_sample = {k: v for k, v in sample.items() if k[-5:] != 'names'}

        for name in names:
            if self.add_batch_dimension:
                names = ['batch'] + sample[name + '_names'][0].split(';')

            else:
                names = sample[name + '_names'].split(';')

            new_sample[name] = sample[name].rename(*names)

        return new_sample


class ToTensor:
    def __init__(self):
        self.names = ['coil', 'height', 'width']

    def __call__(self, sample):
        sample['sensitivity_map'] = transforms.to_tensor(sample['sensitivity_map'], names=self.names).float()
        sample['kspace'] = transforms.to_tensor(sample['kspace'], names=self.names).float()
        if 'target' in sample:
            sample['target'] = sample['target'].refine_names(*self.names)
        if 'sampling_mask' in sample:
            sample['sampling_mask'] = torch.from_numpy(sample['sampling_mask'])

        return sample


def build_mri_transforms(
        train_mask_func: Callable, val_mask_func: Optional[Callable] = None, crop: Optional[int] = None) -> object:
    # Data is cropped to the same size, can trigger CUDNN benchmark.
    if crop is not None:
        logger.info(f'Crop is fixed at {crop}. Enabling cuDNN benchmark.')
        torch.backends.cudnn.benchmark = True

    train_transforms = Compose([
        ToTensor(),
        CropAndMask(crop, train_mask_func, image_space_center_crop=False),
        # Add target here.
        ComputeImage(kspace_key='masked_kspace', target_key='masked_image', type_reconstruction='complex'),
        Normalize(normalize_key='masked_image', percentile=0.99),
        DropNames()
    ])

    if val_mask_func:
        val_transforms = Compose([
            ToTensor(),
            CropAndMask(crop, val_mask_func, image_space_center_crop=True),
            ComputeImage(kspace_key='masked_kspace', target_key='masked_image', type_reconstruction='complex'),
            Normalize(normalize_key='masked_image', percentile=0.99),
            DropNames()
        ])
        return train_transforms, val_transforms

    return train_transforms
