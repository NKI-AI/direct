# coding=utf-8
# Copyright (c) DIRECT Contributors

import torch

import direct.data.transforms as T


def cropper(source, target, resolution):
    """2D source/target cropper.

    Parameters
    ----------
    source: torch.Tensor
        Has shape (batch, height, width)
    target: torch.Tensor
        Has shape (batch, height, width)
    resolution: tuple
        Target resolution.
    """

    if not resolution or all(_ == 0 for _ in resolution):
        return source.unsqueeze(1), target.unsqueeze(1)  # Added channel dimension.

    source_abs = T.center_crop(source, resolution).unsqueeze(1)  # Added channel dimension.
    target_abs = T.center_crop(target, resolution).unsqueeze(1)  # Added channel dimension.

    return source_abs, target_abs


def compute_model_per_coil(models, model_name, data, coil_dim):
    """Computes model per coil."""
    # data is of shape (batch, coil, complex=2, height, width)
    output = []

    for idx in range(data.size(coil_dim)):
        subselected_data = data.select(coil_dim, idx)
        output.append(models[model_name](subselected_data))
    output = torch.stack(output, dim=coil_dim)

    # output is of shape (batch, coil, complex=2, height, width)
    return output


def compute_resolution(key, reconstruction_size):
    """Computes resolution.

    Parameters
    ----------
    key: str
        Can be 'header', 'training' or None.
    reconstruction_size: tuple
        Reconstruction size.

    Returns
    -------
    resolution: tuple
        Resolution of reconstruction.
    """
    if key == "header":
        # This will be of the form [tensor(x_0, x_1, ...), tensor(y_0, y_1,...), tensor(z_0, z_1, ...)] over
        # batches.
        resolution = [_.detach().cpu().numpy().tolist() for _ in reconstruction_size]
        # The volume sampler should give validation indices belonging to the *same* volume, so it should be
        # safe taking the first element, the matrix size are in x,y,z (we work in z,x,y).
        resolution = [_[0] for _ in resolution][:-1]
    elif key == "training":
        resolution = key
    elif not key:
        resolution = None
    else:
        raise ValueError(
            "Cropping should be either set to `header` to get the values from the header or "
            "`training` to take the same value as training."
        )
    return resolution
