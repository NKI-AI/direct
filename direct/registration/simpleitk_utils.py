# Copyright (c) DIRECT Contributors

"""SimpleITK utility functions for transformations between different image formats."""

from __future__ import annotations

from typing import Union

import numpy as np
import SimpleITK as sitk
import torch


def convert_to_sitk_image(input_image: np.ndarray | torch.Tensor) -> sitk.Image:
    """Converts a numpy array or PyTorch tensor to a SimpleITK image.

    Parameters
    ----------
    input_image : Union[np.ndarray, torch.Tensor]
        Input image as a numpy array or PyTorch tensor.

    Returns
    -------
    sitk.Image
        SimpleITK image.
    """
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.numpy()
    sitk_image = sitk.GetImageFromArray(input_image)
    return sitk_image


def convert_to_tensor(image: sitk.Image) -> torch.Tensor:
    """Converts a SimpleITK image to a PyTorch tensor.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image.

    Returns
    -------
    torch.Tensor
        PyTorch tensor.
    """
    array = sitk.GetArrayFromImage(image)
    tensor = torch.tensor(array, dtype=torch.float32)
    return tensor
