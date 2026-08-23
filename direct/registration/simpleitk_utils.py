# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""SimpleITK utility functions for transformations between different image formats."""

import numpy as np
import SimpleITK as sitk
import torch


def convert_to_sitk_image(input_image: np.ndarray | torch.Tensor) -> sitk.Image:
    """Converts a numpy array or PyTorch tensor to a SimpleITK image.

    Args:
        input_image: Input image as a numpy array or PyTorch tensor.

    Returns:
        SimpleITK image.
    """
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.numpy()
    sitk_image = sitk.GetImageFromArray(input_image)
    return sitk_image


def convert_to_tensor(image: sitk.Image) -> torch.Tensor:
    """Converts a SimpleITK image to a PyTorch tensor.

    Args:
        image: SimpleITK image.

    Returns:
        PyTorch tensor.
    """
    array = sitk.GetArrayFromImage(image)
    tensor = torch.tensor(array, dtype=torch.float32)
    return tensor
