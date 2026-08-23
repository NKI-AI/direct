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

"""direct.utils.bbox module."""

import numpy as np
import torch


def crop_to_bbox(data: np.ndarray | torch.Tensor, bbox: list[int], pad_value: int = 0) -> np.ndarray | torch.Tensor:
    """Extract bbox from images, coordinates can be negative.

    Args:
        data: nD array or torch tensor.
        bbox: bbox of the form (coordinates, size), for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with
            height 2 and width 1.
        pad_value: if bounding box would be out of the image, this is value the patch will be padded with.

    Returns:
        Numpy array of data cropped to BoundingBox
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise ValueError(f"Expected `data` to be ndarray or tensor. Got {type(data)}.")

    # Coordinates, size
    ndim = len(bbox) // 2
    if len(bbox) % 2 != 0:
        raise ValueError(f"Bounding box should have the form of [x_0, x_1, ..., h_0, h_1], but got length {ndim}.")
    bbox_coords, bbox_size = np.asarray(bbox[:ndim]), np.asarray(bbox[ndim:])
    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0

    r_offset = (bbox_coords + bbox_size) - np.array(data.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j in zip(bbox_coords + l_offset, bbox_coords + bbox_size - r_offset)]

    if isinstance(data, torch.Tensor):
        # TODO(jt): Investigate if clone is needed
        out = data[tuple(region_idx)].clone()
    else:
        out = data[tuple(region_idx)].copy()

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    if isinstance(data, torch.Tensor):
        patch = pad_value * torch.ones(bbox_size.tolist(), dtype=data.dtype)
    else:
        patch = pad_value * np.ones(bbox_size.tolist(), dtype=data.dtype)

    patch_idx = [slice(i, j) for i, j in zip(l_offset, bbox_size - r_offset)]
    patch[tuple(patch_idx)] = out  # ty: ignore[invalid-assignment]

    return patch


def crop_to_largest(data: list[np.ndarray | torch.Tensor], pad_value: int = 0) -> list[np.ndarray | torch.Tensor]:
    """Given a list of arrays or tensors, return the same list with the data padded to the largest in the set. Can be

    convenient for e.g. logging and tiling several images as with torchvision's `make_grid'`

    Args:
        data: Data.
        pad_value: Pad value.

    Returns:
        The result.
    """
    if not data:
        return data

    shapes = np.asarray([_.shape for _ in data])
    max_shape = shapes.max(axis=0)

    crop_start_per_shape = [-(max_shape - np.asarray(_)) // 2 for _ in shapes]
    crop_boxes = [_.tolist() + max_shape.tolist() for _ in crop_start_per_shape]

    return [crop_to_bbox(curr_data, bbox, pad_value=pad_value) for curr_data, bbox in zip(data, crop_boxes)]
