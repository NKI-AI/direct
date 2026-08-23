# Copyright (c) DIRECT Contributors

"""Warping functions (for PyTorch tensors) for image registration."""

import torch
import torch.nn.functional as F


def create_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
    r"""Creates a grid of coordinates for a given shape.

    Args:
        shape: Shape of the grid to create. Must be (batch_size, C, \*) for ND tensors, where \* is the spatial dimensions
            of length N.
        device: Device to create the grid on.

    Returns:
        Grid tensor of shape (batch_size, N, \*) where \* is the spatial dimensions of the input shape.
    """
    batch_size, _, *spatial_dims = shape

    # Create a mesh grid representing the coordinates
    mesh = torch.meshgrid(*[torch.arange(0, dim) for dim in spatial_dims], indexing="ij")
    # Add a batch dimension and a channel dimension
    mesh = [vec.unsqueeze(0).unsqueeze(0) for vec in mesh]
    # Repeat the mesh grid to match the batch size
    repeat_shape = [batch_size, 1] + [1] * len(spatial_dims)
    mesh = [vec.repeat(*repeat_shape) for vec in mesh]
    grid = torch.cat(mesh[::-1], 1)

    return grid.float().to(device)


def normalize_vector_field(vector: torch.Tensor) -> torch.Tensor:
    r"""Normalizes a vector field to the range [-1, 1] for a given shape.

    Args:
        vector: Input ND vector field tensor of shape (batch_size, C, \*) where \* is the spatial dimensions of length N.

    Returns:
        Normalized vector field tensor with the same shape as the input vector field.
    """
    spatial_dims = vector.shape[2:]
    # Normalize vector field to the range [-1, 1]
    for index, dim in enumerate(spatial_dims[::-1]):
        vector[:, index, ...] = 2.0 * vector[:, index, ...] / max(dim - 1, 1) - 1.0
    return vector


def warp_tensor(x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    r"""Applies a vector-based warping transformation to an input tensor.

    This is also known as spatial transformer networks [1]. Supports both ND tensors.

    Args:
        x: Input tensor of shape (batch_size, C, \*) where \* is the spatial dimensions of length N.
        vector: Flow field / inverse coordinate map tensor of shape (batch_size, N, \*), where N is the number of spatial
            dimensions.

    Returns:
        Warped tensor with the same shape as the input tensor.

    References:
        .. [#] Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks."
            Advances in neural information processing systems 28 (2015).
    """

    if ((x.shape[0],) + x.shape[2:]) != ((vector.shape[0],) + vector.shape[2:]):
        raise ValueError(
            f"Expected the input tensor to have the same spatial dimensions as the vector field. "
            f"Instead, received shapes {x.shape} and {vector.shape} for the input tensor and vector field, respectively."
        )

    grid = create_grid(x.shape, x.device)

    # Add vector to the grid
    vgrid = grid + vector

    # Normalize grid values to the range [-1, 1]
    vgrid = normalize_vector_field(vgrid)

    # Move the channels dimension to the last position
    vgrid = vgrid.permute(0, *range(2, vgrid.ndim), 1)

    # Warp the input tensor using the sampling grid
    output = F.grid_sample(x, vgrid, mode="bilinear", padding_mode="border", align_corners=True)

    # Create a mask of ones with the same shape as the input tensor
    mask = torch.ones_like(x)

    # Warp the mask using the same grid to identify valid sampling locations
    mask = F.grid_sample(mask, vgrid, mode="bilinear", padding_mode="border", align_corners=True)

    # Threshold the mask to create a binary mask
    # This is necessary to ignore invalid regions in the output
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    # Apply the mask to the warped output to ignore invalid regions
    return output * mask


def integrate_vector_field(vector: torch.Tensor, num_steps: int) -> torch.Tensor:
    r"""Integrates a vector field using scaling and squaring.

    Args:
        vector: Flow tensor of shape (batch_size, N, \*), where N is the number of spatial dimensions.
        num_steps: Number of integration steps to perform.

    Returns:
        Integrated vector field tensor with the same shape as the input vector field.
    """

    scale = 1.0 / (2**num_steps)
    vector = vector * scale

    # Integrate vector field using scaling and squaring
    for _ in range(num_steps):
        vector = vector + warp_tensor(vector, vector)
    return vector


def warp(image: torch.Tensor, vector: torch.Tensor, num_integration_steps: int = 1) -> torch.Tensor:
    r"""Applies a vector-based warping transformation to an input image.

    If `num_steps` is set to 0, the vector field is used directly for warping. Otherwise, the vector field is integrated
    using scaling and squaring.

    Args:
        image: Input tensor of shape (batch_size, C, \*) where \* is the spatial dimensions of length N.
        vector: Flow field / inverse coordinate map tensor of shape (batch_size, N, \*), where N is the number of spatial
            dimensions.
        num_integration_steps: Number of integration steps to perform. If set to 0, the vector field is used directly for
            warping. Default is ``1``.

    Returns:
        Warped image with the same shape as the input image.
    """

    if num_integration_steps > 0:
        vector = integrate_vector_field(vector, num_integration_steps)

    return warp_tensor(image, vector)
