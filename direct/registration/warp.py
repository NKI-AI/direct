# Copyright (c) DIRECT Contributors

"""Warping functions (for PyTorch tensors) for image registration."""

import torch
import torch.nn.functional as F


def warp_tensor(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Applies a flow-based warping transformation to an input tensor.

    Supports both 2D and 3D tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.
    flow : torch.Tensor
        Flow field / inverse coordinate map tensor of shape (B, 2, H, W) for 2D or (B, 3, D, H, W) for 3D.
        First channel is the horizontal flow and the second channel is the vertical flow for 2D
        and the third channel is the depth flow for 3D.

    Returns
    -------
    torch.Tensor
        Warped tensor with the same shape as the input tensor.
    """
    batch_size, _, *spatial_dims = x.size()

    # Check if input is 2D or 3D
    if len(spatial_dims) == 2:
        H, W = spatial_dims
        D = None
    elif len(spatial_dims) == 3:
        D, H, W = spatial_dims
    else:
        raise ValueError("Input tensor must be 4D (B, C, H, W) or 5D (B, C, D, H, W)")

    # Create a mesh grid representing the coordinates
    if D is None:
        yy, xx = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)
    else:
        zz, yy, xx = torch.meshgrid(torch.arange(0, D), torch.arange(0, H), torch.arange(0, W), indexing="ij")
        zz = zz.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        grid = torch.cat((zz, yy, xx), 1)

    grid = grid.float().to(x.device)

    # Add flow to the grid
    vgrid = grid + flow

    # Normalize grid values to the range [-1, 1]
    if D is None:
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
    else:
        vgrid[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :, :].clone() / max(D - 1, 1) - 1.0
        vgrid[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 4, 1)

    # Warp the input tensor using the sampling grid
    output = F.grid_sample(x, vgrid, mode="bilinear", padding_mode="border", align_corners=True)

    # Create a mask of ones with the same shape as the input tensor
    mask = torch.ones_like(x)

    # Warp the mask using the same grid to identify valid sampling locations
    mask = F.grid_sample(mask, vgrid, mode="bilinear", padding_mode="border", align_corners=True)

    # Threshold the mask to create a binary mask
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    # Apply the mask to the warped output to ignore invalid regions
    return output * mask
