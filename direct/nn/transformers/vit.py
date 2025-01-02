# Copyright (c) DIRECT Contributors

"""DIRECT Vision Transformer module.

Implementation of Vision Transformer model [1, 2]_ in PyTorch.

Code borrowed from [3]_ which uses code from timm [4]_.

References
----------
.. [1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, 
    M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An Image is Worth 16x16 Words: 
    Transformers for Image Recognition at Scale, http://arxiv.org/abs/2010.11929, (2021).
.. [2] Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., Beyer, L.: How to train your ViT? Data, 
    Augmentation, and Regularization in Vision Transformers, http://arxiv.org/abs/2106.10270, (2022).
.. [3] https://github.com/facebookresearch/convit
.. [4] https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from direct.constants import COMPLEX_SIZE
from direct.nn.transformers.utils import DropoutPath, init_weights, norm, pad_to_divisible, unnorm, unpad_to_original
from direct.types import DirectEnum

__all__ = ["VisionTransformer2D", "VisionTransformer3D"]


class VisionTransformerDimensionality(DirectEnum):

    TWO_DIMENSIONAL = "2D"
    THREE_DIMENSIONAL = "3D"


class MLP(nn.Module):
    """MLP layer with dropout and activation for Vision Transformer.

    Parameters
    ----------
    in_features : int
        Size of the input feature.
    hidden_features : int, optional
        Size of the hidden layer feature. If None, then hidden_features = in_features. Default: None.
    out_features : int, optional
        Size of the output feature. If None, then out_features = in_features. Default: None.
    act_layer : nn.Module, optional
        Activation layer to be used. Default: nn.GELU.
    drop : float, optional
        Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Inits :class:`MLP`.

        Parameters
        ----------
        in_features : int
            Size of the input feature.
        hidden_features : int, optional
            Size of the hidden layer feature. If None, then hidden_features = in_features. Default: None.
        out_features : int, optional
            Size of the output feature. If None, then out_features = in_features. Default: None.
        act_layer : nn.Module, optional
            Activation layer to be used. Default: nn.GELU.
        drop : float, optional
            Dropout probability. Default: 0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MLP`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the network.

        Returns
        -------
        torch.Tensor
            Output tensor of the network.

        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GPSA(nn.Module):
    """Gated Positional Self-Attention module for Vision Transformer.

    Parameters
    ----------
    dimensionality : VisionTransformerDimensionality
        The dimensionality of the input data.
    dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True, include bias terms in the query, key, and value projections.
    qk_scale : float
        Scale factor for query and key.
    attn_drop : float
        Dropout probability for attention weights.
    proj_drop : float
        Dropout probability for output tensor.
    locality_strength : float
        Strength of locality assumption in initialization.
    use_local_init : bool
        If True, use the locality-based initialization.
    grid_size : tuple[int,int], optional
        The size of the grid (height, width) for relative position encoding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        locality_strength: float = 1.0,
        use_local_init: bool = True,
        grid_size=None,
    ) -> None:
        """Inits :class:`GPSA`.

        Parameters
        ----------
        dim : int
            Dimensionality of the input embeddings.
        num_heads : int
            Number of attention heads.
        qkv_bias : bool
            If True, include bias terms in the query, key, and value projections.
        qk_scale : float
            Scale factor for query and key.
        attn_drop : float
            Dropout probability for attention weights.
        proj_drop : float
            Dropout probability for output tensor.
        locality_strength : float
            Strength of locality assumption in initialization.
        use_local_init : bool
            If True, use the locality-based initialization.
        grid_size : tuple[int,int], optional
            The size of the grid (height, width) for relative position encoding.
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)
        self.current_grid_size = grid_size

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the attention scores for each patch in x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Attention scores for each patch in x.
        """
        B, N, C = x.shape

        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pos_score = self.pos_proj(self.get_rel_indices()).expand(B, -1, -1, -1).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn = attn / attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    @abstractmethod
    def local_init(self, locality_strength: Optional[float] = 1.0) -> None:
        pass

    @abstractmethod
    def get_rel_indices(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`GPSA`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor:
        """
        B, N, C = x.shape

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GPSA2D(GPSA):
    """Gated Positional Self-Attention module for Vision Transformer.

    Parameters
    ----------
    dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True, include bias terms in the query, key, and value projections.
    qk_scale : float
        Scale factor for query and key.
    attn_drop : float
        Dropout probability for attention weights.
    proj_drop : float
        Dropout probability for output tensor.
    locality_strength : float
        Strength of locality assumption in initialization.
    use_local_init : bool
        If True, use the locality-based initialization.
    grid_size : tuple[int,int], optional
        The size of the grid (height, width) for relative position encoding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        locality_strength: float = 1.0,
        use_local_init: bool = True,
        grid_size=None,
    ) -> None:
        """Inits :class:`GPSA`.

        Parameters
        ----------
        dim : int
            Dimensionality of the input embeddings.
        num_heads : int
            Number of attention heads.
        qkv_bias : bool
            If True, include bias terms in the query, key, and value projections.
        qk_scale : float
            Scale factor for query and key.
        attn_drop : float
            Dropout probability for attention weights.
        proj_drop : float
            Dropout probability for output tensor.
        locality_strength : float
            Strength of locality assumption in initialization.
        use_local_init : bool
            If True, use the locality-based initialization.
        grid_size : tuple[int,int], optional
            The size of the grid (height, width) for relative position encoding.
        """
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            locality_strength=locality_strength,
            use_local_init=use_local_init,
            grid_size=grid_size,
        )

    def local_init(self, locality_strength: Optional[float] = 1.0) -> None:
        """Initializes the parameters for a locally connected attention mechanism.

        Parameters
        ----------
        locality_strength : float, optional
            A scalar multiplier for the locality distance. Default: 1.0.

        Returns
        -------
        None
        """
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads**0.5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2

        # compute the positional projection weights with locality distance
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self) -> None:
        """Generates relative positional indices for each patch in the input.

        Returns
        -------
        None
        """
        H, W = self.current_grid_size
        N = H * W
        rel_indices = torch.zeros(1, N, N, 3)
        indx = torch.arange(W).view(1, -1) - torch.arange(W).view(-1, 1)
        indx = indx.repeat(H, H)
        indy = torch.arange(H).view(1, -1) - torch.arange(H).view(-1, 1)
        indy = indy.repeat_interleave(W, dim=0).repeat_interleave(W, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)

        return rel_indices.to(self.v.weight.device)


class GPSA3D(GPSA):
    """Gated Positional Self-Attention module for Vision Transformer (3D variant).

    Parameters
    ----------
    dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True, include bias terms in the query, key, and value projections.
    qk_scale : float
        Scale factor for query and key.
    attn_drop : float
        Dropout probability for attention weights.
    proj_drop : float
        Dropout probability for output tensor.
    locality_strength : float
        Strength of locality assumption in initialization.
    use_local_init : bool
        If True, use the locality-based initialization.
    grid_size : tuple[int, int, int], optional
        The size of the grid (depth, height, width) for relative position encoding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        locality_strength: float = 1.0,
        use_local_init: bool = True,
        grid_size=None,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            locality_strength=locality_strength,
            use_local_init=use_local_init,
            grid_size=grid_size,
        )

    def local_init(self, locality_strength: Optional[float] = 1.0) -> None:
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1

        kernel_size = int(self.num_heads ** (1 / 3))
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2

        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                for h3 in range(kernel_size):
                    position = h1 + kernel_size * (h2 + kernel_size * h3)
                    self.pos_proj.weight.data[position, 2] = -1
                    self.pos_proj.weight.data[position, 1] = 2 * (h2 - center) * locality_distance
                    self.pos_proj.weight.data[position, 0] = 2 * (h3 - center) * locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self) -> torch.Tensor:
        D, H, W = self.current_grid_size
        N = D * H * W
        rel_indices = torch.zeros(1, N, N, 3)

        indz = torch.arange(D).view(1, -1) - torch.arange(D).view(-1, 1)
        indz = indz.repeat(H * W, H * W)

        indx = torch.arange(W).view(1, -1) - torch.arange(W).view(-1, 1)
        indx = indx.repeat(D * H, D * H)

        indy = torch.arange(H).view(1, -1) - torch.arange(H).view(-1, 1)
        indy = indy.repeat(D * W, D * W)

        indd = indz**2 + indx**2 + indy**2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)

        return rel_indices.to(self.v.weight.device)


class MHSA(nn.Module):
    """Multi-Head Self-Attention (MHSA) module.

    Parameters
    ----------
    dim : int
        Number of input features.
    num_heads : int
        Number of heads in the attention mechanism. Default is 8.
    qkv_bias : bool
        If True, bias is added to the query, key and value projections. Default is False.
    qk_scale : float or None
        Scaling factor for the query-key dot product. If None, it is set to
        head_dim ** -0.5 where head_dim = dim // num_heads. Default is None.
    attn_drop : float
        Dropout rate for the attention weights. Default is 0.
    proj_drop : float
        Dropout rate for the output of the module. Default is 0.
    grid_size : tuple[int, int] or None
        If not None, the module is designed to work with a grid of
        patches. grid_size is a tuple of the form (H, W) where H and W are the number of patches in
        the vertical and horizontal directions respectively. Default is None.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        grid_size: tuple[int, int] = None,
    ) -> None:
        """Inits :class:`MHSA`.

        Parameters
        ----------
        dim : int
            Number of input features.
        num_heads : int
            Number of heads in the attention mechanism. Default is 8.
        qkv_bias : bool
            If True, bias is added to the query, key and value projections. Default is False.
        qk_scale : float or None
            Scaling factor for the query-key dot product. If None, it is set to
            head_dim ** -0.5 where head_dim = dim // num_heads. Default is None.
        attn_drop : float
            Dropout rate for the attention weights. Default is 0.
        proj_drop : float
            Dropout rate for the output of the module. Default is 0.
        grid_size : tuple[int, int] or None
            If not None, the module is designed to work with a grid of
            patches. grid_size is a tuple of the form (H, W) where H and W are the number of patches in
            the vertical and horizontal directions respectively. Default is None.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(init_weights)
        self.current_grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MHSA`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, C).
        """

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class VisionTransformerBlock(nn.Module):
    """A single transformer block used in the VisionTransformer model.

    Parameters
    ----------
    dimensionality : VisionTransformerDimensionality
        The dimensionality of the input data.
    dim : int
        The feature dimension.
    num_heads : int
        The number of attention heads.
    mlp_ratio : float, optional
        The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
    qkv_bias : bool, optional
        Whether to add bias to the query, key, and value projections. Default: False.
    qk_scale : float, optional
        The scale factor for the query-key dot product. Default: None.
    drop : float, optional
        The dropout probability for all dropout layers except dropout_path. Default: 0.0.
    attn_drop : float, optional
        The dropout probability for the attention layer. Default: 0.0.
    dropout_path : float, optional
        The dropout probability for the dropout path. Default: 0.0.
    act_layer : nn.Module, optional
        The activation layer used in the MLP. Default: nn.GELU.
    norm_layer : nn.Module, optional
        The normalization layer used in the block. Default: nn.LayerNorm.
    use_gpsa : bool, optional
        Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
    **kwargs: Additional arguments for the attention layer.
    """

    def __init__(
        self,
        dimensionality: VisionTransformerDimensionality,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        dropout_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_gpsa: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`VisionTransformerBlock`.

        Parameters
        ----------
        dimensionality : VisionTransformerDimensionality
            The dimensionality of the input data.
        dim : int
            The feature dimension.
        num_heads : int
            The number of attention heads.
        mlp_ratio : float, optional
            The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
        qkv_bias : bool, optional
            Whether to add bias to the query, key, and value projections. Default: False.
        qk_scale : float, optional
            The scale factor for the query-key dot product. Default: None.
        drop : float, optional
            The dropout probability for all dropout layers except dropout_path. Default: 0.0.
        attn_drop : float, optional
            The dropout probability for the attention layer. Default: 0.0.
        dropout_path : float, optional
            The dropout probability for the dropout path. Default: 0.0.
        act_layer : nn.Module, optional
            The activation layer used in the MLP. Default: nn.GELU.
        norm_layer : nn.Module, optional
            The normalization layer used in the block. Default: nn.LayerNorm.
        use_gpsa : bool, optional
            Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
        **kwargs: Additional arguments for the attention layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = (GPSA2D if dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL else GPSA3D)(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                **kwargs,
            )
        else:
            self.attn = MHSA(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                **kwargs,
            )
        self.dropout_path = DropoutPath(dropout_path) if dropout_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass for the :class:`VisionTransformerBlock`.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        grid_size : tuple[int, int]
            The size of the grid used by the attention layer.

        Returns
        -------
        torch.Tensor: The output tensor.
        """
        self.attn.current_grid_size = grid_size
        x = x + self.dropout_path(self.attn(self.norm1(x)))
        x = x + self.dropout_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, patch_size, in_channels, embedding_dim, dimensionality: VisionTransformerDimensionality
    ) -> None:
        """Inits :class:`PatchEmbedding` module for Vision Transformer.

        Parameters
        ----------
        patch_size : int or tuple[int, int]
            The patch size. If an int is provided, the patch will be a square.
        in_channels : int
            Number of input channels.
        embedding_dim : int
            Dimension of the output embedding.
        dimensionality : VisionTransformerDimensionality
            The dimensionality of the input data.
        """
        super().__init__()
        self.proj = (nn.Conv2d if dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL else nn.Conv3d)(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`PatchEmbedding`.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Patch embedding.
        """
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model.

    Parameters
    ----------
    dimensionality : VisionTransformerDimensionality
        The dimensionality of the input data.
    average_img_size : int or tuple[int, int] or tuple[int, int, int]
        The average size of the input image. If an int is provided, this will be determined by the
        `dimensionality`, i.e., (average_img_size, average_img_size) for 2D and
        (average_img_size, average_img_size, average_img_size) for 3D. Default: 320.
    patch_size : int or tuple[int, int] or tuple[int, int, int]
        The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
        (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
    in_channels : int
        Number of input channels. Default: COMPLEX_SIZE.
    out_channels : int or None
        Number of output channels. If None, this will be set to `in_channels`. Default: None.
    embedding_dim : int
        Dimension of the output embedding.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
    qkv_bias : bool
        Whether to add bias to the query, key, and value projections. Default: False.
    qk_scale : float
        The scale factor for the query-key dot product. Default: None.
    drop_rate : float
        The dropout probability for all dropout layers except dropout_path. Default: 0.0.
    attn_drop_rate : float
        The dropout probability for the attention layer. Default: 0.0.
    dropout_path_rate : float
        The dropout probability for the dropout path. Default: 0.0.
    use_gpsa: bool
        Whether to use GPSA layer. Default: True.
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        dimensionality: VisionTransformerDimensionality,
        average_img_size: int | tuple[int, int] | tuple[int, int, int] = 320,
        patch_size: int | tuple[int, int] | tuple[int, int, int] = 16,
        in_channels: int = COMPLEX_SIZE,
        out_channels: int = None,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
    ) -> None:
        """Inits :class:`VisionTransformer`.

        Parameters
        ----------
        dimensionality : VisionTransformerDimensionality
            The dimensionality of the input data.
        average_img_size : int or tuple[int, int] or tuple[int, int, int]
            The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_img_size, average_img_size) for 2D and
            (average_img_size, average_img_size, average_img_size) for 3D. Default: 320.
        patch_size : int or tuple[int, int] or tuple[int, int, int]
            The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
        in_channels : int
            Number of input channels. Default: COMPLEX_SIZE.
        out_channels : int or None
            Number of output channels. If None, this will be set to `in_channels`. Default: None.
        embedding_dim : int
            Dimension of the output embedding.
        depth : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads.
        mlp_ratio : float
            The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
        qkv_bias : bool
            Whether to add bias to the query, key, and value projections. Default: False.
        qk_scale : float
            The scale factor for the query-key dot product. Default: None.
        drop_rate : float
            The dropout probability for all dropout layers except dropout_path. Default: 0.0.
        attn_drop_rate : float
            The dropout probability for the attention layer. Default: 0.0.
        dropout_path_rate : float
            The dropout probability for the dropout path. Default: 0.0.
        use_gpsa: bool
            Whether to use GPSA layer. Default: True.
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        """
        super().__init__()

        self.dimensionality = dimensionality

        self.depth = depth
        embedding_dim *= num_heads
        self.num_features = embedding_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embedding = use_pos_embedding

        if isinstance(average_img_size, int):
            if self.dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL:
                img_size = (average_img_size, average_img_size)
            else:
                img_size = (average_img_size, average_img_size, average_img_size)
        else:
            if len(average_img_size) != (
                2 if self.dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL else 3
            ):
                raise ValueError(
                    f"average_img_size should have length 2 for 2D and 3 for 3D, got {len(average_img_size)}."
                )
            img_size = average_img_size

        if isinstance(patch_size, int):
            if self.dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL:
                self.patch_size = (patch_size, patch_size)
            else:
                self.patch_size = (patch_size, patch_size, patch_size)
        else:
            if len(patch_size) != (2 if self.dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL else 3):
                raise ValueError(f"patch_size should have length 2 for 2D and 3 for 3D, got {len(patch_size)}.")
            self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            dimensionality=dimensionality,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embedding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embedding_dim, *[img_size[i] // self.patch_size[i] for i in range(len(img_size))])
            )

            init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, dropout_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    dimensionality=dimensionality,
                    dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    dropout_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    use_gpsa=use_gpsa,
                    **({"locality_strength": locality_strength} if use_gpsa else {}),
                )
                for i in range(depth)
            ]
        )

        self.normalized = normalized

        self.norm = nn.LayerNorm(embedding_dim)
        # head
        self.feature_info = [dict(num_chs=embedding_dim, reduction=0, module="head")]
        self.head = nn.Linear(self.num_features, self.out_channels * np.prod(self.patch_size))

        self.head.apply(init_weights)

    def get_head(self) -> nn.Module:
        """Returns the head of the model.

        Returns
        -------
        nn.Module
        """
        return self.head

    def reset_head(self) -> None:
        """Resets the head of the model."""
        self.head = nn.Linear(self.num_features, self.out_channels * np.prod(self.patch_size))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feature extraction part of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
        """
        x = self.patch_embed(x)
        size = x.shape[2:]

        if self.use_pos_embedding:
            pos_embed = F.interpolate(
                self.pos_embed,
                size=size,
                mode=(
                    "bilinear"
                    if self.dimensionality == VisionTransformerDimensionality.TWO_DIMENSIONAL
                    else "trilinear"
                ),
                align_corners=False,
            )
            x = x + pos_embed

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for _, block in enumerate(self.blocks):
            x = block(x, size)

        x = self.norm(x)

        return x

    @abstractmethod
    def seq2img(self, x: torch.Tensor, img_size: tuple[int, ...]) -> torch.Tensor:
        """Converts the sequence patches tensor to an image tensor.

        Parameters
        ----------
        x : torch.Tensor
            The sequence tensor.
        img_size : tuple[int, ...]
            The size of the image tensor.

        Returns
        -------
        torch.Tensor
            The image tensor.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`VisionTransformer`.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        x, pads = pad_to_divisible(x, self.patch_size)

        size = x.shape[2:]

        if self.normalized:
            x, mean, std = norm(x)

        x = self.forward_features(x)
        x = self.head(x)
        x = self.seq2img(x, size)

        if self.normalized:
            x = unnorm(x, mean, std)

        x = unpad_to_original(x, *pads)

        return x


class VisionTransformer2D(VisionTransformer):
    """Vision Transformer model for 2D data.

    Parameters
    ----------
    average_img_size : int or tuple[int, int]
        The average size of the input image. If an int is provided, this will be determined by the
        `dimensionality`, i.e., (average_img_size, average_img_size) for 2D and
        (average_img_size, average_img_size, average_img_size) for 3D. Default: 320.
    patch_size : int or tuple[int, int]
        The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
        (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
    in_channels : int
        Number of input channels. Default: COMPLEX_SIZE.
    out_channels : int or None
        Number of output channels. If None, this will be set to `in_channels`. Default: None.
    embedding_dim : int
        Dimension of the output embedding.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
    qkv_bias : bool
        Whether to add bias to the query, key, and value projections. Default: False.
    qk_scale : float
        The scale factor for the query-key dot product. Default: None.
    drop_rate : float
        The dropout probability for all dropout layers except dropout_path. Default: 0.0.
    attn_drop_rate : float
        The dropout probability for the attention layer. Default: 0.0.
    dropout_path_rate : float
        The dropout probability for the dropout path. Default: 0.0.
    use_gpsa: bool
        Whether to use GPSA layer. Default: True.
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        average_img_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = COMPLEX_SIZE,
        out_channels: int = None,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
    ) -> None:
        """Inits :class:`VisionTransformer2D`.

        Parameters
        ----------
        average_img_size : int or tuple[int, int]
            The average size of the input image. If an int is provided, this will be defined as
            (average_img_size, average_img_size). Default: 320.
        patch_size : int or tuple[int, int]
            The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size). Default: 16.
        in_channels : int
            Number of input channels. Default: COMPLEX_SIZE.
        out_channels : int or None
            Number of output channels. If None, this will be set to `in_channels`. Default: None.
        embedding_dim : int
            Dimension of the output embedding.
        depth : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads.
        mlp_ratio : float
            The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
        qkv_bias : bool
            Whether to add bias to the query, key, and value projections. Default: False.
        qk_scale : float
            The scale factor for the query-key dot product. Default: None.
        drop_rate : float
            The dropout probability for all dropout layers except dropout_path. Default: 0.0.
        attn_drop_rate : float
            The dropout probability for the attention layer. Default: 0.0.
        dropout_path_rate : float
            The dropout probability for the dropout path. Default: 0.0.
        use_gpsa: bool
            Whether to use GPSA layer. Default: True.
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        """
        super().__init__(
            dimensionality=VisionTransformerDimensionality.TWO_DIMENSIONAL,
            average_img_size=average_img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_gpsa=use_gpsa,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=normalized,
        )

    def seq2img(self, x: torch.Tensor, img_size: tuple[int, ...]) -> torch.Tensor:
        """Converts the sequence patches tensor to an image tensor.

        Parameters
        ----------
        x : torch.Tensor
            The sequence tensor.
        img_size : tuple[int, ...]
            The size of the image tensor.

        Returns
        -------
        torch.Tensor
            The image tensor.
        """
        x = x.view(x.shape[0], x.shape[1], self.out_channels, self.patch_size[0], self.patch_size[1])
        x = x.chunk(x.shape[1], dim=1)
        x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3)
        x = x.chunk(img_size[0] // self.patch_size[0], dim=3)
        x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3).squeeze(1)

        return x


class VisionTransformer3D(VisionTransformer):
    """Vision Transformer model for 3D data.

    Parameters
    ----------
    average_img_size : int or tuple[int, int, int]
        The average size of the input image. If an int is provided, this will be defined as
        (average_img_size, average_img_size, average_img_size). Default: 320.
    patch_size : int or tuple[int, int, int]
        The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size, patch_size).
        Default: 16.
    in_channels : int
        Number of input channels. Default: COMPLEX_SIZE.
    out_channels : int or None
        Number of output channels. If None, this will be set to `in_channels`. Default: None.
    embedding_dim : int
        Dimension of the output embedding.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
    qkv_bias : bool
        Whether to add bias to the query, key, and value projections. Default: False.
    qk_scale : float
        The scale factor for the query-key dot product. Default: None.
    drop_rate : float
        The dropout probability for all dropout layers except dropout_path. Default: 0.0.
    attn_drop_rate : float
        The dropout probability for the attention layer. Default: 0.0.
    dropout_path_rate : float
        The dropout probability for the dropout path. Default: 0.0.
    use_gpsa: bool
        Whether to use GPSA layer. Default: True.
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        average_img_size: int | tuple[int, int, int] = 320,
        patch_size: int | tuple[int, int, int] = 16,
        in_channels: int = COMPLEX_SIZE,
        out_channels: int = None,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
    ) -> None:
        """Inits :class:`VisionTransformer3D`.

        Parameters
        ----------
        average_img_size : int or tuple[int, int, int]
            The average size of the input image. If an int is provided, this will be defined as
            (average_img_size, average_img_size, average_img_size). Default: 320.
        patch_size : int or tuple[int, int, int]
            The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size, patch_size).
            Default: 16.
        in_channels : int
            Number of input channels. Default: COMPLEX_SIZE.
        out_channels : int or None
            Number of output channels. If None, this will be set to `in_channels`. Default: None.
        embedding_dim : int
            Dimension of the output embedding.
        depth : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads.
        mlp_ratio : float
            The ratio of hidden dimension size to input dimension size in the MLP layer. Default: 4.0.
        qkv_bias : bool
            Whether to add bias to the query, key, and value projections. Default: False.
        qk_scale : float
            The scale factor for the query-key dot product. Default: None.
        drop_rate : float
            The dropout probability for all dropout layers except dropout_path. Default: 0.0.
        attn_drop_rate : float
            The dropout probability for the attention layer. Default: 0.0.
        dropout_path_rate : float
            The dropout probability for the dropout path. Default: 0.0.
        use_gpsa: bool
            Whether to use GPSA layer. Default: True.
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        """

        super().__init__(
            dimensionality=VisionTransformerDimensionality.THREE_DIMENSIONAL,
            average_img_size=average_img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_gpsa=use_gpsa,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=normalized,
        )

    def seq2img(self, x: torch.Tensor, img_size: tuple[int, ...]) -> torch.Tensor:
        """Converts the sequence of 3D patches to a 3D image tensor.

        Parameters
        ----------
        x : torch.Tensor
            The sequence tensor, where each entry corresponds to a flattened 3D patch.
        img_size : tuple of ints
            The size of the 3D image tensor (depth, height, width).

        Returns
        -------
        torch.Tensor
            The reconstructed 3D image tensor.
        """
        # Reshape the sequence into patches of shape (batch, num_patches, out_channels, D, H, W)
        x = x.view(
            x.shape[0], x.shape[1], self.out_channels, self.patch_size[0], self.patch_size[1], self.patch_size[2]
        )

        # Chunk along the sequence dimension (depth, height, width)
        depth_chunks = img_size[0] // self.patch_size[0]  # Number of chunks along depth
        height_chunks = img_size[1] // self.patch_size[1]  # Number of chunks along height
        width_chunks = img_size[2] // self.patch_size[2]  # Number of chunks along width

        # First, chunk along the sequence dimension (width axis)
        x = torch.cat(x.chunk(width_chunks, dim=1), dim=5).permute(0, 1, 2, 3, 4, 5)

        # Now, chunk along the height axis
        x = torch.cat(x.chunk(height_chunks, dim=1), dim=4).permute(0, 1, 2, 3, 4, 5)

        # Finally, chunk along the depth axis
        x = torch.cat(x.chunk(depth_chunks, dim=1), dim=3).permute(0, 1, 2, 3, 4, 5).squeeze(1)

        return x
