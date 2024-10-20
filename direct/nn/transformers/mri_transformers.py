# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import reduce_operator
from direct.nn.transformers.vit import VisionTransformer2D, VisionTransformer3D

__all__ = ["ImageDomainMRIViT2D", "ImageDomainMRIViT3D", "KSpaceDomainMRIViT2D", "KSpaceDomainMRIViT3D"]


class ImageDomainMRIViT2D(nn.Module):
    """Vision Transformer for MRI reconstruction in 2D.

    Parameters
    ----------
    forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Forward operator function.
    backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Backward operator function.
    average_size : int or tuple[int, int]
        The average size of the input image. If an int is provided, this will be determined by the
        `dimensionality`, i.e., (average_size, average_size) for 2D and
        (average_size, average_size, average_size) for 3D. Default: 320.
    patch_size : int or tuple[int, int]
        The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
        (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
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
    gpsa_interval : tuple[int, int]
        The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        average_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`ImageDomainMRIViT2D`.

        Parameters
        ----------
        forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Forward operator function.
        backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Backward operator function.
        average_size : int or tuple[int, int]
            The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and
            (average_size, average_size, average_size) for 3D. Default: 320.
        patch_size : int or tuple[int, int]
            The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
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
        gpsa_interval : tuple[int, int]
            The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        """
        super().__init__()
        self.transformer = VisionTransformer2D(
            average_img_size=average_size,
            patch_size=patch_size,
            in_channels=COMPLEX_SIZE,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            gpsa_interval=gpsa_interval,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=normalized,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ImageDomainMRIViT2D`.

        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2)

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (N, height, width, complex=2).
        """
        image = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        ).permute(0, 3, 1, 2)
        out = self.transformer(image).permute(0, 2, 3, 1)
        return out


class ImageDomainMRIViT3D(VisionTransformer3D):
    """Vision Transformer for MRI reconstruction in 3D.

    Parameters
    ----------
    forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Forward operator function.
    backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Backward operator function.
    average_size : int or tuple[int, int, int]
        The average size of the input image. If an int is provided, this will be defined as
        (average_size, average_size, average_size). Default: 320.
    patch_size : int or tuple[int, int, int]
        The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size, patch_size).
        Default: 16.
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
    gpsa_interval : tuple[int, int]
        The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        average_size: int | tuple[int, int, int] = 320,
        patch_size: int | tuple[int, int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`ImageDomainMRIViT3D`.

        Parameters
        ----------
        forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Forward operator function.
        backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Backward operator function.
        average_size : int or tuple[int, int, int]
            The average size of the input image. If an int is provided, this will be defined as
            (average_size, average_size, average_size). Default: 320.
        patch_size : int or tuple[int, int, int]
            The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size, patch_size).
            Default: 16.
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
        gpsa_interval : tuple[int, int]
            The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        """
        super().__init__()
        self.transformer = VisionTransformer3D(
            average_img_size=average_size,
            patch_size=patch_size,
            in_channels=COMPLEX_SIZE,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            gpsa_interval=gpsa_interval,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=normalized,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (3, 4)

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ImageDomainMRIViT3D`.

        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, slice/time, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, slice/time, height, width, complex=2)

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (N, slice/time, height, width, complex=2).
        """

        image = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        ).permute(0, 4, 1, 2, 3)
        out = self.transformer(image).permute(0, 2, 3, 4, 1)
        return out


class KSpaceDomainMRIViT2D(nn.Module):
    """Vision Transformer for MRI reconstruction in 2D in k-space.

    Parameters
    ----------
    forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Forward operator function.
    backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Backward operator function.
    average_size : int or tuple[int, int]
        The average size of the input image. If an int is provided, this will be determined by the
        `dimensionality`, i.e., (average_size, average_size) for 2D and
        (average_size, average_size, average_size) for 3D. Default: 320.
    patch_size : int or tuple[int, int]
        The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
        (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
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
    gpsa_interval : tuple[int, int]
        The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        average_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`KSpaceDomainMRIViT2D`.

        Parameters
        ----------
        forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Forward operator function.
        backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Backward operator function.
        average_size : int or tuple[int, int]
            The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and
            (average_size, average_size, average_size) for 3D. Default: 320.
        patch_size : int or tuple[int, int]
            The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
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
        gpsa_interval : tuple[int, int]
            The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        compute_per_coil : bool
            Whether to compute the output per coil.
        """
        super().__init__()
        self.transformer = VisionTransformer2D(
            average_img_size=average_size,
            patch_size=patch_size,
            in_channels=COMPLEX_SIZE,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            gpsa_interval=gpsa_interval,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=normalized,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.compute_per_coil = compute_per_coil

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of :class:`KSpaceDomainMRIViT2D`.

        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2)
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (N, height, width, complex=2).
        """
        if self.compute_per_coil:
            out = torch.stack(
                [
                    self.transformer(masked_kspace[:, i].permute(0, 3, 1, 2))
                    for i in range(masked_kspace.shape[self._coil_dim])
                ],
                dim=self._coil_dim,
            ).permute(0, 1, 3, 4, 2)

            out = torch.where(sampling_mask, masked_kspace, out)  # data consistency

            # Create a single image from the coil data and return it
            out = reduce_operator(
                coil_data=self.backward_operator(out, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
            return out
        else:
            # Create a single image from the coil data
            sense_image = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
            # Trasnform the image to the k-space domain
            inp = self.forward_operator(sense_image, dim=[d - 1 for d in self._spatial_dims])

            # Pass to the transformer
            out = self.transformer(inp.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()

            out = self.backward_operator(out, dim=[d - 1 for d in self._spatial_dims])
            return out


class KSpaceDomainMRIViT3D(nn.Module):
    """Vision Transformer for MRI reconstruction in 3D in k-space.

    Parameters
    ----------
    forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Forward operator function.
    backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Backward operator function.
    average_size : int or tuple[int, int]
        The average size of the input image. If an int is provided, this will be determined by the
        `dimensionality`, i.e., (average_size, average_size) for 2D and
        (average_size, average_size, average_size) for 3D. Default: 320.
    patch_size : int or tuple[int, int]
        The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
        (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
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
    gpsa_interval : tuple[int, int]
        The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
    locality_strength : float
        The strength of the locality assumption in initialization. Default: 1.0.
    use_pos_embedding : bool
        Whether to use positional embeddings. Default: True.
    normalized : bool
        Whether to normalize the input tensor. Default: True.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        average_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`KSpaceDomainMRIViT3D`.

        Parameters
        ----------
        forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Forward operator function.
        backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Backward operator function.
        average_size : int or tuple[int, int]
            The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and
            (average_size, average_size, average_size) for 3D. Default: 320.
        patch_size : int or tuple[int, int]
            The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default: 16.
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
        gpsa_interval : tuple[int, int]
            The interval of the blocks where the GPSA layer is used. Default: (-1, -1).
        locality_strength : float
            The strength of the locality assumption in initialization. Default: 1.0.
        use_pos_embedding : bool
            Whether to use positional embeddings. Default: True.
        normalized : bool
            Whether to normalize the input tensor. Default: True.
        compute_per_coil : bool
            Whether to compute the output per coil.
        """
        super().__init__()
        self.transformer = VisionTransformer3D(
            average_img_size=average_size,
            patch_size=patch_size,
            in_channels=COMPLEX_SIZE,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            gpsa_interval=gpsa_interval,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
            normalized=normalized,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.compute_per_coil = compute_per_coil

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (3, 4)

    def forward(
        self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of :class:`KSpaceDomainMRIViT3D`.

        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, slice/time, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, slice/time, height, width, complex=2)
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, 1 or slice/time, height, width, 1).

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (N, slice/time height, width, complex=2).
        """
        if self.compute_per_coil:
            out = torch.stack(
                [
                    self.transformer(masked_kspace[:, i].permute(0, 4, 1, 2, 3))
                    for i in range(masked_kspace.shape[self._coil_dim])
                ],
                dim=self._coil_dim,
            ).permute(0, 1, 3, 4, 5, 2)

            out = torch.where(sampling_mask, masked_kspace, out)  # data consistency

            # Create a single image from the coil data and return it
            out = reduce_operator(
                coil_data=self.backward_operator(out, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
            return out
        else:
            # Create a single image from the coil data
            sense_image = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
            # Trasnform the image to the k-space domain
            inp = self.forward_operator(sense_image, dim=[d - 1 for d in self._spatial_dims])

            # Pass to the transformer
            out = self.transformer(inp.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1).contiguous()

            out = self.backward_operator(out, dim=[d - 1 for d in self._spatial_dims])
            return out
