# Copyright (c) DIRECT Contributors

"""DIRECT Vision Transformer models for MRI reconstruction."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import reduce_operator
from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType, UFormerModel
from direct.nn.transformers.vit import VisionTransformer2D, VisionTransformer3D

__all__ = [
    "ImageDomainMRIUFormer",
    "ImageDomainMRIViT2D",
    "ImageDomainMRIViT3D",
    "KSpaceDomainMRIViT2D",
    "KSpaceDomainMRIViT3D",
]


class ImageDomainMRIUFormer(nn.Module):
    """U-Former model for MRI reconstruction in the image domain.

    Parameters
    ----------
    forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Forward operator function.
    backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
        Backward operator function.
    patch_size : int
        Size of the patch. Default: 256.
    in_channels : int
        Number of input channels. Default: 2.
    out_channels : int, optional
        Number of output channels. Default: None.
    embedding_dim : int
        Size of the feature embedding. Default: 32.
    encoder_depths : tuple
        Number of layers for each stage of the encoder of the U-former, from top to bottom. Default: (2, 2, 2, 2).
    encoder_num_heads : tuple
        Number of attention heads for each layer of the encoder of the U-former, from top to bottom.
        Default: (1, 2, 4, 8).
    bottleneck_depth : int
        Default: 16.
    bottleneck_num_heads : int
        Default: 2.
    win_size : int
        Window size for the attention mechanism. Default: 8.
    mlp_ratio : float
        Ratio of the hidden dimension size to the embedding dimension size in the MLP layers. Default: 4.0.
    qkv_bias : bool
        Whether to use bias in the query, key, and value projections of the attention mechanism. Default: True.
    qk_scale : float
        Scale factor for the query and key projection vectors.
        If set to None, will use the default value of 1 / sqrt(embedding_dim). Default: None.
    drop_rate : float
        Dropout rate for the token-level dropout layer. Default: 0.0.
    attn_drop_rate : float
        Dropout rate for the attention score matrix. Default: 0.0.
    drop_path_rate : float
        Dropout rate for the stochastic depth regularization. Default: 0.1.
    patch_norm : bool
        Whether to use normalization for the patch embeddings. Default: True.
    token_projection : AttentionTokenProjectionType
        Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.LINEAR.
    token_mlp : LeWinTransformerMLPTokenType
        Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.LEFF.
    shift_flag : bool
        Whether to use shift operation in the local attention mechanism. Default: True.
    modulator : bool
        Whether to use a modulator in the attention mechanism. Default: False.
    cross_modulator : bool
        Whether to use cross-modulation in the attention mechanism. Default: False.
    normalized : bool
        Whether to apply normalization before and denormalization after the forward pass. Default: True.
    """

    def __init__(
        self,
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor],
        patch_size: int = 256,
        embedding_dim: int = 32,
        encoder_depths: tuple[int, ...] = (2, 2, 2, 2),
        encoder_num_heads: tuple[int, ...] = (1, 2, 4, 8),
        bottleneck_depth: int = 2,
        bottleneck_num_heads: int = 16,
        win_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.LINEAR,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.LEFF,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
        normalized: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`ImageDomainMRIUFormer`.

        Parameters
        ----------
        forward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Forward operator function.
        backward_operator : Callable[[tuple[Any, ...]], torch.Tensor]
            Backward operator function.
        patch_size : int
            Size of the patch. Default: 256.
        in_channels : int
            Number of input channels. Default: 2.
        out_channels : int, optional
            Number of output channels. Default: None.
        embedding_dim : int
            Size of the feature embedding. Default: 32.
        encoder_depths : tuple
            Number of layers for each stage of the encoder of the U-former, from top to bottom. Default: (2, 2, 2, 2).
        encoder_num_heads : tuple
            Number of attention heads for each layer of the encoder of the U-former, from top to bottom.
            Default: (1, 2, 4, 8).
        bottleneck_depth : int
            Default: 16.
        bottleneck_num_heads : int
            Default: 2.
        win_size : int
            Window size for the attention mechanism. Default: 8.
        mlp_ratio : float
            Ratio of the hidden dimension size to the embedding dimension size in the MLP layers. Default: 4.0.
        qkv_bias : bool
            Whether to use bias in the query, key, and value projections of the attention mechanism. Default: True.
        qk_scale : float
            Scale factor for the query and key projection vectors.
            If set to None, will use the default value of 1 / sqrt(embedding_dim). Default: None.
        drop_rate : float
            Dropout rate for the token-level dropout layer. Default: 0.0.
        attn_drop_rate : float
            Dropout rate for the attention score matrix. Default: 0.0.
        drop_path_rate : float
            Dropout rate for the stochastic depth regularization. Default: 0.1.
        patch_norm : bool
            Whether to use normalization for the patch embeddings. Default: True.
        token_projection : AttentionTokenProjectionType
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.LINEAR.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.LEFF.
        shift_flag : bool
            Whether to use shift operation in the local attention mechanism. Default: True.
        modulator : bool
            Whether to use a modulator in the attention mechanism. Default: False.
        cross_modulator : bool
            Whether to use cross-modulation in the attention mechanism. Default: False.
        normalized : bool
            Whether to apply normalization before and denormalization after the forward pass. Default: True.
        **kwargs: Other keyword arguments to pass to the parent constructor.
        """
        super().__init__()
        self.uformer = UFormerModel(
            patch_size=patch_size,
            in_channels=COMPLEX_SIZE,
            embedding_dim=embedding_dim,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            bottleneck_depth=bottleneck_depth,
            bottleneck_num_heads=bottleneck_num_heads,
            win_size=win_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            modulator=modulator,
            cross_modulator=cross_modulator,
            normalized=normalized,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ImageDomainMRIUFormer`.

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

        out = self.uformer(image).permute(0, 2, 3, 1)

        return out


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
    use_gpsa : bool, optional
                Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
        use_gpsa: tuple[int, int] = (-1, -1),
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
        use_gpsa : bool, optional
            Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
            use_gpsa=use_gpsa,
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
    use_gpsa : bool, optional
        Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
        use_gpsa: tuple[int, int] = (-1, -1),
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
        use_gpsa : bool, optional
            Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
            use_gpsa=use_gpsa,
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
    use_gpsa : bool, optional
        Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
        use_gpsa: tuple[int, int] = (-1, -1),
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
        use_gpsa : bool, optional
            Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
            use_gpsa=use_gpsa,
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
    use_gpsa : bool, optional
        Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
        use_gpsa: tuple[int, int] = (-1, -1),
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
        use_gpsa : bool, optional
            Whether to use the GPSA attention layer. If set to False, the MHSA layer will be used. Default: True.
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
            use_gpsa=use_gpsa,
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
