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

# pylint: disable=too-many-arguments

"""DIRECT Vision Transformer models for MRI reconstruction."""

from __future__ import annotations

import torch
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import reduce_operator
from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType, UFormerModel
from direct.nn.transformers.vit import VisionTransformer2D, VisionTransformer3D
from direct.types import FFTOperator

__all__ = [
    "ImageDomainMRIUFormer",
    "ImageDomainMRIViT2D",
    "ImageDomainMRIViT3D",
    "KSpaceDomainMRIViT2D",
    "KSpaceDomainMRIViT3D",
]


class ImageDomainMRIUFormer(nn.Module):
    """U-Former model for MRI reconstruction in the image domain.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        patch_size: Size of the patch. Default is ``256``.
        in_channels: Number of input channels. Default is ``2``.
        out_channels: Number of output channels. Default is ``None``.
        embedding_dim: Size of the feature embedding. Default is ``32``.
        encoder_depths: Number of layers for each stage of the encoder of the U-former, from top to bottom. Default is ``(``2``,
            ``2``, ``2``, ``2``)``.
        encoder_num_heads: Number of attention heads for each layer of the encoder of the U-former, from top to bottom.
            Default is ``(1, 2, 4, 8)``.
        bottleneck_depth: Default is ``16``.
        bottleneck_num_heads: Default is ``2``.
        win_size: Window size for the attention mechanism. Default is ``8``.
        mlp_ratio: Ratio of the hidden dimension size to the embedding dimension size in the MLP layers. Default is ``4.0``.
        qkv_bias: Whether to use bias in the query, key, and value projections of the attention mechanism. Default is
            ``True``.
        qk_scale: Scale factor for the query and key projection vectors. If set to ``None``, will use the default value of ``1`` /
            sqrt(embedding_dim). Default is ``None``.
        drop_rate: Dropout rate for the token-level dropout layer. Default is ``0.0``.
        attn_drop_rate: Dropout rate for the attention score matrix. Default is ``0.0``.
        drop_path_rate: Dropout rate for the stochastic depth regularization. Default is ``0.1``.
        patch_norm: Whether to use normalization for the patch embeddings. Default is ``True``.
        token_projection: Type of token projection. Must be one of [``"linear"``, ``"conv"``]. Default is
            :attr:`~direct.nn.transformers.uformer.AttentionTokenProjectionType.LINEAR`.
        token_mlp: Type of token-level MLP. Must be one of [``"leff"``, ``"mlp"``, ``"ffn"``]. Default is
            :attr:`~direct.nn.transformers.uformer.LeWinTransformerMLPTokenType.LEFF`.
        shift_flag: Whether to use shift operation in the local attention mechanism. Default is ``True``.
        modulator: Whether to use a modulator in the attention mechanism. Default is ``False``.
        cross_modulator: Whether to use cross-modulation in the attention mechanism. Default is ``False``.
        normalized: Whether to apply normalization before and denormalization after the forward pass. Default is ``True``.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        patch_size: int = 256,
        embedding_dim: int = 32,
        encoder_depths: tuple[int, ...] = (2, 2, 2, 2),
        encoder_num_heads: tuple[int, ...] = (1, 2, 4, 8),
        bottleneck_depth: int = 2,
        bottleneck_num_heads: int = 16,
        win_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
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

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            patch_size: Size of the patch. Default is ``256``.
            in_channels: Number of input channels. Default is ``2``.
            out_channels: Number of output channels. Default is ``None``.
            embedding_dim: Size of the feature embedding. Default is ``32``.
            encoder_depths: Number of layers for each stage of the encoder of the U-former, from top to bottom. Default is ``(``2``,
                ``2``, ``2``, ``2``)``.
            encoder_num_heads: Number of attention heads for each layer of the encoder of the U-former, from top to bottom.
                Default is ``(1, 2, 4, 8)``.
            bottleneck_depth: Default is ``16``.
            bottleneck_num_heads: Default is ``2``.
            win_size: Window size for the attention mechanism. Default is ``8``.
            mlp_ratio: Ratio of the hidden dimension size to the embedding dimension size in the MLP layers. Default is ``4.0``.
            qkv_bias: Whether to use bias in the query, key, and value projections of the attention mechanism. Default is
                ``True``.
            qk_scale: Scale factor for the query and key projection vectors. If set to ``None``, will use the default value of ``1`` /
                sqrt(embedding_dim). Default is ``None``.
            drop_rate: Dropout rate for the token-level dropout layer. Default is ``0.0``.
            attn_drop_rate: Dropout rate for the attention score matrix. Default is ``0.0``.
            drop_path_rate: Dropout rate for the stochastic depth regularization. Default is ``0.1``.
            patch_norm: Whether to use normalization for the patch embeddings. Default is ``True``.
            token_projection: Type of token projection. Must be one of [``"linear"``, ``"conv"``]. Default is
                :attr:`~direct.nn.transformers.uformer.AttentionTokenProjectionType.LINEAR`.
            token_mlp: Type of token-level MLP. Must be one of [``"leff"``, ``"mlp"``, ``"ffn"``]. Default is
                :attr:`~direct.nn.transformers.uformer.LeWinTransformerMLPTokenType.LEFF`.
            shift_flag: Whether to use shift operation in the local attention mechanism. Default is ``True``.
            modulator: Whether to use a modulator in the attention mechanism. Default is ``False``.
            cross_modulator: Whether to use cross-modulation in the attention mechanism. Default is ``False``.
            normalized: Whether to apply normalization before and denormalization after the forward pass. Default is ``True``.
            **kwargs: Other keyword arguments to pass to the parent constructor.

        Returns:
            ``None``.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
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
            Masked k-space of shape ``(N, coil, height, width, complex=2)``.
        sensitivity_map: torch.Tensor
            Sensitivity map of shape ``(N, coil, height, width, complex=2)``

        Args:
            masked_kspace: Masked kspace.
            sensitivity_map: Sensitivity map.

        Returns:
            The output tensor of shape ``(N, height, width, complex=2)``.
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

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        average_size: The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and (average_size, average_size, average_size) for 3D.
            Default is ``320``.
        patch_size: The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default is ``16``.
        embedding_dim: Dimension of the output embedding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
        qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
        qk_scale: The scale factor for the query-key dot product. Default is ``None``.
        drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
        attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
        dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
        use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
            ``True``.
        locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
        use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
        normalized: Whether to normalize the input tensor. Default is ``True``.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        average_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`ImageDomainMRIViT2D`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            average_size: The average size of the input image. If an int is provided, this will be determined by the
                `dimensionality`, i.e., (average_size, average_size) for 2D and (average_size, average_size, average_size) for 3D.
                Default is ``320``.
            patch_size: The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
                (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default is ``16``.
            embedding_dim: Dimension of the output embedding.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
            qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
            qk_scale: The scale factor for the query-key dot product. Default is ``None``.
            drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
            attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
            dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
            use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
                ``True``.
            locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
            use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
            normalized: Whether to normalize the input tensor. Default is ``True``.

        Returns:
            ``None``.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
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
            Masked k-space of shape ``(N, coil, height, width, complex=2)``.
        sensitivity_map: torch.Tensor
            Sensitivity map of shape ``(N, coil, height, width, complex=2)``

        Args:
            masked_kspace: Masked kspace.
            sensitivity_map: Sensitivity map.

        Returns:
            The output tensor of shape ``(N, height, width, complex=2)``.
        """
        image = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        ).permute(0, 3, 1, 2)
        out = self.transformer(image).permute(0, 2, 3, 1)
        return out


class ImageDomainMRIViT3D(nn.Module):
    """Vision Transformer for MRI reconstruction in 3D.

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        average_size: The average size of the input image. If an int is provided, this will be defined as (average_size,
            average_size, average_size). Default is ``320``.
        patch_size: The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size,
            patch_size). Default is ``16``.
        embedding_dim: Dimension of the output embedding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
        qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
        qk_scale: The scale factor for the query-key dot product. Default is ``None``.
        drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
        attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
        dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
        use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
            ``True``.
        locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
        use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
        normalized: Whether to normalize the input tensor. Default is ``True``.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        average_size: int | tuple[int, int, int] = 320,
        patch_size: int | tuple[int, int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`ImageDomainMRIViT3D`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            average_size: The average size of the input image. If an int is provided, this will be defined as (average_size,
                average_size, average_size). Default is ``320``.
            patch_size: The size of the patch. If an int is provided, this will be defined as (patch_size, patch_size,
                patch_size). Default is ``16``.
            embedding_dim: Dimension of the output embedding.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
            qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
            qk_scale: The scale factor for the query-key dot product. Default is ``None``.
            drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
            attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
            dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
            use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
                ``True``.
            locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
            use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
            normalized: Whether to normalize the input tensor. Default is ``True``.

        Returns:
            ``None``.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
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
            Masked k-space of shape ``(N, coil, slice/time, height, width, complex=2)``.
        sensitivity_map: torch.Tensor
            Sensitivity map of shape ``(N, coil, slice/time, height, width, complex=2)``

        Args:
            masked_kspace: Masked kspace.
            sensitivity_map: Sensitivity map.

        Returns:
            The output tensor of shape ``(N, slice/time, height, width, complex=2)``.
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

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        average_size: The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and (average_size, average_size, average_size) for 3D.
            Default is ``320``.
        patch_size: The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default is ``16``.
        embedding_dim: Dimension of the output embedding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
        qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
        qk_scale: The scale factor for the query-key dot product. Default is ``None``.
        drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
        attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
        dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
        use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
            ``True``.
        locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
        use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
        normalized: Whether to normalize the input tensor. Default is ``True``.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        average_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`KSpaceDomainMRIViT2D`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            average_size: The average size of the input image. If an int is provided, this will be determined by the
                `dimensionality`, i.e., (average_size, average_size) for 2D and (average_size, average_size, average_size) for 3D.
                Default is ``320``.
            patch_size: The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
                (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default is ``16``.
            embedding_dim: Dimension of the output embedding.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
            qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
            qk_scale: The scale factor for the query-key dot product. Default is ``None``.
            drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
            attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
            dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
            use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
                ``True``.
            locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
            use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
            normalized: Whether to normalize the input tensor. Default is ``True``.
            compute_per_coil: Whether to compute the output per coil.

        Returns:
            ``None``.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
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
            Masked k-space of shape ``(N, coil, height, width, complex=2)``.
        sensitivity_map: torch.Tensor
            Sensitivity map of shape ``(N, coil, height, width, complex=2)``
        sampling_mask: torch.Tensor
            Sampling mask of shape ``(N, 1, height, width, 1)``.

        Args:
            masked_kspace: Masked kspace.
            sensitivity_map: Sensitivity map.
            sampling_mask: Sampling mask.

        Returns:
            The output tensor of shape ``(N, height, width, complex=2)``.
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

        # Otherwise, create a single image from the coil data
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

    Args:
        forward_operator: Forward operator function.
        backward_operator: Backward operator function.
        average_size: The average size of the input image. If an int is provided, this will be determined by the
            `dimensionality`, i.e., (average_size, average_size) for 2D and (average_size, average_size, average_size) for 3D.
            Default is ``320``.
        patch_size: The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
            (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default is ``16``.
        embedding_dim: Dimension of the output embedding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
        qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
        qk_scale: The scale factor for the query-key dot product. Default is ``None``.
        drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
        attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
        dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
        use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
            ``True``.
        locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
        use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
        normalized: Whether to normalize the input tensor. Default is ``True``.
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        average_size: int | tuple[int, int, int] = 320,
        patch_size: int | tuple[int, int, int] = 16,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_gpsa: bool = True,
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ) -> None:
        """Inits :class:`KSpaceDomainMRIViT3D`.

        Args:
            forward_operator: Forward operator function.
            backward_operator: Backward operator function.
            average_size: The average size of the input image. If an int is provided, this will be determined by the
                `dimensionality`, i.e., (average_size, average_size) for 2D and (average_size, average_size, average_size) for 3D.
                Default is ``320``.
            patch_size: The size of the patch. If an int is provided, this will be determined by the `dimensionality`, i.e.,
                (patch_size, patch_size) for 2D and (patch_size, patch_size, patch_size) for 3D. Default is ``16``.
            embedding_dim: Dimension of the output embedding.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: The ratio of hidden dimension size to input dimension size in the MLP layer. Default is ``4.0``.
            qkv_bias: Whether to add bias to the query, key, and value projections. Default is ``False``.
            qk_scale: The scale factor for the query-key dot product. Default is ``None``.
            drop_rate: The dropout probability for all dropout layers except dropout_path. Default is ``0.0``.
            attn_drop_rate: The dropout probability for the attention layer. Default is ``0.0``.
            dropout_path_rate: The dropout probability for the dropout path. Default is ``0.0``.
            use_gpsa: Whether to use the GPSA attention layer. If set to ``False``, the MHSA layer will be used. Default is
                ``True``.
            locality_strength: The strength of the locality assumption in initialization. Default is ``1.0``.
            use_pos_embedding: Whether to use positional embeddings. Default is ``True``.
            normalized: Whether to normalize the input tensor. Default is ``True``.
            compute_per_coil: Whether to compute the output per coil.

        Returns:
            ``None``.
        """
        super().__init__()
        for extra_key in kwargs:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")
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
            Masked k-space of shape ``(N, coil, slice/time, height, width, complex=2)``.
        sensitivity_map: torch.Tensor
            Sensitivity map of shape ``(N, coil, slice/time, height, width, complex=2)``
        sampling_mask: torch.Tensor
            Sampling mask of shape ``(N, 1, 1 or slice/time, height, width, 1)``.

        Args:
            masked_kspace: Masked kspace.
            sensitivity_map: Sensitivity map.
            sampling_mask: Sampling mask.

        Returns:
            The output tensor of shape ``(N, slice/time height, width, complex=2)``.
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
