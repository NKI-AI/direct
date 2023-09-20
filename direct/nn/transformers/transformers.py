# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from direct.data.transforms import apply_mask, apply_padding, expand_operator, reduce_operator
from direct.nn.transformers.uformer import *
from direct.nn.transformers.utils import norm, pad, pad_to_square, unnorm, unpad
from direct.nn.transformers.vision_transformers import VisionTransformer
from direct.types import DirectEnum

__all__ = [
    "MRITransformer",
    "ImageDomainVisionTransformer",
    "ImageDomainUFormer",
    "KSpaceDomainUFormerMultiCoilInputMode",
    "KSpaceDomainUFormer",
    "VariationalUFormer",
]


class VariationalUFormerBlock(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
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
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.leff,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
        normalized: bool = True,
    ):
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.uformer = UFormerModel(
            patch_size=patch_size,
            in_channels=2,
            out_channels=2,
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
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`VariationalUFormerBlock`.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        learning_rate : torch.Tensor
            (Trainable) Learning rate parameter of shape (1,).

        Returns
        -------
        torch.Tensor
            Next k-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_error = apply_mask(current_kspace - masked_kspace, sampling_mask, return_mask=False)

        regularization_term = reduce_operator(
            self.backward_operator(current_kspace, dim=self._spatial_dims), sensitivity_map, dim=self._coil_dim
        ).permute(0, 3, 1, 2)

        regularization_term = self.uformer(regularization_term).permute(0, 2, 3, 1)

        regularization_term = self.forward_operator(
            expand_operator(regularization_term, sensitivity_map, dim=self._coil_dim), dim=self._spatial_dims
        )

        return current_kspace - learning_rate * kspace_error + regularization_term


class VariationalUFormer(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps: int = 8,
        no_weight_sharing: bool = True,
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
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.leff,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.blocks = nn.ModuleList(
            [
                VariationalUFormerBlock(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    patch_size=patch_size,
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
                )
                for _ in range((num_steps if no_weight_sharing else 1))
            ]
        )
        self.lr = nn.Parameter(torch.tensor([1.0] * num_steps))
        self.num_steps = num_steps
        self.no_weight_sharing = no_weight_sharing

        self.padding_factor = win_size * (2 ** len(encoder_depths))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`VariationalUFormer`.

        Parameters
        ----------
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map : torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        padding : torch.Tensor, optional
            Padding of shape (N, 1, height, width, 1). Default: None.

        Returns
        -------
        torch.Tensor
            k-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_prediction = masked_kspace.clone()
        for step_idx in range(self.num_steps):
            kspace_prediction = self.blocks[step_idx if self.no_weight_sharing else 0](
                kspace_prediction, masked_kspace, sampling_mask, sensitivity_map, self.lr[step_idx]
            )
        kspace_prediction = masked_kspace + apply_mask(kspace_prediction, ~sampling_mask, return_mask=False)
        if padding is not None:
            kspace_prediction = apply_padding(kspace_prediction, padding)
        return kspace_prediction


class MRIUFormer(nn.Module):
    """A PyTorch module that implements MRI image reconstruction using an image domain UFormer."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
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
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.leff,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
        **kwargs,
    ):
        """Inits :class:`MRIUFormer`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        patch_size : int
            Size of the patch. Default: 256.
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
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.linear.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.leff.
        shift_flag : bool
            Whether to use shift operation in the local attention mechanism. Default: True.
        modulator : bool
            Whether to use a modulator in the attention mechanism. Default: False.
        cross_modulator : bool
            Whether to use cross-modulation in the attention mechanism. Default: False.
        **kwargs: Other keyword arguments to pass to the parent constructor.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.uformer = UFormer(
            patch_size=patch_size,
            in_channels=2,
            out_channels=2,
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
        )
        self.padding_factor = win_size * (2 ** len(encoder_depths))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)


class KSpaceDomainUFormerMultiCoilInputMode(DirectEnum):
    sense_sum = "sense_sum"
    compute_per_coil = "compute_per_coil"


class KSpaceDomainUFormer(MRIUFormer):
    """A PyTorch module that implements MRI image reconstruction using a k-space domain UFormer."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        multicoil_input_mode: KSpaceDomainUFormerMultiCoilInputMode = KSpaceDomainUFormerMultiCoilInputMode.sense_sum,
        patch_size: int = 128,
        embedding_dim: int = 16,
        encoder_depths: tuple[int, ...] = (2, 2, 2),
        encoder_num_heads: tuple[int, ...] = (1, 2, 4),
        bottleneck_depth: int = 2,
        bottleneck_num_heads: int = 8,
        win_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.leff,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
        **kwargs,
    ):
        """Inits :class:`KSpaceDomainUFormer`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        multicoil_input_mode: KSpaceDomainUFormerMultiCoilInputMode
            Set to "sense_sum" to aggregate all coil data, or "compute_per_coil" to pass each coil data in
            a different pass to the same model. Default: KSpaceDomainUFormerMultiCoilInputMode.sense_sum.
        patch_size : int
            Size of the patch. Default: 128.
        embedding_dim : int
            Size of the feature embedding. Default: 16.
        encoder_depths : tuple
            Number of layers for each stage of the encoder of the U-former, from top to bottom. Default: (2, 2, 2).
        encoder_num_heads : tuple
            Number of attention heads for each layer of the encoder of the U-former, from top to bottom.
            Default: (1, 2, 4).
        bottleneck_depth : int
            Default: 8.
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
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.linear.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.leff.
        shift_flag : bool
            Whether to use shift operation in the local attention mechanism. Default: True.
        modulator : bool
            Whether to use a modulator in the attention mechanism. Default: False.
        cross_modulator : bool
            Whether to use cross-modulation in the attention mechanism. Default: False.
        **kwargs: Other keyword arguments to pass to the parent constructor.
        """
        super().__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            patch_size=patch_size,
            in_channels=2,
            out_channels=2,
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
        )

        self.multicoil_input_mode = multicoil_input_mode

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Performs forward pass of :class:`KSpaceDomainUFormer`.

        Parameters
        ----------
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor, optional
            Sampling mask of shape (N, 1, height, width, 1). If not None, it will use
            :math:`y_{inp} + (1-M) * f_\theta(y_{inp})` as output, else :math:`f_\theta(y_{inp})`.
        padding : torch.Tensor, optional
            Zero-padding (i.e. not sampled locations) that may be present in original k-space
            of shape (N, 1, height, width, 1). If not None, padding will be applied to output.

        Returns
        -------
        out : torch.Tensor
            Prediction of output image of shape (N, height, width, complex=2).
        """

        # Pad to square in image domain
        inp = self.backward_operator(masked_kspace, dim=self._spatial_dims).permute(0, 1, 4, 2, 3)
        inp, _, wpad, hpad = pad_to_square(inp, self.padding_factor)
        padded_sensitivity_map, _, _, _ = pad_to_square(sensitivity_map.permute(0, 1, 4, 2, 3), self.padding_factor)
        padded_sensitivity_map = padded_sensitivity_map.permute(0, 1, 3, 4, 2)

        # Project back to k-space
        inp = self.forward_operator(inp.permute(0, 1, 3, 4, 2).contiguous(), dim=self._spatial_dims)
        if self.multicoil_input_mode == "sense_sum":
            # Construct SENSE reconstruction
            # \sum_{k=1}^{n_c} S^k * \mathcal{F}^{-1} (y^k)
            inp = reduce_operator(
                coil_data=self.backward_operator(inp, dim=self._spatial_dims),
                sensitivity_map=padded_sensitivity_map,
                dim=self._coil_dim,
            )
            # Project the SENSE reconstruction to k-space domain and use as input to model
            inp = self.forward_operator(inp, dim=[d - 1 for d in self._spatial_dims])
            inp = inp.permute(0, 3, 1, 2)

            inp, mean, std = norm(inp)

            out = self.uformer(inp)

            out = unnorm(out, mean, std)

            # Project k-space to image domain and unpad
            out = self.backward_operator(out.permute(0, 2, 3, 1), dim=[d - 1 for d in self._spatial_dims])
        else:
            # Pass each coil k-space to model
            out = []
            for coil_idx in range(masked_kspace.shape[self._coil_dim]):
                coil_data = inp[:, coil_idx].permute(0, 3, 1, 2)

                coil_data, mean, std = norm(coil_data)

                coil_data = self.uformer(coil_data)

                coil_data = unnorm(coil_data, mean, std).permute(0, 2, 3, 1)

                out.append(coil_data)
            out = torch.stack(out, dim=self._coil_dim)

            out = reduce_operator(
                coil_data=self.backward_operator(out, dim=self._spatial_dims),
                sensitivity_map=padded_sensitivity_map,
                dim=self._coil_dim,
            )
        out = unpad(out.permute(0, 3, 1, 2), wpad, hpad).permute(0, 2, 3, 1)

        out = self.forward_operator(expand_operator(out, sensitivity_map, self._coil_dim), dim=self._spatial_dims)
        if sampling_mask is not None:
            out = masked_kspace + apply_mask(out, ~sampling_mask, return_mask=False)
        if padding is not None:
            out = apply_padding(out, padding)
        return out


class ImageDomainUFormer(MRIUFormer):
    """A PyTorch module that implements MRI image reconstruction using an image domain UFormer."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
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
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.leff,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
        **kwargs,
    ):
        """Inits :class:`ImageDomainUFormer`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        patch_size : int
            Size of the patch. Default: 256.
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
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.linear.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.leff.
        shift_flag : bool
            Whether to use shift operation in the local attention mechanism. Default: True.
        modulator : bool
            Whether to use a modulator in the attention mechanism. Default: False.
        cross_modulator : bool
            Whether to use cross-modulation in the attention mechanism. Default: False.
        **kwargs: Other keyword arguments to pass to the parent constructor.
        """
        super().__init__(
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            patch_size=patch_size,
            in_channels=2,
            out_channels=2,
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
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs forward pass of :class:`ImageDomainUFormer`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor, optional
            Sampling mask of shape (N, 1, height, width, 1). If not None, it will use
            :math:`y_{inp} + (1-M) * f_\theta(y_{inp})` as output, else :math:`f_\theta(y_{inp})`.
        padding : torch.Tensor, optional
            Zero-padding (i.e. not sampled locations) that may be present in original k-space
            of shape (N, 1, height, width, 1). If not None, padding will be applied to output.

        Returns
        -------
        out : torch.Tensor
            Prediction of output image of shape (N, height, width, complex=2).
        """
        inp = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        )
        inp = inp.permute(0, 3, 1, 2)

        inp, padding_mask, wpad, hpad = pad_to_square(inp, factor=self.padding_factor)
        inp, mean, std = norm(inp)

        out = self.uformer(inp, padding_mask)

        out = unnorm(out, mean, std)
        out = unpad(out, wpad, hpad).permute(0, 2, 3, 1)

        out = self.forward_operator(
            expand_operator(out, sensitivity_map, dim=self._coil_dim),
            dim=self._spatial_dims,
        )

        if sampling_mask is not None:
            out = masked_kspace + apply_mask(out, ~sampling_mask, return_mask=False)
        if padding is not None:
            out = apply_padding(out, padding)

        return out


class MRITransformer(nn.Module):
    """A PyTorch module that implements MRI image reconstruction using VisionTransformer."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_gradient_descent_steps: int,
        average_img_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 10,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        **kwargs,
    ):
        """Inits :class:`MRITransformer`.

        Parameters
        ----------
        forward_operator : Callable
            Forward operator function.
        backward_operator : Callable
            Backward operator function.
        num_gradient_descent_steps : int
            Number of gradient descent steps to perform.
        average_img_size : int or tuple[int, int], optional
            Size to which the input image is rescaled before processing.
        patch_size : int or tuple[int, int], optional
            Patch size used in VisionTransformer.
        embedding_dim : int, optional
            The number of embedding dimensions in the VisionTransformer.
        depth : int, optional
            The number of layers in the VisionTransformer.
        num_heads : int, optional
            The number of attention heads in the VisionTransformer.
        mlp_ratio : float, optional
            The ratio of MLP hidden size to embedding size in the VisionTransformer.
        qkv_bias : bool, optional
            Whether to include bias terms in the projection matrices in the VisionTransformer.
        qk_scale : float, optional
            Scale factor for query and key in the attention calculation in the VisionTransformer.
        drop_rate : float, optional
            Dropout probability for the VisionTransformer.
        attn_drop_rate : float, optional
            Dropout probability for the attention layer in the VisionTransformer.
        dropout_path_rate : float, optional
            Dropout probability for the intermediate skip connections in the VisionTransformer.
        norm_layer : nn.Module, optional
            Normalization layer used in the VisionTransformer.
        gpsa_interval : tuple[int, int], optional
            Interval for performing Generalized Positional Self-Attention (GPSA) in the VisionTransformer.
        locality_strength : float, optional
            The strength of locality in the GPSA in the VisionTransformer.
        use_pos_embedding : bool, optional
            Whether to use positional embedding in the VisionTransformer.
        """
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                VisionTransformer(
                    average_img_size=average_img_size,
                    patch_size=patch_size,
                    in_channels=2,
                    out_channels=2,
                    embedding_dim=embedding_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    dropout_path_rate=dropout_path_rate,
                    norm_layer=norm_layer,
                    gpsa_interval=gpsa_interval,
                    locality_strength=locality_strength,
                    use_pos_embedding=use_pos_embedding,
                )
                for _ in range(num_gradient_descent_steps)
            ]
        )
        self.learning_rate = nn.Parameter(torch.ones(num_gradient_descent_steps))
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_gradient_descent_steps = num_gradient_descent_steps

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def _forward_operator(
        self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        forward = apply_mask(
            self.forward_operator(expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
            sampling_mask,
            return_mask=False,
        )
        return forward

    def _backward_operator(
        self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        backward = reduce_operator(
            self.backward_operator(apply_mask(kspace, sampling_mask, return_mask=False), self._spatial_dims),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(
        self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Performs forward pass of :class:`ImageDomainVisionTransformer`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        out : torch.Tensor
            Prediction of output image of shape (N, height, width, complex=2).
        """
        x = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        )
        for _ in range(self.num_gradient_descent_steps):
            x_trans, wpad, hpad = pad(x.permute(0, 3, 1, 2), self.transformers[0].patch_size)
            x_trans, mean, std = norm(x_trans)

            x_trans = x_trans + self.transformers[_](x_trans)

            x_trans = unnorm(x_trans, mean, std)
            x_trans = unpad(x_trans, wpad, hpad).permute(0, 2, 3, 1)

            x = x - self.learning_rate[_] * (
                self._backward_operator(
                    self._forward_operator(x, sampling_mask, sensitivity_map) - masked_kspace,
                    sampling_mask,
                    sensitivity_map,
                )
                + x_trans
            )

        return x


class ImageDomainVisionTransformer(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        use_mask: bool = True,
        average_img_size: int | tuple[int, int] = 320,
        patch_size: int | tuple[int, int] = 10,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        gpsa_interval: tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tranformer = VisionTransformer(
            average_img_size=average_img_size,
            patch_size=patch_size,
            in_channels=4 if use_mask else 2,
            out_channels=2,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            gpsa_interval=gpsa_interval,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.use_mask = use_mask

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Performs forward pass of :class:`ImageDomainVisionTransformer`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        out : torch.Tensor
            Prediction of output image of shape (N, height, width, complex=2).
        """
        inp = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        )

        if self.use_mask and sampling_mask is not None:
            sampling_mask_inp = torch.cat(
                [
                    sampling_mask,
                    torch.zeros(*sampling_mask.shape, device=sampling_mask.device),
                ],
                dim=self._complex_dim,
            ).to(inp.dtype)
            # project it in image domain
            sampling_mask_inp = self.backward_operator(sampling_mask_inp, dim=self._spatial_dims).squeeze(
                self._coil_dim
            )
            inp = torch.cat([inp, sampling_mask_inp], dim=self._complex_dim)

        inp = inp.permute(0, 3, 1, 2)

        inp, wpad, hpad = pad(inp, self.transformer.patch_size)
        inp, mean, std = norm(inp)

        out = self.transformer(inp)

        out = unnorm(out, mean, std)
        out = unpad(out, wpad, hpad)

        return out.permute(0, 2, 3, 1)
