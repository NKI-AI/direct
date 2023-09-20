# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_

from direct.nn.transformers.utils import DropoutPath, init_weights, norm, pad_to_square, unnorm, unpad
from direct.types import DirectEnum

__all__ = ["AttentionTokenProjectionType", "LeWinTransformerMLPTokenType", "UFormer", "UFormerModel"]


class ECALayer1d(nn.Module):
    """Efficient Channel Attention (ECA) module for 1D data."""

    def __init__(self, channel: int, k_size: int = 3):
        """Inits :class:`ECALayer1d`.

        Parameters
        ----------
        channel : int
            Number of channels of the input feature map.
        k_size : int
            Adaptive selection of kernel size. Default: 3.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the output of the ECA layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        y : torch.Tensor
            Output of the ECA layer.
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self) -> int:
        """Computes the number of floating point operations in :class:`ECA`.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class SepConv2d(torch.nn.Module):
    """A 2D Separable Convolutional layer.

    Applies a depthwise convolution followed by a pointwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        act_layer: nn.Module = nn.ReLU,
    ):
        """Inits :class:`SepConv2d`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple of ints
            Size of the convolution kernel.
        stride : int or tuple of ints
            Stride of the convolution. Default: 1.
        padding : int or tuple of ints
            Padding added to all four sides of the input. Default: 0.
        dilation : int or tuple of ints
            Spacing between kernel elements. Default: 1.
        act_layer : torch.nn.Module
            Activation layer applied after depthwise convolution. Default: nn.ReLU.
        """
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SepConv2d`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying depthwise and pointwise convolutions with activation.
        """
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW: int) -> int:
        """Calculate the number of floating point operations in :class:`SepConv2d`.

        Parameters
        ----------
        HW : int
            Size of the spatial dimension of the input tensor.

        Returns
        -------
        int : Number of floating point operations.
        """
        flops = 0
        flops += HW * self.in_channels * self.kernel_size**2 / self.stride**2
        flops += HW * self.in_channels * self.out_channels
        return int(flops)


######## Embedding for q,k,v ########
class ConvProjectionModule(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        kernel_size: int = 3,
        q_stride: int = 1,
        k_stride: int = 1,
        v_stride: int = 1,
        bias: bool = True,
    ):
        """Inits :class:`ConvProjectionModule`.

        Parameters
        ----------
        dim : int
            Number of channels in the input tensor.
        heads : int
            Number of heads in multi-head attention. Default: 8.
        dim_head : int
            Dimension of each head. Default: 64.
        kernel_size : int
            Size of convolutional kernel. Default: 3.
        q_stride : int
            Stride of the convolutional kernel for queries. Default: 1.
        k_stride : int
            Stride of the convolutional kernel for keys. Default: 1.
        v_stride : int
            Stride of the convolutional kernel for values. Default: 1.
        bias : bool
            Whether to include a bias term. Default: True.
        """
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(
        self, x: torch.Tensor, attn_kv: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`ConvProjectionModule`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        attn_kv : torch.Tensor, optional
            Attention key/value tensor. Default None.

        Returns
        -------
        q : torch.Tensor
            Query tensor.
        k : torch.Tensor
            Key tensor.
        v : torch.Tensor
            Value tensor.
        """
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, "b (l w) c -> b c l w", l=l, w=w)
        attn_kv = rearrange(attn_kv, "b (l w) c -> b c l w", l=l, w=w)
        q = self.to_q(x)
        q = rearrange(q, "b (h d) l w -> b h (l w) d", h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, "b (h d) l w -> b h (l w) d", h=h)
        v = rearrange(v, "b (h d) l w -> b h (l w) d", h=h)
        return q, k, v

    def flops(self, q_L: int, kv_L: Optional[int] = None) -> int:
        """Calculate the number of floating point operations in :class:`ConvProjectionModule`.

        Parameters
        ----------
        q_L : int
            Size of input patches.
        kv_L : int, optional
            Size of key/value patches. Default None.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjectionModule(nn.Module):
    """Linear projection layer used in the window attention mechanism of the Transformer model."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, bias: bool = True):
        """Inits :class:LinearProjectionModule`.

        Parameters
        ----------
        dim : int
            The input feature dimension.
        heads : int
            The number of heads in the multi-head attention mechanism. Default: 8.
        dim_head : int, optional
            The feature dimension of each head. Default: 64.
        bias : bool, optional
            Whether to use bias in the linear projections. Default: True.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(
        self, x: torch.Tensor, attn_kv: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass of :class:`LinearProjectionModule`.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_length, dim)
            The input tensor.
        attn_kv : torch.Tensor of shape (batch_size, seq_length, dim), optional
            The tensor to be used for computing the attention scores. If None, the input tensor is used. Default: None.

        Returns
        -------
        q : torch.Tensor of shape (batch_size, seq_length, heads, dim_head)
            The tensor resulting from the linear projection of x used for computing the queries.
        k : torch.Tensor of shape (batch_size, seq_length, heads, dim_head)
            The tensor resulting from the linear projection of attn_kv used for computing the keys.
        v : torch.Tensor of shape (batch_size, seq_length, heads, dim_head)
            The tensor resulting from the linear projection of attn_kv used for computing the values.

        """
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, q_L: int, kv_L: Optional[int] = None) -> int:
        """Calculate the number of floating point operations in :class:`LinearProjectionModule`.

        Parameters
        ----------
        q_L : int
            Size of input patches.
        kv_L : int, optional
            Size of key/value patches. Default None.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops


########### window-based self-attention #############
class AttentionTokenProjectionType(DirectEnum):
    conv = "conv"
    linear = "linear"


class WindowAttentionModule(nn.Module):
    """A window-based multi-head self-attention module."""

    def __init__(
        self,
        dim: int,
        win_size: tuple[int, int],
        num_heads: int,
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """Inits :class:`WindowAttentionModule`.

        Parameters
        ----------
        dim : int
            Input feature dimension.
        win_size : tuple[int, int]
            The window size (height and width).
        num_heads : int
            Number of heads for multi-head self-attention.
        token_projection : AttentionTokenProjectionType
            Type of projection for token-level queries, keys, and values. Either "conv" or "linear".
        qkv_bias : bool
            Whether to use bias in the linear projection layer for queries, keys, and values.
        qk_scale : float
            Scale factor for query and key.
        attn_drop : float
            Dropout rate for attention weights.
        proj_drop : float
            Dropout rate for the output of the last linear projection layer.

        """
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        if token_projection == "conv":
            self.qkv = ConvProjectionModule(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == "linear":
            self.qkv = LinearProjectionModule(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, attn_kv: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Performs forward pass of :class:`WindowAttentionModule`.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape `(B, N, C)` representing the input features, where `B` is the batch size, `N` is the
            sequence length, and `C` is the input feature dimension.
        attn_kv : torch.Tensor, optional
            An optional tensor of shape `(B, N, C)` representing the key-value pairs used for attention computation.
            If `None`, the key-value pairs are computed from `x` itself. Default: None.
        mask : torch.Tensor, optional
            An optional tensor of shape representing the binary mask for the input sequence.
            If `None`, no masking is applied. Default: None.

        Returns
        -------
        torch.Tensor
            A tensor of shape `(B, N, C)` representing the output features after attention computation.
        """
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, "nH l c -> nH l (c d)", d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, "nW m n -> nW m (n d)", d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}"

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`LinearProjectionModule` for 1 window
        with token length of N.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N

        flops += self.qkv.flops(H * W, H * W)

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        flops += nW * N * self.dim * self.dim
        return int(flops)


########### self-attention #############
class AttentionModule(nn.Module):
    """Self-attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Inits :class:`AttentionModule`.

        Parameters
        ----------
        dim : int
            The input feature dimension.
        num_heads : int
            The number of attention heads.
        qkv_bias : bool
            Whether to include biases in the query, key, and value projections. Default: True.
        qk_scale : float, optional
            Scaling factor for the query and key projections. Default: None.
        attn_drop : float
            Dropout probability for the attention weights. Default: 0.0.
        proj_drop : float
            Dropout probability for the output of the attention module. Default: 0.0.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = LinearProjectionModule(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, attn_kv: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`AttentionModule`.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        attn_kv : torch.Tensor, optional
            The attention key/value tensor.
        mask : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"

    def flops(self, q_num: int, kv_num: int) -> int:
        """Calculate the number of floating point operations in :class:`LinearProjectionModule`.

        Parameters
        ----------
        q_num : int
            Size of input patches.
        kv_num : int, optional
            Size of key/value patches. Default None.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0

        flops += self.qkv.flops(q_num, kv_num)

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        flops += q_num * self.dim * self.dim
        return flops


#########################################
########### feed-forward network #############
class MLP(nn.Module):
    """Multi-layer perceptron with optional dropout regularization."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        """Inits :class:`MLP`.

        Parameters:
        -----------
        in_features : int
            Number of input features.
        hidden_features : int, optional
            Number of output features in the hidden layer. If not specified, `in_features` is used.
        out_features : int, optional
            Number of output features. If not specified, `in_features` is used.
        act_layer : nn.Module
            Activation layer. Default: GeLU.
        drop : float
            Dropout probability. Default: 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the :class:`MLP`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`MLP`.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # fc1
        flops += H * W * self.in_features * self.hidden_features
        # fc2
        flops += H * W * self.hidden_features * self.out_features
        return flops


class LeFF(nn.Module):
    """Locally-enhanced Feed-Forward Network module."""

    def __init__(self, dim: int = 32, hidden_dim: int = 128, act_layer: nn.Module = nn.GELU, use_eca: bool = False):
        """Inits :class:`LeFF`.

        Parameters
        ----------
        dim : int
            Dimension of the input and output features. Default: 32.
        hidden_dim : int
            Dimension of the hidden features. Default: 128.
        act_layer : nn.Module
            Activation layer to apply after the first linear layer and the depthwise convolution. Default: GELU.
        use_eca : bool
            If True, adds a 1D ECA layer after the second linear layer. Default: False.
        """
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = ECALayer1d(dim) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`LeFF`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, " b (h w) (c) -> b c h w ", h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flatten
        x = rearrange(x, " b c h w -> b (h w) c", h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`LeFF`.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        # eca
        if hasattr(self.eca, "flops"):
            flops += self.eca.flops()
        return flops


#########################################
########### window operation#############
def window_partition(x: torch.Tensor, win_size: int, dilation_rate: int = 1) -> torch.Tensor:
    """Partition the input tensor into windows of specified size.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be partitioned into windows.
    win_size : int
        The size of the square windows to partition the tensor into.
    dilation_rate : int
        The dilation rate for convolution. Default: 1.

    Returns
    -------
    windows : torch.Tensor
        The tensor representing windows partitioned from input tensor.
    """
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, "dilation_rate should be a int"
        x = F.unfold(
            x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1), stride=win_size
        )  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows: torch.Tensor, win_size: int, H: int, W: int, dilation_rate: int = 1) -> torch.Tensor:
    """Rearrange the partitioned tensor back to the original tensor.

    Parameters
    ----------
    windows : torch.Tensor
        The tensor representing windows partitioned from input tensor.
    win_size : int
        The size of the square windows used to partition the tensor.
    H : int
        The height of the original tensor before partitioning.
    W : int
        The width of the original tensor before partitioning.
    dilation_rate : int
        The dilation rate for convolution. Default 1.

    Returns
    -------
    x: torch.Tensor
        The original tensor rearranged from the partitioned tensor.

    """
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(
            x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1), stride=win_size
        )
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DownSampleBlock(nn.Module):
    """Convolution based downsample block."""

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`DownSampleBlock`.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolution.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`DownSampleBlock`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled output.
        """
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`DownSampleBlock`.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channels * self.out_channels * 4 * 4
        return int(flops)


class UpSampleBlock(nn.Module):
    """Convolution based upsample block."""

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`UpSampleBlock`.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolution.
        """
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`UpSampleBlock`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Upsampled output.
        """
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`UpSampleBlock`.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channels * self.out_channels * 2 * 2
        return flops


class InputProjection(nn.Module):
    """Input convolutional projection used in the U-Former model."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        norm_layer: Optional[nn.Module] = None,
        act_layer: nn.Module = nn.LeakyReLU,
    ):
        """Inits :class:`InputProjection`.

        Parameters
        ----------
        in_channels : int
            Number of input channels. Default: 3.
        out_channels : int
            Number of output channels after the projection. Default: 64.
        kernel_size : int or tuple of ints
            Convolution kernel size. Default: 3.
        stride : int or tuple of ints
            Stride of the convolution. Default: 1.
        norm_layer : nn.Module, optional
            Normalization layer to apply after the projection. Default: None.
        act_layer : nn.Module
            Activation layer to apply after the projection. Default: nn.LeakyReLU.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True),
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`InputProjection`.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`InputProjection`.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # conv
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channels
        return flops


class OutputProjection(nn.Module):
    """Output convolutional projection used in the U-Former model."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        """Inits :class:`InputProjection`.

        Parameters
        ----------
        in_channels : int
            Number of input channels. Default: 64.
        out_channels : int
            Number of output channels after the projection. Default: 3.
        kernel_size : int or tuple of ints
            Convolution kernel size. Default: 3.
        stride : int or tuple of ints
            Stride of the convolution. Default: 1.
        norm_layer : nn.Module, optional
            Normalization layer to apply after the projection. Default: None.
        act_layer : nn.Module, optional
            Activation layer to apply after the projection. Default: None.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`OutputProjection`.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H: int, W: int) -> int:
        """Calculate the number of floating point operations in :class:`InputProjection`.

        Parameters
        ----------
        H : int
            Height.
        W : int
            Width.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # conv
        flops += H * W * self.in_channels * self.out_channels * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channels
        return flops


class LeWinTransformerMLPTokenType(DirectEnum):
    mlp = "mlp"
    ffn = "ffn"
    leff = "leff"


class LeWinTransformerBlock(nn.Module):
    """Applies a window-based multi-head self-attention and MLP or LeFF on the input tensor."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        win_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.leff,
        modulator: bool = False,
        cross_modulator: bool = False,
    ):
        r"""Inits :class:`LeWinTransformerBlock`.

        Parameters
        ----------
        dim : int
            Number of input channels.
        input_resolution : tuple of ints
            Input resolution.
        num_heads : int
            Number of attention heads.
        win_size : int
            Window size for the attention mechanism. Default: 8.
        shift_size : int
             The number of pixels to shift the window. Default: 0.
        mlp_ratio : float
            Ratio of the hidden dimension size to the embedding dimension size in the MLP layers. Default: 4.0.
        qkv_bias : bool
            Whether to use bias in the query, key, and value projections of the attention mechanism. Default: True.
        qk_scale : float, optional
            Scale factor for the query and key projection vectors.
            If set to None, will use the default value of :math`1 / \sqrt(dim)`. Default: None.
        drop : float
            Dropout rate for the token-level dropout layer. Default: 0.0.
        attn_drop : float
            Dropout rate for the attention score matrix. Default: 0.0.
        drop_path : float
            Dropout rate for the stochastic depth regularization. Default: 0.0.
        act_layer : nn.Module
            The activation function to use. Default: nn.GELU.
        norm_layer : nn.Module
            The normalization layer to use. Default: nn.LayerNorm.
        token_projection : AttentionTokenProjectionType
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.linear.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.leff.
        modulator : bool
            Whether to use a modulator in the attention mechanism. Default: False.
        cross_modulator : bool
            Whether to use cross-modulation in the attention mechanism. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
            self.cross_attn = AttentionModule(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionModule(
            dim,
            win_size=(self.win_size, self.win_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            token_projection=token_projection,
        )

        self.drop_path = DropoutPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ["ffn", "mlp"]:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == "leff":
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"
        )

    def forward(self, x, mask=None):
        """Performs the forward pass of :class:`LeWinTransformerBlock`.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2
            )  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0)
            )
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self) -> int:
        """Calculate the number of floating point operations in :class:`LeWinTransformerBlock`.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        return flops


class BasicUFormerLayer(nn.Module):
    """Basic layer of U-Former."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        win_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[bool] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] | float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        token_projection: AttentionTokenProjectionType = AttentionTokenProjectionType.linear,
        token_mlp: LeWinTransformerMLPTokenType = LeWinTransformerMLPTokenType.ffn,
        shift_flag: bool = True,
        modulator: bool = False,
        cross_modulator: bool = False,
    ):
        r"""Inits :class:`BasicUFormerLayer`.

        Parameters
        ----------
        dim : int
            Number of input channels.
        input_resolution : tuple of ints
            Input resolution.
        num_heads : int
            Number of attention heads.
        win_size : int
            Window size for the attention mechanism. Default: 8.
        mlp_ratio : float
            Ratio of the hidden dimension size to the embedding dimension size in the MLP layers. Default: 4.0.
        qkv_bias : bool
            Whether to use bias in the query, key, and value projections of the attention mechanism. Default: True.
        qk_scale : float, optional
            Scale factor for the query and key projection vectors.
            If set to None, will use the default value of :math`1 / \sqrt(dim)`. Default: None.
        drop : float
            Dropout rate for the token-level dropout layer. Default: 0.0.
        attn_drop : float
            Dropout rate for the attention score matrix. Default: 0.0.
        drop_path : float
            Dropout rate for the stochastic depth regularization. Default: 0.0.
        norm_layer : nn.Module
            The normalization layer to use. Default: nn.LayerNorm.
        token_projection : AttentionTokenProjectionType
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.linear.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.leff.
        shift_flag : bool
            Whether to use shift in the attention sliding windows or not. Default: True.
        modulator : bool
            Whether to use a modulator in the attention mechanism. Default: False.
        cross_modulator : bool
            Whether to use cross-modulation in the attention mechanism. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                LeWinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    win_size=win_size,
                    shift_size=(0 if (i % 2 == 0) else win_size // 2) if shift_flag else 0,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    token_projection=token_projection,
                    token_mlp=token_mlp,
                    modulator=modulator,
                    cross_modulator=cross_modulator,
                )
                for i in range(depth)
            ]
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`BasicUFormerLayer`.

        Parameters
        ----------
        x : torch.Tensor
        mask : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        for blk in self.blocks:
            x = blk(x, mask)
        return x

    def flops(self) -> int:
        """Calculate the number of floating point operations in :class:`BasicUFormerLayer`.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class UFormer(nn.Module):
    """U-Former is a transformer-based architecture that can process high-resolution images."""

    def __init__(
        self,
        patch_size: int = 256,
        in_channels: int = 2,
        out_channels: Optional[int] = None,
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
    ):
        """Inits :class:`UFormer`.

        Parameters
        ----------
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
        if len(encoder_num_heads) != len(encoder_depths):
            raise ValueError(
                f"The number of heads for each layer should be the same as the number of layers. "
                f"Got {len(encoder_num_heads)} for {len(encoder_depths)} layers."
            )
        if patch_size < (2 ** len(encoder_depths) * win_size):
            raise ValueError(
                f"Patch size must be greater or equal than 2 ** number of scales * window size."
                f" Received: patch_size={patch_size}, number of scales=={len(encoder_depths)},"
                f" and window_size={win_size}."
            )
        self.num_enc_layers = len(encoder_num_heads)
        self.num_dec_layers = len(encoder_num_heads)
        depths = (*encoder_depths, bottleneck_depth, *encoder_depths[::-1])
        num_heads = (*encoder_num_heads, bottleneck_num_heads, bottleneck_num_heads, *encoder_num_heads[::-1][:-1])
        self.embedding_dim = embedding_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = patch_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[: self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[self.num_enc_layers + 1]
        dec_dpr = enc_dpr[::-1]

        # Build layers

        # Input
        self.input_proj = InputProjection(
            in_channels=in_channels, out_channels=embedding_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU
        )
        out_channels = out_channels if out_channels else in_channels
        # Output
        self.output_proj = OutputProjection(
            in_channels=2 * embedding_dim, out_channels=out_channels, kernel_size=3, stride=1
        )
        if in_channels != out_channels:
            self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_enc_layers):
            layer_name = f"encoderlayer_{i}"
            layer_input_resolution = (patch_size // (2**i), patch_size // (2**i))
            layer_dim = embedding_dim * (2**i)
            layer_depth = depths[i]
            layer_drop_path = enc_dpr[sum(depths[:i]) : sum(depths[: i + 1])]
            layer = BasicUFormerLayer(
                dim=layer_dim,
                input_resolution=layer_input_resolution,
                depth=layer_depth,
                num_heads=num_heads[i],
                win_size=win_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=layer_drop_path,
                norm_layer=nn.LayerNorm,
                token_projection=token_projection,
                token_mlp=token_mlp,
                shift_flag=shift_flag,
            )
            self.encoder_layers.add_module(layer_name, layer)

            downsample_layer_name = f"downsample_{i}"
            downsample_layer = DownSampleBlock(layer_dim, embedding_dim * (2 ** (i + 1)))
            self.downsamples.add_module(downsample_layer_name, downsample_layer)
        # Bottleneck
        self.bottleneck = BasicUFormerLayer(
            dim=embedding_dim * (2**self.num_enc_layers),
            input_resolution=(patch_size // (2**self.num_enc_layers), patch_size // (2**self.num_enc_layers)),
            depth=depths[self.num_enc_layers],
            num_heads=num_heads[self.num_enc_layers],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=conv_dpr,
            norm_layer=nn.LayerNorm,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
        )
        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_dec_layers, 0, -1):
            upsample_layer_name = f"upsample_{self.num_dec_layers - i}"
            if i == self.num_dec_layers:
                upsample_in_channels = embedding_dim * (2**i)
            else:
                upsample_in_channels = embedding_dim * (2 ** (i + 1))
            upsample_out_channels = embedding_dim * (2 ** (i - 1))
            upsample_layer = UpSampleBlock(upsample_in_channels, upsample_out_channels)
            self.upsamples.add_module(upsample_layer_name, upsample_layer)

            layer_name = f"decoderlayer_{self.num_dec_layers - i}"
            layer_input_resolution = (patch_size // (2 ** (i - 1)), patch_size // (2 ** (i - 1)))
            layer_dim = embedding_dim * (2**i)
            layer_num = self.num_enc_layers + self.num_dec_layers - i + 1
            layer_depth = depths[layer_num]
            if i == self.num_dec_layers:
                layer_drop_path = dec_dpr[: depths[layer_num]]
            else:
                start = self.num_enc_layers + 1
                layer_drop_path = dec_dpr[sum(depths[start:layer_num]) : sum(depths[start : layer_num + 1])]
            layer = BasicUFormerLayer(
                dim=layer_dim,
                input_resolution=layer_input_resolution,
                depth=layer_depth,
                num_heads=num_heads[layer_num],
                win_size=win_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=layer_drop_path,
                norm_layer=nn.LayerNorm,
                token_projection=token_projection,
                token_mlp=token_mlp,
                shift_flag=shift_flag,
                modulator=modulator,
                cross_modulator=cross_modulator,
            )
            self.decoder_layers.add_module(layer_name, layer)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`UFormer`.

        Parameters
        ----------
        input : torch.Tensor
        mask : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """
        # Input Projection
        output = self.input_proj(input)
        output = self.pos_drop(output)

        # Encoder
        stack = []
        for encoder_layer, downsample in zip(self.encoder_layers, self.downsamples):
            output = encoder_layer(output, mask=mask)
            stack.append(output)
            output = downsample(output)
        # Bottleneck
        output = self.bottleneck(output, mask=mask)

        # Decoder
        for decoder_layer, upsample in zip(self.decoder_layers, self.upsamples):
            downsampled_output = stack.pop()
            output = upsample(output)

            output = torch.cat([output, downsampled_output], -1)
            output = decoder_layer(output, mask=mask)

        # Output Projection
        output = self.output_proj(output)
        if self.in_channels != self.out_channels:
            input = self.conv_out(input)
        return input + output

    def flops(self) -> int:
        """Calculate the number of floating point operations in :class:`UFormer`.

        Returns
        -------
        flops : int
            Number of floating point operations.
        """
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        for i, (encoder_layer, downsample) in enumerate(zip(self.encoder_layers, self.downsamples)):
            resolution = self.reso // (2**i)
            flops += encoder_layer.flops() + downsample.flops(resolution, resolution)

        # Bottleneck
        flops += self.bottleneck.flops()

        # Decoder
        for i, upsample, decoder_layer in zip(range(self.num_dec_layers, 0, -1), self.upsamples, self.decoder_layers):
            resolution = self.reso // (2**i)
            flops += upsample.flops(resolution, resolution) + decoder_layer.flops()
        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


class UFormerModel(nn.Module):
    """U-Former model."""

    def __init__(
        self,
        patch_size: int = 256,
        in_channels: int = 2,
        out_channels: Optional[int] = None,
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
        """Inits :class:`UFormer`.

        Parameters
        ----------
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
            Type of token projection. Must be one of ["linear", "conv"]. Default: AttentionTokenProjectionType.linear.
        token_mlp : LeWinTransformerMLPTokenType
            Type of token-level MLP. Must be one of ["leff", "mlp", "ffn"]. Default: LeWinTransformerMLPTokenType.leff.
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

        self.uformer = UFormer(
            patch_size,
            in_channels,
            out_channels,
            embedding_dim,
            encoder_depths,
            encoder_num_heads,
            bottleneck_depth,
            bottleneck_num_heads,
            win_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            patch_norm,
            token_projection,
            token_mlp,
            shift_flag,
            modulator,
            cross_modulator,
        )

        self.normalized = normalized

        self.padding_factor = win_size * (2 ** len(encoder_depths))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass of :class:`UFormer`.

        Parameters
        ----------
        x : torch.Tensor
        mask : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
        """

        x, _, wpad, hpad = pad_to_square(x, self.padding_factor)
        if self.normalized:
            x, mean, std = norm(x)

        x = self.uformer(x, mask)

        if self.normalized:
            x = unnorm(x, mean, std)
        x = unpad(x, wpad, hpad)

        return x

    def flops(self):
        return self.uformer.flops()
