# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed from https://github.com/facebookresearch/convit which uses code from
# timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from direct.nn.transformers.utils import DropoutPath, init_weights, norm, pad, unnorm, unpad

__all__ = ["VisionTransformer", "VisionTransformerModel"]


class MLP(nn.Module):
    """MLP layer with dropout and activation.


    Parameters
    ----------
    in_features : int
        Size of the input feature.
    hidden_features : int, optional
        Size of the hidden layer feature. If None, then hidden_features = in_features. (Default: None)
    out_features : int, optional
        Size of the output feature. If None, then out_features = in_features. (Default: None)
    act_layer : nn.Module, optional
        Activation layer to be used. (Default: nn.GELU)
    drop : float, optional
        Dropout probability. (Default: 0.)

    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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
    """Gated Positional Self-Attention module."""

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
    ):
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

        pos_score = self.pos_proj(self.rel_indices).expand(B, -1, -1, -1).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn = attn / attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x: torch.Tensor, return_map: Optional[bool] = False):
        """Compute the attention map for the input tensor x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C).
        return_map : bool, optional
            Whether to return the attention map. Default: False.

        Returns
        -------
        torch.Tensor
            A scalar value representing the average attention distance between patches in the input tensor x.
            If `return_map` is True, the method also returns the attention map tensor.
        """

        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** 0.5
        dist = torch.einsum("nm,hnm->h", (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

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
        if not hasattr(self, "rel_indices") or self.rel_indices.size(1) != N:
            self.get_rel_indices()

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MHSA(nn.Module):
    """Multi-Head Self-Attention (MHSA) module."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, grid_size=None):
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
        grid_size : Tuple[int, int] or None
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

    def get_attention_map(self, x, return_map=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the attention map of the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C) where B is the batch size, N is the number of
            patches and C is the number of input features.
        return_map : bool
            If True, return the attention map along with the distance. Default is False.

        Returns
        -------
            A torch.Tensor of shape (num_heads,) containing the distances between patches if return_map is
            False. Otherwise, return a tuple containing the distance and the attention map. The attention
            map is a torch.Tensor of shape (num_heads, N, N).
        """
        rel_indices = self.get_rel_indices()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)  # average over batch
        distances = rel_indices.squeeze()[:, :, -1] ** 0.5
        dist = torch.einsum("nm,hnm->h", (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def get_rel_indices(self) -> torch.Tensor:
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

        return rel_indices.to(self.qkv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """A single transformer block used in the VisionTransformer model."""

    def __init__(
        self,
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
    ):
        """Inits :class:`VisionTransformerBlock`.

        Parameters
        ----------
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
            self.attn = GPSA(
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

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass for the :class:`VisionTransformerBlock`.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        grid_size : Tuple[int, int]
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

    def __init__(self, patch_size, in_channels, embedding_dim):
        """Inits :class:`PatchEmbedding`.

        Parameters
        ----------
        patch_size : int or Tuple[int, int]
            The patch size. If an int is provided, the patch will be a square.
        in_channels : int
            Number of input channels.
        embedding_dim : int
            Dimension of the output embedding.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
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
    """Vision Transformer"""

    def __init__(
        self,
        average_img_size: Union[int, Tuple[int, int]] = 320,
        patch_size: Union[int, Tuple[int, int]] = 10,
        in_channels: int = 1,
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
        norm_layer: nn.Module = nn.LayerNorm,
        gpsa_interval: Tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
    ):
        super().__init__()

        self.depth = depth
        embedding_dim *= num_heads
        self.num_features = embedding_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embedding = use_pos_embedding

        if isinstance(average_img_size, int):
            img_size = (average_img_size, average_img_size)
        else:
            img_size = average_img_size

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size, in_channels=in_channels, embedding_dim=embedding_dim
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embedding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embedding_dim, img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
            )

            init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, dropout_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    dropout_path=dpr[i],
                    norm_layer=norm_layer,
                    use_gpsa=gpsa_interval[0] - 1 <= i < gpsa_interval[1],
                    **(
                        {"locality_strength": locality_strength}
                        if gpsa_interval[0] - 1 <= i < gpsa_interval[1]
                        else {}
                    ),
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embedding_dim)
        # head
        self.feature_info = [dict(num_chs=embedding_dim, reduction=0, module="head")]
        self.head = nn.Linear(self.num_features, self.out_channels * self.patch_size[0] * self.patch_size[1])

        self.head.apply(init_weights)

    def seq2img(self, x: torch.Tensor, img_size: Tuple[int, ...]):
        x = x.view(x.shape[0], x.shape[1], self.out_channels, self.patch_size[0], self.patch_size[1])
        x = x.chunk(x.shape[1], dim=1)
        x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3)
        x = x.chunk(img_size[0] // self.patch_size[0], dim=3)
        x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3).squeeze(1)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def get_head(self) -> nn.Module:
        return self.head

    def reset_head(self) -> None:
        self.head = nn.Linear(self.num_features, self.out_channels * self.patch_size[0] * self.patch_size[1])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        _, _, H, W = x.shape

        if self.use_pos_embedding:
            pos_embed = F.interpolate(self.pos_embed, size=[H, W], mode="bilinear", align_corners=False)
            x = x + pos_embed

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for _, block in enumerate(self.blocks):
            x = block(x, (H, W))

        x = self.norm(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`VisionTransformer`.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        _, _, H, W = x.shape
        x = self.forward_features(x)
        x = self.head(x)
        x = self.seq2img(x, (H, W))

        return x


class VisionTransformerModel(VisionTransformer):
    def __init__(
        self,
        average_img_size: Union[int, Tuple[int, int]] = 320,
        patch_size: Union[int, Tuple[int, int]] = 10,
        in_channels: int = 1,
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
        norm_layer: nn.Module = nn.LayerNorm,
        gpsa_interval: Tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
        normalized: bool = True,
    ):
        super().__init__(
            average_img_size,
            patch_size,
            in_channels,
            out_channels,
            embedding_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            dropout_path_rate,
            norm_layer,
            gpsa_interval,
            locality_strength,
            use_pos_embedding,
        )
        self.normalized = normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`VisionTransformerModel`.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        _, _, H, W = x.shape
        x, wpad, hpad = pad(x, self.patch_size)

        if self.normalized:
            x, mean, std = norm(x)

        x = self.forward_features(x)
        x = self.head(x)
        x = self.seq2img(x, (H, W))

        if self.normalized:
            x = unnorm(x, mean, std)

        x = unpad(x, wpad, hpad)

        return x
