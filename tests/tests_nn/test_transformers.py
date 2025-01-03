# Copyright (c) DIRECT Contributors

"""Tests for transformers models."""

import pytest
import torch

from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType, UFormerModel
from direct.nn.transformers.vit import VisionTransformer2D, VisionTransformer3D


def create_input(shape):
    data = torch.rand(shape).float()

    return data


# @pytest.mark.parametrize(
#     "shape",
#     [
#         [3, 2, 32, 32],
#         [3, 2, 16, 16],
#     ],
# )
# @pytest.mark.parametrize(
#     "embedding_dim",
#     [20],
# )
# @pytest.mark.parametrize(
#     "patch_size",
#     [140],
# )
# @pytest.mark.parametrize(
#     "encoder_depths, encoder_num_heads, bottleneck_depth, bottleneck_num_heads",
#     [
#         [(2, 2, 2), (1, 2, 4), 1, 8],
#         [(2, 2, 2, 2), (1, 2, 4, 8), 2, 8],
#     ],
# )
# @pytest.mark.parametrize(
#     "patch_norm",
#     [True, False],
# )
# @pytest.mark.parametrize(
#     "win_size",
#     [8],
# )
# @pytest.mark.parametrize(
#     "mlp_ratio",
#     [2],
# )
# @pytest.mark.parametrize(
#     "qkv_bias",
#     [True, False],
# )
# @pytest.mark.parametrize(
#     "qk_scale",
#     [None, 0.5],
# )
# @pytest.mark.parametrize(
#     "token_projection",
#     [AttentionTokenProjectionType.LINEAR, AttentionTokenProjectionType.CONV],
# )
# @pytest.mark.parametrize(
#     "token_mlp",
#     [LeWinTransformerMLPTokenType.FFN, LeWinTransformerMLPTokenType.MLP, LeWinTransformerMLPTokenType.LEFF],
# )
# def test_uformer(
#     shape,
#     patch_size,
#     embedding_dim,
#     encoder_depths,
#     encoder_num_heads,
#     bottleneck_depth,
#     bottleneck_num_heads,
#     win_size,
#     mlp_ratio,
#     patch_norm,
#     qkv_bias,
#     qk_scale,
#     token_projection,
#     token_mlp,
# ):
#     model = UFormerModel(
#         patch_size=patch_size,
#         in_channels=2,
#         embedding_dim=embedding_dim,
#         encoder_depths=encoder_depths,
#         encoder_num_heads=encoder_num_heads,
#         bottleneck_depth=bottleneck_depth,
#         bottleneck_num_heads=bottleneck_num_heads,
#         win_size=win_size,
#         mlp_ratio=mlp_ratio,
#         qkv_bias=qkv_bias,
#         qk_scale=qk_scale,
#         patch_norm=patch_norm,
#         token_projection=token_projection,
#         token_mlp=token_mlp,
#     )
#     data = create_input(shape).cpu()
#     out = model(data)
#     assert list(out.shape) == shape


@pytest.mark.parametrize(
    "shape, average_img_size",
    [
        [[1, 3, 128, 128], 128],
        [[3, 2, 64, 50], (64, 50)],
    ],
)
@pytest.mark.parametrize(
    "patch_size",
    [16, 8, (16, 10)],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [6, 12],
)
@pytest.mark.parametrize(
    "depth",
    [2, 4],
)
@pytest.mark.parametrize(
    "num_heads",
    [3, 4],
)
@pytest.mark.parametrize(
    "mlp_ratio",
    [4.0, 2.0],
)
@pytest.mark.parametrize(
    "qkv_bias",
    [True, False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [None, 0.5],
)
@pytest.mark.parametrize(
    "use_gpsa",
    [True, False],
)
@pytest.mark.parametrize(
    "locality_strength",
    [0.5],
)
@pytest.mark.parametrize(
    "use_pos_embedding",
    [True, False],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_vision_transformer_2d(
    shape,
    average_img_size,
    patch_size,
    embedding_dim,
    depth,
    num_heads,
    mlp_ratio,
    qkv_bias,
    qk_scale,
    use_gpsa,
    locality_strength,
    use_pos_embedding,
    normalized,
):
    model = VisionTransformer2D(
        average_img_size=average_img_size,
        patch_size=patch_size,
        in_channels=shape[1],
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        use_gpsa=use_gpsa,
        locality_strength=locality_strength,
        use_pos_embedding=use_pos_embedding,
        normalized=normalized,
    )
    data = create_input(shape).cpu()
    out = model(data)
    assert list(out.shape) == [shape[0], shape[1], shape[2], shape[3]]


@pytest.mark.parametrize(
    "shape, average_img_size",
    [
        [[1, 3, 64, 64, 64], 64],
        [[2, 2, 32, 32, 32], (32, 32, 32)],
    ],
)
@pytest.mark.parametrize(
    "patch_size",
    [8, (8, 6, 8)],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [8, 16],
)
@pytest.mark.parametrize(
    "depth",
    [4, 8],
)
@pytest.mark.parametrize(
    "num_heads",
    [6],
)
@pytest.mark.parametrize(
    "mlp_ratio",
    [4.0],
)
@pytest.mark.parametrize(
    "qkv_bias",
    [True, False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [None, 0.5],
)
@pytest.mark.parametrize(
    "use_gpsa",
    [True, False],
)
@pytest.mark.parametrize(
    "locality_strength",
    [1.0],
)
@pytest.mark.parametrize(
    "use_pos_embedding",
    [True, False],
)
@pytest.mark.parametrize(
    "normalized",
    [True, False],
)
def test_vision_transformer_3d(
    shape,
    average_img_size,
    patch_size,
    embedding_dim,
    depth,
    num_heads,
    mlp_ratio,
    qkv_bias,
    qk_scale,
    use_gpsa,
    locality_strength,
    use_pos_embedding,
    normalized,
):
    model = VisionTransformer3D(
        average_img_size=average_img_size,
        patch_size=patch_size,
        in_channels=shape[1],
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        use_gpsa=use_gpsa,
        locality_strength=locality_strength,
        use_pos_embedding=use_pos_embedding,
        normalized=normalized,
    )
    data = create_input(shape).cpu()
    out = model(data)
    assert list(out.shape) == [shape[0], shape[1], shape[2], shape[3], shape[4]]
