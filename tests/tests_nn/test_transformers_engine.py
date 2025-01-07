# Copyright (c) DIRECT Contributors

"""Tests for `direct.nn.transformers.transformers_engine` module."""

import functools

import numpy as np
import pytest
import torch

from direct.config.defaults import DefaultConfig, FunctionConfig, LossConfig, TrainingConfig, ValidationConfig
from direct.data.transforms import fft2, ifft2
from direct.nn.transformers.config import (
    ImageDomainMRIUFormerConfig,
    ImageDomainMRIViT2DConfig,
    ImageDomainMRIViT3DConfig,
    KSpaceDomainMRIViT2DConfig,
    KSpaceDomainMRIViT3DConfig,
)
from direct.nn.transformers.transformers import (
    ImageDomainMRIUFormer,
    ImageDomainMRIViT2D,
    ImageDomainMRIViT3D,
    KSpaceDomainMRIViT2D,
    KSpaceDomainMRIViT3D,
)
from direct.nn.transformers.transformers_engine import (
    ImageDomainMRIUFormerEngine,
    ImageDomainMRIViT2DEngine,
    ImageDomainMRIViT3DEngine,
    KSpaceDomainMRIViT2DEngine,
    KSpaceDomainMRIViT3DEngine,
)
from direct.nn.transformers.uformer import AttentionTokenProjectionType, LeWinTransformerMLPTokenType


def create_sample(shape, **kwargs):
    sample = dict()
    sample["masked_kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["kspace"] = torch.from_numpy(np.random.randn(*shape)).float()
    sample["sensitivity_map"] = torch.from_numpy(np.random.randn(*shape)).float()
    for k, v in locals()["kwargs"].items():
        sample[k] = v
    return sample


@pytest.mark.parametrize(
    "shape",
    [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "kspace_nmse_loss", "kspace_nmae_loss"]],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [20],
)
@pytest.mark.parametrize(
    "patch_size",
    [140],
)
@pytest.mark.parametrize(
    "encoder_depths, encoder_num_heads, bottleneck_depth, bottleneck_num_heads",
    [
        [(2, 2, 2), (1, 2, 4), 1, 8],
    ],
)
@pytest.mark.parametrize(
    "patch_norm",
    [True],
)
@pytest.mark.parametrize(
    "win_size",
    [8],
)
@pytest.mark.parametrize(
    "mlp_ratio",
    [2],
)
@pytest.mark.parametrize(
    "qkv_bias",
    [False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [0.5],
)
@pytest.mark.parametrize(
    "token_projection",
    [AttentionTokenProjectionType.CONV],
)
@pytest.mark.parametrize(
    "token_mlp",
    [LeWinTransformerMLPTokenType.MLP],
)
def test_image_uformer_engine(
    shape,
    loss_fns,
    embedding_dim,
    patch_size,
    encoder_depths,
    encoder_num_heads,
    bottleneck_depth,
    bottleneck_num_heads,
    patch_norm,
    win_size,
    mlp_ratio,
    qkv_bias,
    qk_scale,
    token_projection,
    token_mlp,
):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = ImageDomainMRIUFormerConfig(
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
        patch_norm=patch_norm,
        token_projection=token_projection,
        token_mlp=token_mlp,
    )
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    # Models
    model = ImageDomainMRIUFormer(
        forward_operator,
        backward_operator,
        patch_size=model_config.patch_size,
        embedding_dim=model_config.embedding_dim,
        encoder_depths=model_config.encoder_depths,
        encoder_num_heads=model_config.encoder_num_heads,
        bottleneck_depth=model_config.bottleneck_depth,
        bottleneck_num_heads=model_config.bottleneck_num_heads,
        win_size=model_config.win_size,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        qk_scale=model_config.qk_scale,
        patch_norm=model_config.patch_norm,
        token_projection=model_config.token_projection,
        token_mlp=model_config.token_mlp,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Define engine
    engine = ImageDomainMRIUFormerEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    out.output_image.shape == (shape[0],) + tuple(shape[2:-1])


@pytest.mark.parametrize(
    "shape",
    [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "kspace_nmse_loss", "kspace_nmae_loss"]],
)
@pytest.mark.parametrize(
    "patch_size",
    [8, (8, 10)],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [8],
)
@pytest.mark.parametrize(
    "depth",
    [4],
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
    [False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [None],
)
@pytest.mark.parametrize(
    "use_gpsa",
    [False],
)
@pytest.mark.parametrize(
    "locality_strength",
    [1.0],
)
@pytest.mark.parametrize(
    "use_pos_embedding",
    [True],
)
@pytest.mark.parametrize(
    "normalized",
    [True],
)
def test_image_vit2d_engine(
    shape,
    loss_fns,
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
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = ImageDomainMRIViT2DConfig(
        patch_size=patch_size,
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
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    # Models
    model = ImageDomainMRIViT2D(
        forward_operator,
        backward_operator,
        patch_size=model_config.patch_size,
        embedding_dim=model_config.embedding_dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        qk_scale=model_config.qk_scale,
        use_gpsa=model_config.use_gpsa,
        locality_strength=model_config.locality_strength,
        use_pos_embedding=model_config.use_pos_embedding,
        normalized=model_config.normalized,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Define engine
    engine = ImageDomainMRIViT2DEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    out.output_image.shape == (shape[0],) + tuple(shape[2:-1])


@pytest.mark.parametrize(
    "shape",
    [(4, 3, 10, 16, 2), (5, 1, 10, 12, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [["l1_loss", "kspace_nmse_loss", "kspace_nmae_loss"]],
)
@pytest.mark.parametrize(
    "patch_size",
    [(10, 10)],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [8],
)
@pytest.mark.parametrize(
    "depth",
    [4],
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
    [False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [None],
)
@pytest.mark.parametrize(
    "use_gpsa",
    [True],
)
@pytest.mark.parametrize(
    "locality_strength",
    [1.0],
)
@pytest.mark.parametrize(
    "use_pos_embedding",
    [True],
)
@pytest.mark.parametrize(
    "normalized",
    [True],
)
@pytest.mark.parametrize(
    "compute_per_coil",
    [True, False],
)
def test_kspace_vit2d_engine(
    shape,
    loss_fns,
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
    compute_per_coil,
):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = KSpaceDomainMRIViT2DConfig(
        patch_size=patch_size,
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
        compute_per_coil=compute_per_coil,
    )
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    # Models
    model = KSpaceDomainMRIViT2D(
        forward_operator,
        backward_operator,
        patch_size=model_config.patch_size,
        embedding_dim=model_config.embedding_dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        qk_scale=model_config.qk_scale,
        use_gpsa=model_config.use_gpsa,
        locality_strength=model_config.locality_strength,
        use_pos_embedding=model_config.use_pos_embedding,
        normalized=model_config.normalized,
        compute_per_coil=model_config.compute_per_coil,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Define engine
    engine = KSpaceDomainMRIViT2DEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 2
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, shape[2], shape[3], 1)).bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3])).float(),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    out.output_image.shape == (shape[0],) + tuple(shape[2:-1])


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4, 10, 16, 2), (1, 11, 8, 12, 16, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [
        [
            "l1_loss",
            "snr_loss",
            "kspace_nmae_loss",
            "ssim_3d_loss",
        ]
    ],
)
@pytest.mark.parametrize(
    "patch_size",
    [(4, 8, 10)],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [8],
)
@pytest.mark.parametrize(
    "depth",
    [4],
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
    [False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [None],
)
@pytest.mark.parametrize(
    "use_gpsa",
    [False],
)
@pytest.mark.parametrize(
    "locality_strength",
    [1.0],
)
@pytest.mark.parametrize(
    "use_pos_embedding",
    [True],
)
@pytest.mark.parametrize(
    "normalized",
    [False],
)
def test_image_vit3d_engine(
    shape,
    loss_fns,
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
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = ImageDomainMRIViT3DConfig(
        patch_size=patch_size,
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
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    # Models
    model = ImageDomainMRIViT3D(
        forward_operator,
        backward_operator,
        patch_size=model_config.patch_size,
        embedding_dim=model_config.embedding_dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        qk_scale=model_config.qk_scale,
        use_gpsa=model_config.use_gpsa,
        locality_strength=model_config.locality_strength,
        use_pos_embedding=model_config.use_pos_embedding,
        normalized=model_config.normalized,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Define engine
    engine = ImageDomainMRIViT3DEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 3
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, 1, shape[3], shape[4], 1)).bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3], shape[4])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    out.output_image.shape == (shape[0],) + tuple(shape[2:-1])


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4, 10, 16, 2), (1, 11, 8, 12, 16, 2)],
)
@pytest.mark.parametrize(
    "loss_fns",
    [
        [
            "l1_loss",
            "snr_loss",
            "kspace_nmse_loss",
            "ssim_3d_loss",
        ]
    ],
)
@pytest.mark.parametrize(
    "patch_size",
    [6],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [12],
)
@pytest.mark.parametrize(
    "depth",
    [4],
)
@pytest.mark.parametrize(
    "num_heads",
    [6],
)
@pytest.mark.parametrize(
    "mlp_ratio",
    [2.0],
)
@pytest.mark.parametrize(
    "qkv_bias",
    [False],
)
@pytest.mark.parametrize(
    "qk_scale",
    [None],
)
@pytest.mark.parametrize(
    "use_gpsa",
    [True],
)
@pytest.mark.parametrize(
    "locality_strength",
    [1.0],
)
@pytest.mark.parametrize(
    "use_pos_embedding",
    [False],
)
@pytest.mark.parametrize(
    "normalized",
    [True],
)
@pytest.mark.parametrize(
    "compute_per_coil",
    [True, False],
)
def test_kspace_vit3d_engine(
    shape,
    loss_fns,
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
    compute_per_coil,
):
    # Operators
    forward_operator = functools.partial(fft2, centered=True)
    backward_operator = functools.partial(ifft2, centered=True)
    # Configs
    loss_config = LossConfig(losses=[FunctionConfig(loss) for loss in loss_fns])
    training_config = TrainingConfig(loss=loss_config)
    validation_config = ValidationConfig(crop=None)
    model_config = KSpaceDomainMRIViT3DConfig(
        patch_size=patch_size,
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
        compute_per_coil=compute_per_coil,
    )
    config = DefaultConfig(training=training_config, validation=validation_config, model=model_config)
    # Models
    model = KSpaceDomainMRIViT3D(
        forward_operator,
        backward_operator,
        patch_size=model_config.patch_size,
        embedding_dim=model_config.embedding_dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        qk_scale=model_config.qk_scale,
        use_gpsa=model_config.use_gpsa,
        locality_strength=model_config.locality_strength,
        use_pos_embedding=model_config.use_pos_embedding,
        normalized=model_config.normalized,
        compute_per_coil=model_config.compute_per_coil,
    )
    sensitivity_model = torch.nn.Conv2d(2, 2, kernel_size=1)
    # Define engine
    engine = KSpaceDomainMRIViT3DEngine(config, model, "cpu", fft2, ifft2, sensitivity_model=sensitivity_model)
    engine.ndim = 3
    # Test _do_iteration function with a single data batch
    data = create_sample(
        shape,
        sampling_mask=torch.from_numpy(np.random.randn(1, 1, 1, shape[3], shape[4], 1)).bool(),
        target=torch.from_numpy(np.random.randn(shape[0], shape[2], shape[3], shape[4])).float(),
        scaling_factor=torch.ones(shape[0]),
    )
    loss_fns = engine.build_loss()
    out = engine._do_iteration(data, loss_fns)
    out.output_image.shape == (shape[0],) + tuple(shape[2:-1])
