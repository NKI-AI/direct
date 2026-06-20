#!/usr/bin/env python3
"""Generate fixed vSHARP paper configs under projects/modulated_convolution/configs/vsharp/."""

from __future__ import annotations

import copy
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "projects" / "modulated_convolution" / "configs" / "vsharp"

KNEE_TRAINING = {
    "num_iterations": 80001,
    "lr_step_size": 15000,
    "random_flip_probability": 0.5,
    "random_rotation_probability": 0.5,
}

PROSTATE_TRAINING = {
    "num_iterations": 150000,
    "lr_step_size": 30000,
    "random_flip_probability": 0.0,
    "random_rotation_probability": 0.0,
}

LONG_TRAINING = {
    "num_iterations": 150001,
    "lr_step_size": 15000,
    "random_flip_probability": 0.5,
    "random_rotation_probability": 0.5,
}


def base_model() -> dict:
    return {
        "model_name": "vsharp.vsharp.VSharpNet",
        "num_steps": 12,
        "num_steps_dc_gd": 10,
        "image_init": "SENSE",
        "no_parameter_sharing": True,
        "auxiliary_steps": -1,
        "image_model_architecture": "UNET",
        "initializer_channels": [32, 32, 64, 64],
        "initializer_dilations": [1, 1, 2, 4],
        "initializer_multiscale": 1,
        "initializer_activation": "PRELU",
        "conv_modulation": "NONE",
        "aux_in_features": 2,
        "log_aux": True,
        "fc_hidden_features": [32, 32],
        "fc_groups": 1,
        "fc_activation": "SOFTPLUS",
        "num_weights": None,
        "modulation_at_input": False,
        "image_resnet_hidden_channels": 128,
        "image_resnet_num_blocks": 15,
        "image_resnet_batchnorm": True,
        "image_resnet_scale": 0.1,
        "image_unet_num_filters": 32,
        "image_unet_num_pool_layers": 4,
        "image_unet_dropout": 0.0,
        "image_unet_norm_type": "INSTANCE",
        "image_unet_adain_hidden_features": None,
        "image_didn_hidden_channels": 16,
        "image_didn_num_dubs": 6,
        "image_didn_num_convs_recon": 9,
        "image_conv_hidden_channels": 64,
        "image_conv_n_convs": 15,
        "image_conv_activation": "ActivationType.RELU",
        "image_conv_batchnorm": False,
    }


def sensitivity_model() -> dict:
    return {
        "model_name": "unet.unet_2d.UnetModel2d",
        "in_channels": 2,
        "out_channels": 2,
        "num_filters": 16,
        "num_pool_layers": 4,
        "dropout_probability": 0.0,
        "modulation": "NONE",
        "aux_in_features": None,
        "fc_hidden_features": None,
        "fc_groups": 1,
        "fc_activation": "SIGMOID",
        "num_weights": None,
    }


def build_cfg(anatomy: str, training_overrides: dict, model_overrides: dict) -> dict:
    if anatomy == "knee":
        train_params = KNEE_TRAINING
    elif anatomy == "prostate":
        train_params = PROSTATE_TRAINING
    else:
        raise ValueError(anatomy)

    train_params = {**train_params, **training_overrides}
    transforms = {
        "crop": "reconstruction_size",
        "estimate_sensitivity_maps": True,
        "scaling_key": "masked_kspace",
        "image_center_crop": False,
        "masking": {
            "name": "FastMRIEquispaced",
            "accelerations": [4, 16],
            "center_fractions": [0.08, 0.02],
            "uniform_range": False,
            "linear_range": True,
        },
        "scale_percentile": 0.995,
        "use_seed": False,
        "delete_kspace": False,
    }
    if train_params["random_flip_probability"] > 0.0:
        transforms["random_flip_probability"] = train_params["random_flip_probability"]
    if train_params["random_rotation_probability"] > 0.0:
        transforms["random_rotation_probability"] = train_params["random_rotation_probability"]

    validation_datasets = []
    for accel, cf, desc in [(4, 0.08, "4x"), (8, 0.04, "8x"), (16, 0.02, "16x")]:
        validation_datasets.append(
            {
                "name": "FastMRI",
                "transforms": {
                    "estimate_sensitivity_maps": True,
                    "scaling_key": "masked_kspace",
                    "masking": {
                        "name": "FastMRIEquispaced",
                        "accelerations": [accel],
                        "center_fractions": [cf],
                    },
                    "scale_percentile": 0.995,
                    "use_seed": True,
                },
                "text_description": desc,
            }
        )

    model = base_model()
    model.update(model_overrides)

    return {
        "model": model,
        "additional_models": {"sensitivity_model": sensitivity_model()},
        "physics": {
            "forward_operator": "fft2",
            "backward_operator": "ifft2",
            "use_noise_matrix": False,
            "noise_matrix_scaling": 1.0,
        },
        "training": {
            "datasets": [{"name": "FastMRI", "transforms": transforms}],
            "model_checkpoint": None,
            "optimizer": "Adam",
            "lr": 0.002,
            "weight_decay": 0.0,
            "batch_size": 1,
            "lr_step_size": train_params["lr_step_size"],
            "lr_gamma": 0.8,
            "lr_warmup_iter": 1000,
            "swa_start_iter": None,
            "num_iterations": train_params["num_iterations"],
            "validation_steps": 4000,
            "gradient_steps": 1,
            "gradient_clipping": 0.0,
            "gradient_debug": False,
            "loss": {
                "crop": "header",
                "losses": [
                    {"function": "l1_loss", "multiplier": 1.0},
                    {"function": "ssim_loss", "multiplier": 1.0},
                    {"function": "hfen_l2_norm_loss", "multiplier": 1.0},
                    {"function": "hfen_l1_norm_loss", "multiplier": 1.0},
                    {"function": "kspace_nmae_loss", "multiplier": 1.0},
                    {"function": "kspace_nmse_loss", "multiplier": 1.0},
                ],
            },
            "checkpointer": {"checkpoint_steps": 4000},
            "metrics": [],
            "regularizers": [],
        },
        "validation": {
            "datasets": validation_datasets,
            "batch_size": 20,
            "metrics": ["fastmri_psnr", "fastmri_ssim", "fastmri_nmse"],
            "regularizers": [],
            "crop": "header",
        },
        "inference": {
            "dataset": {
                "name": "FastMRI",
                "transforms": {
                    "masking": {
                        "name": "FastMRIEquispaced",
                        "accelerations": [4.0],
                        "center_fractions": [0.08],
                        "uniform_range": False,
                        "mode": "STATIC",
                    },
                    "cropping": {"crop": None},
                    "sensitivity_map_estimation": {"estimate_sensitivity_maps": True},
                    "normalization": {
                        "scaling_key": "masked_kspace",
                        "scale_percentile": 0.995,
                    },
                    "use_seed": True,
                },
                "text_description": "inference-4x",
            },
            "batch_size": 1,
            "crop": "header",
        },
        "logging": {"log_as_image": None, "tensorboard": {"num_images": 4}},
    }


VARIANTS = {
    "triang": {
        "model": {"conv_modulation": "NONE"},
        "training": {},
        "knee_only": False,
    },
    "modconv_features_triang": {
        "model": {
            "conv_modulation": "FEATURES",
            "aux_in_features": 2,
            "log_aux": True,
            "fc_hidden_features": [32, 32],
            "fc_activation": "SOFTPLUS",
        },
        "training": {},
        "knee_only": False,
    },
    "modconv_features_triang_32_8": {
        "model": {
            "conv_modulation": "FEATURES",
            "aux_in_features": 2,
            "log_aux": True,
            "fc_hidden_features": [32, 8],
            "fc_activation": "SOFTPLUS",
        },
        "training": {},
        "knee_only": False,
    },
    "modconv_features_triang_32_16": {
        "model": {
            "conv_modulation": "FEATURES",
            "aux_in_features": 2,
            "log_aux": True,
            "fc_hidden_features": [32, 16],
            "fc_activation": "SOFTPLUS",
        },
        "training": {},
        "knee_only": False,
    },
    "modconv_features_triang_32_16_mod_inp": {
        "model": {
            "conv_modulation": "FEATURES",
            "aux_in_features": 2,
            "log_aux": True,
            "fc_hidden_features": [32, 16],
            "fc_activation": "SOFTPLUS",
            "modulation_at_input": True,
        },
        "training": LONG_TRAINING,
        "knee_only": True,
    },
    "modconv_partial_in_triang": {
        "model": {
            "conv_modulation": "PARTIAL_IN",
            "aux_in_features": 2,
            "log_aux": True,
            "fc_hidden_features": [32, 32],
            "fc_activation": "SOFTPLUS",
        },
        "training": {},
        "knee_only": False,
    },
    "adain_triang_32_16": {
        "model": {
            "conv_modulation": "NONE",
            "aux_in_features": 2,
            "log_aux": True,
            "image_unet_norm_type": "ADAIN",
            "image_unet_adain_hidden_features": [32, 16],
        },
        "training": LONG_TRAINING,
        "knee_only": True,
    },
}


def main() -> None:
    for anatomy in ("knee", "prostate"):
        out_dir = OUT / anatomy
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, spec in VARIANTS.items():
            if spec["knee_only"] and anatomy != "knee":
                continue
            cfg = build_cfg(anatomy, spec["training"], spec["model"])
            path = out_dir / f"vsharp_{name}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f"Wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
