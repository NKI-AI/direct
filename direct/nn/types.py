# Copyright (c) DIRECT Contributors

"""direct.nn.types module."""

from direct.types import DirectEnum


class ActivationType(DirectEnum):
    RELU = "relu"
    PRELU = "prelu"
    LEAKY_RELU = "leaky_relu"


class ModelName(DirectEnum):
    UNET = "unet"
    NORMUNET = "normunet"
    RESNET = "resnet"
    DIDN = "didn"
    CONV = "conv"


class InitType(DirectEnum):
    INPUT_IMAGE = "input_image"
    SENSE = "sense"
    ZERO_FILLED = "zero_filled"
    ZEROS = "zeros"


class LossFunType(DirectEnum):
    L1_LOSS = "l1_loss"
    KSPACE_L1_LOSS = "kspace_l1_loss"
    L2_LOSS = "l2_loss"
    KSPACE_L2_LOSS = "kspace_l2_loss"
    SSIM_LOSS = "ssim_loss"
    SSIM_3D_LOSS = "ssim_3d_loss"
    GRAD_L1_LOSS = "grad_l1_loss"
    GRAD_L2_LOSS = "grad_l2_loss"
    NMSE_LOSS = "nmse_loss"
    KSPACE_NMSE_LOSS = "kspace_nmse_loss"
    NRMSE_LOSS = "nrmse_loss"
    KSPACE_NRMSE_LOSS = "kspace_nrmse_loss"
    NMAE_LOSS = "nmae_loss"
    KSPACE_NMAE_LOSS = "kspace_nmae_loss"
    SNR_LOSS = "snr_loss"
    PSNR_LOSS = "psnr_loss"
    HFEN_L1_LOSS = "hfen_l1_loss"
    HFEN_L2_LOSS = "hfen_l2_loss"
    HFEN_L1_NORM_LOSS = "hfen_l1_norm_loss"
    HFEN_L2_NORM_LOSS = "hfen_l2_norm_loss"
    DISPLACEMENT_FIELD_NMSE_LOSS = "displacement_field_nmse_loss"
    DISPLACEMENT_FIELD_NRMSE_LOSS = "displacement_field_nrmse_loss"
    DISPLACEMENT_FIELD_NMAE_LOSS = "displacement_field_nmae_loss"
    SMOOTH_LOSS_L1 = "smooth_loss_l1"
    KSPACE_SMOOTH_LOSS_L1 = "kspace_smooth_loss_l1"
    DISPLACEMENT_FIELD_SMOOTH_LOSS_L1 = "displacement_field_smooth_loss_l1"
    SMOOTH_LOSS_L2 = "smooth_loss_l2"
    KSPACE_SMOOTH_LOSS_L2 = "kspace_smooth_loss_l2"
    DISPLACEMENT_FIELD_SMOOTH_LOSS_L2 = "displacement_field_smooth_loss_l2"
