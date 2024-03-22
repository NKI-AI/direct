# coding=utf-8
# Copyright (c) DIRECT Contributors

from direct.types import DirectEnum


class ActivationType(DirectEnum):
    relu = "relu"
    prelu = "prelu"
    leaky_rely = "leaky_relu"


class ModelName(DirectEnum):
    unet = "unet"
    normunet = "normunet"
    resnet = "resnet"
    didn = "didn"
    conv = "conv"


class InitType(DirectEnum):
    sense = "sense"
    zero_filled = "zero_filled"
    input_image = "input_image"


class LossFunType(DirectEnum):
    L1_LOSS = "l1_loss"
    KSPACE_L1_LOSS = "kspace_l1_loss"
    L2_LOSS = "l2_loss"
    KSPACE_L2_LOSS = "kspace_l2_loss"
    SSIM_LOSS = "ssim_loss"
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
