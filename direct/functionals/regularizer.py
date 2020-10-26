# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch


def body_coil(predicted_image, gamma=0.5, epsilon=10e-8, **sample):
    body_coil_image = sample["body_coil_image"]

    regularizer_term = (
        torch.pow(body_coil_image + epsilon, -0.5).align_as(predicted_image)
        * predicted_image
    )
    regularizer_term = (regularizer_term ** 2).sum("complex").sum("height").sum("width")

    return gamma * regularizer_term
