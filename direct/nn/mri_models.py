# Copyright (c) DIRECT Contributors

"""MRI model engine of DIRECT."""

from __future__ import annotations

import gc
import pathlib
import time
from collections import defaultdict
from os import PathLike
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

import direct.data.transforms as T
import direct.functionals as D
from direct.config import BaseConfig
from direct.engine import DoIterationOutput, Engine
from direct.nn.types import LossFunType
from direct.types import TensorOrNone
from direct.utils import (
    communication,
    detach_dict,
    dict_to_device,
    filter_arguments_by_signature,
    merge_list_of_dicts,
    multiply_function,
    reduce_list_of_dicts,
)
from direct.utils.communication import reduce_tensor_dict


class MRIModelEngine(Engine):
    """Engine for MRI models.

    Each child class should implement their own :meth:`forward_function`.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`MRIModelEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models.
        """
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )
        self._spatial_dims = (2, 3)
        self._coil_dim = 1
        self._complex_dim = -1

    def forward_function(self, data: dict[str, Any]) -> tuple[TensorOrNone, TensorOrNone]:
        """This method performs the model's forward method given `data` which contains all tensor inputs.

        Must be implemented by child classes.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: Optional[dict[str, Callable]] = None,
        regularizer_fns: Optional[dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[dict[str, Callable]]
            Callable regularization functions.

        Returns
        -------
        DoIterationOutput
            Contains outputs.
        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        data = dict_to_device(data, self.device)

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_image, output_kspace = self.forward_function(data)
            output_image = T.modulus_if_complex(output_image, complex_axis=self._complex_dim)

            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
            regularizer_dict = {
                k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
            }

            if self.ndim == 3 and "registration_model" in self.models:
                # Perform registration and compute loss on registered image and displacement field
                registered_image, displacement_field = self.do_registration(data, output_image)

                # If DL-based model calculate loss
                if len(list(self.models["registration_model"].parameters())) > 0:
                    shape = data["reference_image"].shape
                    loss_dict = self.compute_loss_on_data(
                        loss_dict,
                        loss_fns,
                        data,
                        output_image=registered_image,
                        target_image=(
                            data["reference_image"]
                            if shape == registered_image.shape
                            else data["reference_image"].tile((1, registered_image.shape[1], *([1] * len(shape[1:]))))
                        ),
                    )
                    loss_dict = self.compute_loss_on_data(
                        loss_dict,
                        loss_fns,
                        data,
                        output_displacement_field=displacement_field,
                        target_displacement_field=data["displacement_field"],
                    )
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, output_kspace)
            regularizer_dict = self.compute_loss_on_data(
                regularizer_dict, regularizer_fns, data, output_image, output_kspace
            )

            loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.
        regularizer_dict = detach_dict(regularizer_dict)

        return DoIterationOutput(
            output_image=(
                (output_image, registered_image)
                if (self.ndim == 3 and "registration_model" in self.models)
                else output_image
            ),
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
            data_dict={**loss_dict, **regularizer_dict},
        )

    def build_loss(self) -> dict:
        def get_resolution(reconstruction_size):
            return _compute_resolution(self.cfg.training.loss.crop, reconstruction_size)  # type: ignore

        def nmae_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate NMAE loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, *).
            target: torch.Tensor
                Has shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            nmae_loss: torch.Tensor
                NMAE loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution=resolution)

            nmae_loss = D.NMAELoss(reduction=reduction).forward(source, target)

            return nmae_loss

        def nmse_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate NMSE loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, *).
            target: torch.Tensor
                Has shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            nmse_loss: torch.Tensor
                NMSE loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution=resolution)
            nmse_loss = D.NMSELoss(reduction=reduction).forward(source, target)

            return nmse_loss

        def nrmse_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate NRMSE loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, *).
            target: torch.Tensor
                Has shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            nrmse_loss: torch.Tensor
                NRMSE loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution=resolution)
            nrmse_loss = D.NRMSELoss(reduction=reduction).forward(source, target)

            return nrmse_loss

        def l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate L1 loss given source image and target.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, *).
            target: torch.Tensor
                Target tensor of shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            l1_loss: torch.Tensor
                L1 loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution=resolution)
            l1_loss = F.l1_loss(source, target, reduction=reduction)

            return l1_loss

        def l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate L2 loss (MSE) given source image and and `data` containing target.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, *).
            target: torch.Tensor
                Target tensor of shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            l2_loss: torch.Tensor
                L2 loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution=resolution)
            l2_loss = F.mse_loss(source, target, reduction=reduction)

            return l2_loss

        def ssim_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate SSIM loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            ssim_loss: torch.Tensor
                SSIM loss.
            """
            resolution = get_resolution(reconstruction_size)
            if reduction != "mean":
                raise AssertionError(
                    f"SSIM loss can only be computed with reduction == 'mean'." f" Got reduction == {reduction}."
                )
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = D.SSIMLoss().to(source_abs.device).forward(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        def ssim_3d_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate SSIM3D loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, slice, height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, slice, height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            ssim_loss: torch.Tensor
                SSIM loss.
            """
            if self.ndim != 3:
                raise NotImplementedError(
                    f"Requested to compute `ssim_3d_loss` with Engine with ndim={self.ndim}, "
                    f"but ssim_3d_loss is only implemented for 3D data."
                )
            resolution = get_resolution(reconstruction_size)
            if reduction != "mean":
                raise AssertionError(
                    f"SSIM loss can only be computed with reduction == 'mean'." f" Got reduction == {reduction}."
                )
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = D.SSIM3DLoss().to(source_abs.device).forward(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        def grad_l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate Sobel gradient L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            grad_loss: torch.Tensor
                Sobel grad L1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)
            grad_l1_loss = D.SobelGradL1Loss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return grad_l1_loss

        def grad_l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate Sobel gradient L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            grad_loss: torch.Tensor
                Sobel grad L1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)
            grad_l2_loss = D.SobelGradL2Loss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return grad_l2_loss

        def psnr_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate peak signal-to-noise ratio loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            psnr_loss: torch.Tensor
               PSNR loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)
            psnr_loss = -D.PSNRLoss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return psnr_loss

        def snr_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate signal-to-noise loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            snr_loss: torch.Tensor
                SNR loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)
            snr_loss = -D.SNRLoss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return snr_loss

        def hfen_l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                HFEN l1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)

            return D.HFENL1Loss(reduction=reduction, norm=False).to(source_abs.device).forward(source_abs, target_abs)

        def hfen_l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                HFEN l2 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)

            return D.HFENL2Loss(reduction=reduction, norm=False).to(source_abs.device).forward(source_abs, target_abs)

        def hfen_l1_norm_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                Normalized HFEN l1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)

            return D.HFENL1Loss(reduction=reduction, norm=True).to(source_abs.device).forward(source_abs, target_abs)

        def hfen_l2_norm_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice/time], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                Normalized HFEN l2 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution=resolution)

            return D.HFENL2Loss(reduction=reduction, norm=True).to(source_abs.device).forward(source_abs, target_abs)

        def smooth_loss_l1(
            source: torch.Tensor,
            _: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate smoothness loss based on the L1 penalty of the gradients of the input tensor.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
            """
            resolution = get_resolution(reconstruction_size)

            source = _reduce_slice_dim(source)[0]
            source = _crop_volume(source, resolution=resolution)[0]

            return D.SmoothLossL1(reduction=reduction).to(source.device).forward(source)

        def smooth_loss_l2(
            source: torch.Tensor,
            _: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[tuple] = None,
        ) -> torch.Tensor:
            """Calculate smoothness loss based on the L2 penalty of the gradients of the input tensor.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice/time], height, width).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            torch.Tensor
            """
            resolution = get_resolution(reconstruction_size)

            source = _reduce_slice_dim(source)[0]
            source = _crop_volume(source, resolution=resolution)[0]

            return D.SmoothLossL1(reduction=reduction).to(source.device).forward(source)

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:  # type: ignore
            loss_fn = curr_loss.function
            if loss_fn in [LossFunType.L1_LOSS, LossFunType.KSPACE_L1_LOSS]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l1_loss)
            elif loss_fn in [LossFunType.L2_LOSS, LossFunType.KSPACE_L2_LOSS]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l2_loss)
            elif loss_fn == LossFunType.SSIM_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, ssim_loss)
            elif loss_fn == LossFunType.SSIM_3D_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, ssim_3d_loss)
            elif loss_fn == LossFunType.GRAD_L1_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, grad_l1_loss)
            elif loss_fn == LossFunType.GRAD_L2_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, grad_l2_loss)
            elif loss_fn in [
                LossFunType.NMSE_LOSS,
                LossFunType.KSPACE_NMSE_LOSS,
                LossFunType.DISPLACEMENT_FIELD_NMSE_LOSS,
            ]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nmse_loss)
            elif loss_fn in [
                LossFunType.NRMSE_LOSS,
                LossFunType.KSPACE_NRMSE_LOSS,
                LossFunType.DISPLACEMENT_FIELD_NRMSE_LOSS,
            ]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nrmse_loss)
            elif loss_fn in [
                LossFunType.NMAE_LOSS,
                LossFunType.KSPACE_NMAE_LOSS,
                LossFunType.DISPLACEMENT_FIELD_NMAE_LOSS,
            ]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nmae_loss)
            elif loss_fn == LossFunType.SNR_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, snr_loss)
            elif loss_fn == LossFunType.PSNR_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, psnr_loss)
            elif loss_fn == LossFunType.HFEN_L1_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l1_loss)
            elif loss_fn == LossFunType.HFEN_L2_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l2_loss)
            elif loss_fn == LossFunType.HFEN_L1_NORM_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l1_norm_loss)
            elif loss_fn == LossFunType.HFEN_L2_NORM_LOSS:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l2_norm_loss)
            elif loss_fn in [
                LossFunType.SMOOTH_LOSS_L1,
                LossFunType.KSPACE_SMOOTH_LOSS_L1,
                LossFunType.DISPLACEMENT_FIELD_SMOOTH_LOSS_L1,
            ]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, smooth_loss_l1)
            elif loss_fn in [
                LossFunType.SMOOTH_LOSS_L2,
                LossFunType.KSPACE_SMOOTH_LOSS_L2,
                LossFunType.DISPLACEMENT_FIELD_SMOOTH_LOSS_L2,
            ]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, smooth_loss_l2)
            else:
                raise ValueError(f"{loss_fn} not permissible.")

        return loss_dict

    def compute_sensitivity_map(self, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sensitivity maps :math:`\{S^k\}_{k=1}^{n_c}` if `sensitivity_model` is available.

        :math:`\{S^k\}_{k=1}^{n_c}` are normalized such that

        .. math::
            \sum_{k=1}^{n_c}S^k {S^k}^* = I.

        Parameters
        ----------
        sensitivity_map: torch.Tensor
            Sensitivity maps of shape (batch, coil, height,  width, complex=2).

        Returns
        -------
        sensitivity_map: torch.Tensor
            Normalized and refined sensitivity maps of shape (batch, coil, height,  width, complex=2).
        """

        multicoil = sensitivity_map.shape[self._coil_dim] > 1

        # Pass to sensitivity model only if multiple coils
        if multicoil and ("sensitivity_model" in self.models or "sensitivity_model_3d" in self.models):
            # Move channels to first axis
            sensitivity_map = sensitivity_map.permute(
                (0, 1, 4, 2, 3) if self.ndim == 2 else (0, 1, 5, 2, 3, 4)
            )  # shape (batch, coil, complex=2, height,  width)

            if self.ndim == 2:
                sensitivity_map = self.compute_model_per_coil("sensitivity_model", sensitivity_map)
            else:
                if "sensitivity_model_3d" in self.models:
                    sensitivity_map = self.compute_model_per_coil("sensitivity_model_3d", sensitivity_map)
                else:
                    sensitivity_map = torch.stack(
                        [
                            self.compute_model_per_coil("sensitivity_model", sensitivity_map[:, :, :, _])
                            for _ in range(sensitivity_map.shape[3])
                        ],
                        dim=3,
                    )
            sensitivity_map = sensitivity_map.permute(
                (0, 1, 3, 4, 2) if self.ndim == 2 else (0, 1, 3, 4, 5, 2)
            )  # has channel last: shape (batch, coil, [slice/time], height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch, height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

    def perform_sampling(self, data: dict[str, Any]) -> dict[str, Any]:
        """Performs adaptive sampling.

        Parameters
        ----------
        data: dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.

        Returns
        -------
        dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity
        """
        if "sampling_model" in self.models:
            if "kspace" not in data:
                raise ValueError("Expected data to contain key `kspace`, but not found.")

            sampling_model_kwargs = {"kspace": data["kspace"], "mask": data["sampling_mask"].float()}

            acceleration = data["acceleration"][:, 0]

            sampling_model_kwargs.update(
                filter_arguments_by_signature(
                    self.models["sampling_model"].forward,
                    {
                        "masked_kspace": data["masked_kspace"],
                        "sensitivity_map": data["sensitivity_map"],
                        "acceleration": acceleration,
                    },
                )
            )

            if "padding" in data:
                sampling_model_kwargs.update({"padding": data["padding"].float()})

            masked_kspace, masks, probability_masks = self.models["sampling_model"](**sampling_model_kwargs)

            data["masked_kspace"] = masked_kspace
            data["sampling_mask"] = masks[-1].bool()
            data["masks"] = masks
            data["probability_masks"] = probability_masks

        return data

    @torch.no_grad()
    def reconstruct_volumes(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[dict[str, Callable]] = None,
        regularizer_fns: Optional[dict[str, Callable]] = None,
        add_target: bool = True,
        crop: Optional[str] = None,
    ):
        """Validation process. Assumes that each batch only contains slices of the same volume *AND* that these are
        sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: dict[str, Callable], optional
        regularizer_fns: dict[str, Callable], optional
        add_target: bool
            If true, will add the target to the output
        crop: str, optional
            Crop type.

        Yields
        ------
        (curr_volume, [curr_target,] loss_dict_list, filename): torch.Tensor, [torch.Tensor,], dict, pathlib.Path
        """
        # pylint: disable=too-many-locals, arguments-differ
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Let us inspect this data
        all_filenames = list(data_loader.dataset.volume_indices.keys())  # type: ignore
        num_for_this_process = len(list(data_loader.batch_sampler.sampler.volume_indices.keys()))  # type: ignore
        self.logger.info(
            "Reconstructing a total of %s volumes. This process has %s volumes (world size: %s).",
            len(all_filenames),
            num_for_this_process,
            communication.get_world_size(),
        )

        last_filename = None  # At the start of evaluation, there are no filenames.
        curr_volume = None
        curr_target = None
        if "registration_model" in self.models:
            curr_registration_volume = None
            curr_registration_target = None
        instance_counter = 0
        filenames_seen = 0

        # Loop over dataset. This requires the use of direct.data.sampler.DistributedSequentialSampler as this sampler
        # splits the data over the different processes, and outputs the slices linearly. The implicit assumption here is
        # that the slices are outputted from the Dataset *sequentially* for each volume one by one, and each batch only
        # contains data from one volume.
        time_start = time.time()
        loss_dict_list = []
        # TODO: Use iter_idx to keep track of volume
        for _, data in enumerate(data_loader):
            torch.cuda.empty_cache()
            gc.collect()
            filename = _get_filename_from_batch(data)
            if last_filename is None:
                last_filename = filename  # First iteration last_filename is not set.
            if last_filename != filename:
                curr_volume = None
                curr_target = None
                instance_counter = 0
                last_filename = filename

            scaling_factors = data["scaling_factor"].clone()
            resolution = _compute_resolution(
                key=crop,
                reconstruction_size=data.get("reconstruction_size", None),
            )
            # Compute output
            iteration_output = self._do_iteration(data, loss_fns=loss_fns, regularizer_fns=regularizer_fns)
            output = iteration_output.output_image
            if "registration_model" in self.models:
                output, registered_output = output

            sampling_mask = iteration_output.sampling_mask
            if sampling_mask is not None:
                sampling_mask = sampling_mask.squeeze(-1).float()  # Last dimension is 1 (complex dim)
            loss_dict = iteration_output.data_dict

            # Output can be complex-valued, and has to be cropped. This holds for both output and target.
            output_abs = _process_output(
                output,
                scaling_factors,
                resolution=resolution,
                complex_axis=self._complex_dim,
            )
            if "registration_model" in self.models:
                registered_output_abs = _process_output(
                    registered_output,
                    scaling_factors,
                    resolution=resolution,
                    complex_axis=self._complex_dim,
                )

            if add_target:
                target_abs = _process_output(
                    data["target"],
                    scaling_factors,
                    resolution=resolution,
                    complex_axis=self._complex_dim,
                )

                if "registration_model" in self.models:
                    registration_target_abs = _process_output(
                        data["reference_image"],
                        scaling_factors,
                        resolution=resolution,
                        complex_axis=self._complex_dim,
                    )

            if curr_volume is None:
                volume_size = len(data_loader.batch_sampler.sampler.volume_indices[filename])  # type: ignore
                curr_volume = torch.zeros(*(volume_size, *output_abs.shape[1:]), dtype=output_abs.dtype)

                if "registration_model" in self.models:
                    curr_registration_volume = torch.zeros(
                        *(volume_size, *registered_output_abs.shape[1:]), dtype=registered_output_abs.dtype
                    )

                curr_mask = (
                    torch.zeros(*(volume_size, *sampling_mask.shape[1:]), dtype=sampling_mask.dtype)
                    if sampling_mask is not None
                    else None
                )
                loss_dict_list.append(loss_dict)
                if add_target:
                    curr_target = curr_volume.clone()
                    if "registration_model" in self.models:
                        curr_registration_target = curr_registration_volume.clone()[:, :, 0]

            curr_volume[instance_counter : instance_counter + output_abs.shape[0], ...] = output_abs.cpu()
            if "registration_model" in self.models:
                curr_registration_volume[instance_counter : instance_counter + output_abs.shape[0], ...] = (
                    registered_output_abs.cpu()
                )
            if sampling_mask is not None:
                curr_mask[instance_counter : instance_counter + output_abs.shape[0], ...] = sampling_mask.cpu()
            if add_target:
                curr_target[instance_counter : instance_counter + output_abs.shape[0], ...] = target_abs.cpu()  # type: ignore
                if "registration_model" in self.models:
                    curr_registration_target[instance_counter : instance_counter + output_abs.shape[0], ...] = (
                        registration_target_abs.cpu()
                    )

            instance_counter += output_abs.shape[0]

            # Check if we had the last batch
            if instance_counter == volume_size:
                filenames_seen += 1

                self.logger.info(
                    "%i of %i volumes reconstructed: %s (shape = %s) in %.3fs.",
                    filenames_seen,
                    num_for_this_process,
                    last_filename,
                    list(curr_volume.shape),
                    time.time() - time_start,
                )
                # Maybe not needed.
                del data

                if "registration_model" in self.models:
                    curr_volume = (curr_volume, curr_registration_volume)

                if add_target and "registration_model" in self.models:
                    curr_target = (curr_target, curr_registration_target)

                yield (
                    (curr_volume, curr_target, curr_mask, reduce_list_of_dicts(loss_dict_list), filename)
                    if add_target
                    else (
                        curr_volume,
                        curr_mask,
                        reduce_list_of_dicts(loss_dict_list),
                        filename,
                    )
                )

    @torch.no_grad()
    def reconstruct_and_evaluate(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[dict[str, Callable]] = None,
    ):
        inf_metrics = self.build_metrics(self.cfg.inference.metrics)  # type: ignore
        inf_losses = []
        inf_volume_metrics: dict[PathLike, dict] = defaultdict(dict)

        out = []

        for _, output in enumerate(
            self.reconstruct_volumes(
                data_loader, loss_fns=loss_fns, add_target=True, crop=self.cfg.inference.crop  # type: ignore
            )
        ):
            volume, target, mask, volume_loss_dict, filename = output
            if isinstance(volume, tuple):
                volume, registration_volume = volume
            else:
                registration_volume = None
            if isinstance(target, tuple):
                target, registration_target = target
            else:
                registration_target = None
            if self.ndim == 3:
                # Put slice and time data together
                sc, c, z, x, y = volume.shape
                volume_for_eval = volume.clone().transpose(1, 2).reshape(sc * z, c, x, y)
                if registration_volume is not None:
                    registration_volume_for_eval = registration_volume.clone().transpose(1, 2).reshape(sc * z, c, x, y)
                target_for_eval = target.clone().transpose(1, 2).reshape(sc * z, c, x, y)
                if registration_target is not None:
                    registration_target_for_eval = (
                        registration_target.clone()
                        .unsqueeze(2)
                        .transpose(1, 2)
                        .tile(1, z, 1, 1, 1)
                        .reshape(sc * z, c, x, y)
                    )
            else:
                volume_for_eval = volume.clone()
                target_for_eval = target.clone()
                if registration_volume is not None or registration_target is not None:
                    raise NotImplementedError("Registration not implemented for 2D data.")

            curr_metrics = {
                metric_name: metric_fn(target_for_eval, volume_for_eval).clone().item()
                for metric_name, metric_fn in inf_metrics.items()
            }

            if registration_volume is not None and registration_target is not None:
                curr_metrics.update(
                    {
                        "registration_"
                        + metric_name: metric_fn(registration_target_for_eval, registration_volume_for_eval)
                        .clone()
                        .item()
                        for metric_name, metric_fn in inf_metrics.items()
                    }
                )

            del target, target_for_eval

            curr_metrics_string = ", ".join([f"{x}: {float(y)}" for x, y in curr_metrics.items()])
            self.logger.info("Metrics for %s: %s", filename, curr_metrics_string)

            inf_volume_metrics[filename.name] = curr_metrics
            inf_losses.append(volume_loss_dict)

            out.append((volume, mask, filename))

        # Average loss dict
        loss_dict = reduce_list_of_dicts(inf_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(inf_volume_metrics))

        return out, all_gathered_metrics

    @torch.no_grad()
    def evaluate(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[dict[str, Callable]],
    ):
        """Validation process.

        Assumes that each batch only contains slices of the same volume *AND* that these are sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: dict[str, Callable], optional

        Returns
        -------
        loss_dict, all_gathered_metrics, visualize_slices, visualize_target
        """
        # TODO(jt): visualization should be a namedtuple or a dict or so
        # TODO(gy): Implement visualization of extra keys. E.g. sensitivity_map.
        # pylint: disable=arguments-differ, too-many-locals

        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        volume_metrics = self.build_metrics(self.cfg.validation.metrics)  # type: ignore
        val_losses = []
        val_volume_metrics: dict[PathLike, dict] = defaultdict(dict)

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices: list[np.ndarray] = []
        visualize_mask: list[np.ndarray] = []
        visualize_target: list[np.ndarray] = []
        visualize_registration_slices: list[np.ndarray] | None = None
        visualize_registration_target: list[np.ndarray] | None = None

        for _, output in enumerate(
            self.reconstruct_volumes(
                data_loader, loss_fns=loss_fns, add_target=True, crop=self.cfg.validation.crop  # type: ignore
            )
        ):
            volume, target, mask, volume_loss_dict, filename = output
            if isinstance(volume, tuple):
                volume, registration_volume = volume
            else:
                registration_volume = None
            if isinstance(target, tuple):
                target, registration_target = target
            else:
                registration_target = None

            if self.ndim == 3:
                # Put slice and time data together
                sc, c, z, x, y = volume.shape
                volume_for_eval = volume.clone().transpose(1, 2).reshape(sc * z, c, x, y)
                target_for_eval = target.clone().transpose(1, 2).reshape(sc * z, c, x, y)

                if registration_volume is not None:
                    registration_volume_for_eval = registration_volume.clone().transpose(1, 2).reshape(sc * z, c, x, y)
                    registration_target_for_eval = (
                        registration_target.clone()
                        .unsqueeze(2)
                        .transpose(1, 2)
                        .tile(1, z, 1, 1, 1)
                        .reshape(sc * z, c, x, y)
                    )
            else:
                volume_for_eval = volume.clone()
                target_for_eval = target.clone()
                if registration_volume is not None or registration_target is not None:
                    raise NotImplementedError("Registration not implemented for 2D data.")

            curr_metrics = {
                metric_name: metric_fn(target_for_eval, volume_for_eval).clone()
                for metric_name, metric_fn in volume_metrics.items()
            }
            del volume_for_eval, target_for_eval

            # Calculate image metrics for registered images
            if registration_volume is not None and registration_target is not None:
                curr_metrics.update(
                    {
                        "registration_"
                        + metric_name: metric_fn(registration_target_for_eval, registration_volume_for_eval).clone()
                        for metric_name, metric_fn in volume_metrics.items()
                    }
                )
                del registration_volume_for_eval, registration_target_for_eval

            curr_metrics_string = ", ".join([f"{x}: {float(y)}" for x, y in curr_metrics.items()])
            self.logger.info("Metrics for %s: %s", filename, curr_metrics_string)
            # TODO: Path can be tricky if it is not unique (e.g. image.h5)
            val_volume_metrics[filename.name] = curr_metrics
            val_losses.append(volume_loss_dict)

            # Log the center slice of the volume
            if len(visualize_slices) < self.cfg.logging.tensorboard.num_images:  # type: ignore
                if self.ndim == 3:
                    # If 3D data get every third slice
                    volume = torch.cat([volume[:, :, _] for _ in range(0, z)], dim=2)
                    target = torch.cat([target[:, :, _] for _ in range(0, z)], dim=2)
                    mask = torch.cat([mask[:, :, _] for _ in range(0, mask.shape[2])], dim=2)

                    # Also visualize registration items
                    if registration_volume is not None:
                        if visualize_registration_slices is None:
                            visualize_registration_slices = []
                            visualize_registration_target = []
                        registration_target = torch.cat([registration_target] * registration_volume.shape[2], dim=2)
                        registration_volume = torch.cat(
                            [registration_volume[:, :, _] for _ in range(0, registration_volume.shape[2])], dim=2
                        )

                visualize_slices.append(volume[volume.shape[0] // 2])
                if mask is not None:
                    visualize_mask.append(mask[mask.shape[0] // 2])
                visualize_target.append(target[target.shape[0] // 2])
                if registration_volume is not None:
                    visualize_registration_slices.append(registration_volume[registration_volume.shape[0] // 2])
                    visualize_registration_target.append(registration_target[registration_target.shape[0] // 2])

        # Average loss dict
        loss_dict = reduce_list_of_dicts(val_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        # TODO: Does not work yet with normal gather.
        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(val_volume_metrics))

        if len(visualize_mask) == 0:
            visualize_mask = None
        if visualize_registration_slices is not None:
            visualize_slices = (visualize_slices, visualize_registration_slices)
            visualize_target = (visualize_target, visualize_registration_target)
        return loss_dict, all_gathered_metrics, visualize_slices, visualize_mask, visualize_target

    def compute_model_per_coil(self, model_name: str, data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of model `model_name` in `self.models` per coil.

        Parameters
        ----------
        model_name: str
            Model to run.
        data: torch.Tensor
            Multi-coil data of shape (batch, coil, complex=2, height, width).

        Returns
        -------
        output: torch.Tensor
            Computed output per coil.
        """
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.models[model_name](subselected_data))

        return torch.stack(output, dim=self._coil_dim)

    def compute_loss_on_data(
        self,
        loss_dict: dict[str, torch.Tensor],
        loss_fns: dict[str, Callable],
        data: dict[str, Any],
        output_image: Optional[torch.Tensor] = None,
        output_kspace: Optional[torch.Tensor] = None,
        output_displacement_field: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_kspace: Optional[torch.Tensor] = None,
        target_displacement_field: Optional[torch.Tensor] = None,
        weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if output_image is None and output_kspace is None and output_displacement_field is None:
            raise ValueError(
                "Inputs for `output_image`, `output_kspace` and `output_displacement_field` are all None."
            )
        for key, value in loss_dict.items():
            if "kspace" in key:
                if output_kspace is not None:
                    output = output_kspace
                    target = data["kspace"] if target_kspace is None else target_kspace
                    reconstruction_size = None
                else:
                    continue
            elif "displacement_field" in key:
                if output_displacement_field is not None:
                    output = output_displacement_field
                    target = (
                        data["displacement_field"] if target_displacement_field is None else target_displacement_field
                    )
                    reconstruction_size = data.get("reconstruction_size", None)
                else:
                    continue
            else:
                if output_image is not None:
                    output, target, reconstruction_size = (
                        output_image,
                        data["target"] if target_image is None else target_image,
                        data.get("reconstruction_size", None),
                    )
                else:
                    continue
            loss_dict[key] = value + weight * loss_fns[key](output, target, "mean", reconstruction_size)
        return loss_dict

    def do_registration(self, data: dict[str, Any], moving_image) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs registration.

        The registration model is expected to be in `self.models`. The registration model
        should take the moving image and the reference image as input and return the registered image and the
        displacement field.


        Parameters
        ----------
        data: dict[str, Any]
            Data dictionary containing the reference image.
        moving_image: torch.Tensor
            Moving image of shape (batch, height, width).

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Registered image and displacement field of shape (batch, height, width) and (batch, 2, height, width).
        """

        reference_image = data["reference_image"]
        registered_image, displacement_field = self.models["registration_model"](moving_image, reference_image)

        return registered_image, displacement_field

    def _forward_operator(
        self, image: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward operator of multi-coil accelerated MRI.

        This will apply the expand operator, compute the k-space by applying the forward Fourier transform,
        and apply the sampling mask.

        Parameters
        ----------
        image: torch.Tensor
            Image tensor of shape (batch, time/slice, height, width, [complex=2]).
        sensitivity_map: torch.Tensor
            Sensitivity map tensor of shape (batch, coil, time/slice, height, width, [complex=2]).
        sampling_mask: torch.Tensor
            Sampling mask tensor of shape (batch, time/slice or 1, height, width, 1).

        Returns
        -------
        torch.Tensor
            k-space tensor of shape (batch, coil, time/slice, height, width, [complex=2]).
        """
        return T.apply_mask(
            self.forward_operator(
                T.expand_operator(image, sensitivity_map, dim=self._coil_dim),
                dim=self._spatial_dims,
            ),
            sampling_mask,
            return_mask=False,
        )

    def _backward_operator(
        self, kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor
    ) -> torch.Tensor:
        """Backward operator of multi-coil accelerated MRI.

        This will apply the sampling mask, compute the image by applying the adjoint Fourier transform,
        and apply the reduce operator using the sensitivity map.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space tensor of shape (batch, coil, time/slice, height, width, [complex=2]).
        sensitivity_map: torch.Tensor
            Sensitivity map tensor of shape (batch, coil, time/slice, height, width, [complex=2]).
        sampling_mask: torch.Tensor
            Sampling mask tensor of shape (batch, time/slice or 1, height, width, 1).

        Returns
        -------
        torch.Tensor
            Image tensor of shape (batch, time/slice, height, width, [complex=2]).
        """
        return T.reduce_operator(
            self.backward_operator(T.apply_mask(kspace, sampling_mask, return_mask=False), dim=self._spatial_dims),
            sensitivity_map,
            dim=self._coil_dim,
        )


def _crop_volume(*tensors: torch.Tensor, resolution: Union[list[int], tuple[int, ...]]) -> tuple[torch.Tensor, ...]:
    """Crops the spatial dimensions of multiple tensors.

    Parameters
    ----------
    tensors: torch.Tensor
        A variable number of tensors, each with shape (batch, height, width).
    resolution: list of ints or tuple of ints
        Target resolution for cropping.

    Returns
    -------
    tuple of torch.Tensor
        Cropped tensors, each with an added channel dimension.
    """
    if not resolution or all(_ == 0 for _ in resolution):
        return tuple(tensor.unsqueeze(1) for tensor in tensors)  # Add channel dimension

    # Apply cropping and add channel dimension
    cropped_tensors = [T.center_crop(tensor, resolution).unsqueeze(1) for tensor in tensors]

    return tuple(cropped_tensors)


def _reduce_slice_dim(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Combines batch and slice dimensions for a variable number of input tensors.

    Batch and slice dimensions are assumed to be on the first and second axes of each tensor: `b, s = tensor.shape[:2]`.

    Parameters
    ----------
    tensors: torch.Tensor
        A variable number of tensors, all with shape (batch, slice, *).

    Returns
    -------
    tuple of torch.Tensor
        Each tensor will have shape (batch * slice, *).
    """
    shape = tensors[0].shape

    if len(tensors) > 1:
        # Check that all tensors have the same shape
        assert all(tensor.shape == shape for tensor in tensors), "All tensors must have the same shape."

    b, s = shape[:2]

    # Reshape each tensor
    reshaped_tensors = [tensor.reshape(b * s, *shape[2:]) for tensor in tensors]

    return tuple(reshaped_tensors)


def _process_output(
    data: torch.Tensor,
    scaling_factors: Optional[torch.Tensor] = None,
    resolution: Optional[Union[list[int], tuple[int]]] = None,
    complex_axis: Optional[int] = -1,
) -> torch.Tensor:
    """Crops and scales input tensor.

    Parameters
    ----------
    data: torch.Tensor
    scaling_factors: Optional[torch.Tensor]
        Scaling factor. Default: None.
    resolution: Optional[Union[list[int], tuple[int]]]
        Resolution. Default: None.
    complex_axis: Optional[int]
        Dimension along which modulus of `data` will be computed (if it's complex). Default: -1 (last).

    Returns
    -------
    torch.Tensor
    """
    # data is of shape (batch, complex=2, height, width)
    if scaling_factors is not None:
        data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(data.device)

    data = T.modulus_if_complex(data, complex_axis=complex_axis)

    if len(data.shape) in [3, 4]:  # (batch, height, width)
        data = data.unsqueeze(1)  # Added channel dimension.

    if resolution is not None:
        data = T.center_crop(data, resolution).contiguous()

    return data


def _compute_resolution(
    key: Optional[str], reconstruction_size: Optional[Union[list[int], tuple[int]]] = None
) -> Union[list[int], None]:
    """Computes resolution.

    Parameters
    ----------
    key: str
        Can be `header` or None.
    reconstruction_size: Optional[Union[list[int], tuple[int]]]
        Reconstruction size. Default: None.

    Returns
    -------
    resolution: Union[str, list[int], None]
        Resolution of reconstruction.
    """

    if key == "header":
        # This will be of the form [tensor(x_0, x_1, ...), tensor(y_0, y_1,...), tensor(z_0, z_1, ...)] over
        # batches.
        resolution = [_.detach().cpu().numpy().tolist() for _ in reconstruction_size]  # type: ignore
        # The volume sampler should give validation indices belonging to the *same* volume, so it should be
        # safe taking the first element, the matrix size are in x,y,z (we work in z,x,y).
        resolution = [_[0] for _ in resolution][:-1]
        return resolution
    elif not key:
        return None
    else:
        raise ValueError("Cropping should be either set to `header` to get the values from the header or None.")


def _get_filename_from_batch(data: dict) -> pathlib.Path:
    filenames = data["filename"]
    if len(set(filenames)) != 1:
        raise ValueError(
            f"Expected a batch during validation to only contain filenames of one case. " f"Got {set(filenames)}."
        )
    # This can be fixed when there is a custom collate_fn
    return pathlib.Path(filenames[0])
