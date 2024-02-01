# coding=utf-8
# Copyright (c) DIRECT Contributors

"""MRI model engine of DIRECT."""
from __future__ import annotations

import gc
import pathlib
import time
from collections import defaultdict
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    def forward_function(self, data: Dict[str, Any]) -> Tuple[TensorOrNone, TensorOrNone]:
        """This method performs the model's forward method given `data` which contains all tensor inputs.

        Must be implemented by child classes.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : Dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[Dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[Dict[str, Callable]]
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
            data["sampling_mask"] = data["sampling_mask"].int()
            data = self.perform_sampling(data)

            output_image, output_kspace = self.forward_function(data)
            output_image = T.modulus_if_complex(output_image, complex_axis=self._complex_dim)

            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
            regularizer_dict = {
                k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
            }
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
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
            data_dict={**loss_dict, **regularizer_dict},
        )

    def build_loss(self) -> Dict:
        def get_resolution(reconstruction_size):
            return _compute_resolution(self.cfg.training.loss.crop, reconstruction_size)  # type: ignore

        def nmae_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
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
                source, target = _crop_volume(source, target, resolution)

            nmae_loss = D.NMAELoss(reduction=reduction).forward(source, target)

            return nmae_loss

        def nmse_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
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
                source, target = _crop_volume(source, target, resolution)
            nmse_loss = D.NMSELoss(reduction=reduction).forward(source, target)

            return nmse_loss

        def nrmse_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
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
                source, target = _crop_volume(source, target, resolution)
            nrmse_loss = D.NRMSELoss(reduction=reduction).forward(source, target)

            return nrmse_loss

        def l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
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
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            l1_loss: torch.Tensor
                L1 loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)
            l1_loss = F.l1_loss(source, target, reduction=reduction)

            return l1_loss

        def l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
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
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            l2_loss: torch.Tensor
                L2 loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)
            l2_loss = F.mse_loss(source, target, reduction=reduction)

            return l2_loss

        def ssim_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate SSIM loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
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
            source_abs, target_abs = _crop_volume(source, target, resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = D.SSIMLoss().to(source_abs.device).forward(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        def ssim_3d_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
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
            reconstruction_size: Optional[Tuple]
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
            source_abs, target_abs = _crop_volume(source, target, resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = D.SSIM3DLoss().to(source_abs.device).forward(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        def grad_l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate Sobel gradient L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            grad_loss: torch.Tensor
                Sobel grad L1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)
            grad_l1_loss = D.SobelGradL1Loss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return grad_l1_loss

        def grad_l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate Sobel gradient L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            grad_loss: torch.Tensor
                Sobel grad L1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)
            grad_l2_loss = D.SobelGradL2Loss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return grad_l2_loss

        def psnr_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate peak signal-to-noise ratio loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            psnr_loss: torch.Tensor
               PSNR loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)
            psnr_loss = -D.PSNRLoss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return psnr_loss

        def snr_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate signal-to-noise loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            snr_loss: torch.Tensor
                SNR loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)
            snr_loss = -D.SNRLoss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return snr_loss

        def hfen_l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                HFEN l1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)

            return D.HFENL1Loss(reduction=reduction, norm=False).to(source_abs.device).forward(source_abs, target_abs)

        def hfen_l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                HFEN l2 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)

            return D.HFENL2Loss(reduction=reduction, norm=False).to(source_abs.device).forward(source_abs, target_abs)

        def hfen_l1_norm_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                Normalized HFEN l1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)

            return D.HFENL1Loss(reduction=reduction, norm=True).to(source_abs.device).forward(source_abs, target_abs)

        def hfen_l2_norm_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate normalized HFEN L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, [slice], height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, [slice], height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            torch.Tensor
                Normalized HFEN l2 loss.
            """
            resolution = get_resolution(reconstruction_size)
            if self.ndim == 3:
                source, target = _reduce_slice_dim(source, target)
            source_abs, target_abs = _crop_volume(source, target, resolution)

            return D.HFENL2Loss(reduction=reduction, norm=True).to(source_abs.device).forward(source_abs, target_abs)

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:  # type: ignore
            loss_fn = curr_loss.function
            if loss_fn in ["l1_loss", "kspace_l1_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l1_loss)
            elif loss_fn in ["l2_loss", "kspace_l2_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l2_loss)
            elif loss_fn == "ssim_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, ssim_loss)
            elif loss_fn == "ssim_3d_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, ssim_3d_loss)
            elif loss_fn == "grad_l1_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, grad_l1_loss)
            elif loss_fn == "grad_l2_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, grad_l2_loss)
            elif loss_fn in ["nmse_loss", "kspace_nmse_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nmse_loss)
            elif loss_fn in ["nrmse_loss", "kspace_nrmse_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nrmse_loss)
            elif loss_fn in ["nmae_loss", "kspace_nmae_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nmae_loss)
            elif loss_fn in ["snr_loss", "psnr_loss"]:
                loss_dict[loss_fn] = multiply_function(
                    curr_loss.multiplier, (snr_loss if loss_fn == "snr" else psnr_loss)
                )
            elif loss_fn == "hfen_l1_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l1_loss)
            elif loss_fn == "hfen_l2_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l2_loss)
            elif loss_fn == "hfen_l1_norm_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l1_norm_loss)
            elif loss_fn == "hfen_l2_norm_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, hfen_l2_norm_loss)
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
            )  # has channel last: shape (batch, coil, [slice], height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch, height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

    def perform_sampling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "sampling_model" in self.models:
            if "kspace" not in data:
                raise ValueError("Expected data to contain key `kspace`, but not found.")
            sampling_model_kwargs = {"kspace": data["kspace"], "mask": data["sampling_mask"]}

            acceleration = data["acceleration"]
            center_fraction = data["center_fraction"]

            if acceleration.dim > 1:
                acceleration = acceleration[:, 0]
                center_fraction = center_fraction[:, 0]

            sampling_model_kwargs.update(
                filter_arguments_by_signature(
                    self.models["sampling_model"].forward,
                    {
                        "masked_kspace": data["masked_kspace"],
                        "sensitivity_map": data["sensitivity_map"],
                        "acceleration": acceleration,
                        "center_fraction": center_fraction,
                    },
                )
            )
            if "padding" in data:
                sampling_model_kwargs.update({"padding": data["padding"]})
            masked_kspace, masks, probability_masks = self.models["sampling_model"](**sampling_model_kwargs)

            data["masked_kspace"] = masked_kspace
            data["sampling_mask"] = masks[-1]
            data["masks"] = masks
            data["probability_masks"] = probability_masks

        return data

    @torch.no_grad()
    def reconstruct_volumes(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
        add_target: bool = True,
        crop: Optional[str] = None,
    ):
        """Validation process. Assumes that each batch only contains slices of the same volume *AND* that these are
        sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: Dict[str, Callable], optional
        regularizer_fns: Dict[str, Callable], optional
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
            loss_dict = iteration_output.data_dict

            # Output can be complex-valued, and has to be cropped. This holds for both output and target.
            output_abs = _process_output(
                output,
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

            if curr_volume is None:
                instance_size = len(data_loader.batch_sampler.sampler.volume_indices[filename])  # type: ignore
                curr_volume = []
                loss_dict_list.append(loss_dict)
                if add_target:
                    curr_target = []

            curr_volume.append(output_abs.cpu())
            if add_target:
                curr_target.append(target_abs.cpu())

            instance_counter += output_abs.shape[0]

            # Check if we had the last batch
            if instance_counter == instance_size:
                filenames_seen += 1
                if self.ndim == 2:
                    cat_dim = 0  # batch dim
                else:
                    # Sometimes due to memory constraints 3D samples may be split in
                    # (1, coils, num_slices, height, width) and num_slices may vary on last batch, so concatenate
                    # across slice dimension as batch size is assumed 1 for 3D samples
                    # In this case data contains slice indexes instead of slice number
                    if isinstance(
                        next(_ for _ in data_loader.dataset.data if _[0] == filename)[1], tuple  # type: ignore
                    ):
                        cat_dim = 2
                    # Otherwise 3D samples may be part of 4D data (e.g. 3D + time) and batch size contains samples
                    # of same time-step (2D+time) or same slice (3D), so concatenate across batch dimension.
                    else:
                        cat_dim = 0
                curr_volume = torch.cat(curr_volume, cat_dim)
                if add_target:
                    curr_target = torch.cat(curr_target, cat_dim)
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

                sampling_mask = iteration_output.sampling_mask
                sampling_mask = sampling_mask.squeeze().cpu()
                yield (
                    curr_volume,
                    curr_target,
                    sampling_mask,
                    reduce_list_of_dicts(loss_dict_list),
                    filename,
                ) if add_target else (
                    curr_volume,
                    sampling_mask,
                    reduce_list_of_dicts(loss_dict_list),
                    filename,
                )

    @torch.no_grad()
    def evaluate(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]],
    ):
        """Validation process.

        Assumes that each batch only contains slices of the same volume *AND* that these are sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: Dict[str, Callable], optional

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
        val_volume_metrics: Dict[PathLike, Dict] = defaultdict(dict)

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices: List[torch.Tensor] = []
        visualize_target: List[torch.Tensor] = []
        visualize_mask: List[torch.Tensor] = []

        for _, output in enumerate(
            self.reconstruct_volumes(
                data_loader, loss_fns=loss_fns, add_target=True, crop=self.cfg.validation.crop  # type: ignore
            )
        ):
            volume, target, mask, volume_loss_dict, filename = output
            if self.ndim == 3:
                # Put slice and time data together
                sc, c, z, x, y = volume.shape
                volume_for_eval = volume.clone().transpose(1, 2).reshape(sc * z, c, x, y)
                target_for_eval = target.clone().transpose(1, 2).reshape(sc * z, c, x, y)
            else:
                volume_for_eval = volume.clone()
                target_for_eval = target.clone()

            curr_metrics = {
                metric_name: metric_fn(target_for_eval, volume_for_eval).clone()
                for metric_name, metric_fn in volume_metrics.items()
            }
            del volume_for_eval, target_for_eval

            curr_metrics_string = ", ".join([f"{x}: {float(y)}" for x, y in curr_metrics.items()])
            self.logger.info("Metrics for %s: %s", filename, curr_metrics_string)
            # TODO: Path can be tricky if it is not unique (e.g. image.h5)
            val_volume_metrics[filename.name] = curr_metrics
            val_losses.append(volume_loss_dict)

            # Log the center slice of the volume
            if len(visualize_slices) < self.cfg.logging.tensorboard.num_images:  # type: ignore
                if self.ndim == 3:
                    # If 3D data visualize every third slice
                    volume = torch.cat([volume[:, :, _] for _ in range(0, z, 3)][:6], dim=2)
                    target = torch.cat([target[:, :, _] for _ in range(0, z, 3)][:6], dim=2)
                    if mask.ndim > 2:
                        # In case of dynamic masks
                        mask = torch.cat([mask[_] for _ in range(0, mask.shape[0], 3)][:6], dim=0)
                visualize_slices.append(volume[volume.shape[0] // 2])
                visualize_target.append(target[target.shape[0] // 2])
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                visualize_mask.append(mask[mask.shape[0] // 2][None])

        # Average loss dict
        loss_dict = reduce_list_of_dicts(val_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        # TODO: Does not work yet with normal gather.
        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(val_volume_metrics))
        return loss_dict, all_gathered_metrics, visualize_slices, visualize_target, visualize_mask

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
        loss_dict: Dict[str, torch.Tensor],
        loss_fns: Dict[str, Callable],
        data: Dict[str, Any],
        output_image: Optional[torch.Tensor] = None,
        output_kspace: Optional[torch.Tensor] = None,
        weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if output_image is None and output_kspace is None:
            raise ValueError("Inputs for `output_image` and `output_kspace` cannot be both None.")
        for key, value in loss_dict.items():
            if "kspace" in key:
                if output_kspace is not None:
                    output, target, reconstruction_size = output_kspace, data["kspace"], None
                else:
                    continue
            else:
                if output_image is not None:
                    output, target, reconstruction_size = (
                        output_image,
                        data["target"],
                        data.get("reconstruction_size", None),
                    )
                else:
                    continue
            loss_dict[key] = value + weight * loss_fns[key](output, target, "mean", reconstruction_size)
        return loss_dict

    def _forward_operator(self, image, sensitivity_map, sampling_mask):
        return T.apply_mask(
            self.forward_operator(
                T.expand_operator(image, sensitivity_map, dim=self._coil_dim),
                dim=self._spatial_dims,
            ),
            sampling_mask,
            return_mask=False,
        )

    def _backward_operator(self, kspace, sensitivity_map, sampling_mask):
        return T.reduce_operator(
            self.backward_operator(T.apply_mask(kspace, sampling_mask, return_mask=False), dim=self._spatial_dims),
            sensitivity_map,
            dim=self._coil_dim,
        )


def _crop_volume(
    source: torch.Tensor, target: torch.Tensor, resolution: Union[List[int], Tuple[int, ...]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D source/target cropper.

    Parameters
    ----------
    source: torch.Tensor
        Has shape (batch, height, width)
    target: torch.Tensor
        Has shape (batch, height, width)
    resolution: list of ints or tuple of ints
        Target resolution.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
    """

    if not resolution or all(_ == 0 for _ in resolution):
        return source.unsqueeze(1), target.unsqueeze(1)  # Added channel dimension.

    source_abs = T.center_crop(source, resolution).unsqueeze(1)  # Added channel dimension.
    target_abs = T.center_crop(target, resolution).unsqueeze(1)  # Added channel dimension.

    return source_abs, target_abs


def _reduce_slice_dim(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """This will combine batch and slice dims, for source and target tensors.

    Batch and slice dimensions are assumed to be on first and second axes: `b, c = source.shape[:2]`.

    Parameters
    ----------
    source: torch.Tensor
        Has shape (batch, slice, *).
    target: torch.Tensor
        Has shape (batch, slice, *).

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        Have shape (batch * slice, *).
    """
    assert source.shape == target.shape
    shape = source.shape
    b, s = shape[:2]
    source = source.reshape(b * s, *shape[2:])
    target = target.reshape(b * s, *shape[2:])
    return source, target


def _process_output(
    data: torch.Tensor,
    scaling_factors: Optional[torch.Tensor] = None,
    resolution: Optional[Union[List[int], Tuple[int]]] = None,
    complex_axis: Optional[int] = -1,
) -> torch.Tensor:
    """Crops and scales input tensor.

    Parameters
    ----------
    data: torch.Tensor
    scaling_factors: Optional[torch.Tensor]
        Scaling factor. Default: None.
    resolution: Optional[Union[List[int], Tuple[int]]]
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
    key: Optional[str], reconstruction_size: Optional[Union[List[int], Tuple[int]]] = None
) -> Union[List[int], None]:
    """Computes resolution.

    Parameters
    ----------
    key: str
        Can be `header` or None.
    reconstruction_size: Optional[Union[List[int], Tuple[int]]]
        Reconstruction size. Default: None.

    Returns
    -------
    resolution: Union[str, List[int], None]
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
