# coding=utf-8
# Copyright (c) DIRECT Contributors

import time
from collections import defaultdict
from os import PathLike
from typing import Callable, DefaultDict, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput, Engine
from direct.functionals import SSIMLoss
from direct.utils import (
    communication,
    detach_dict,
    dict_to_device,
    merge_list_of_dicts,
    multiply_function,
    reduce_list_of_dicts,
)
from direct.utils.communication import reduce_tensor_dict


class RecurrentVarNetEngine(Engine):
    """Recurrent Variational Network Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: int,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

        self._complex_dim = -1
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def _do_iteration(
        self,
        data: Dict[str, torch.Tensor],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:

        # loss_fns can be done, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        loss_dicts = []
        regularizer_dicts = []

        data = dict_to_device(data, self.device)

        # sensitivity_map of shape (batch, coil, height,  width, complex=2)
        sensitivity_map = data["sensitivity_map"]

        if "sensitivity_model" in self.models:  # SER Module

            # Move channels to first axis
            sensitivity_map = data["sensitivity_map"].permute(
                (0, 1, 4, 2, 3)
            )  # shape (batch, coil, complex=2, height,  width)

            sensitivity_map = self.compute_model_per_coil("sensitivity_model", sensitivity_map).permute(
                (0, 1, 3, 4, 2)
            )  # has channel last: shape (batch, coil, height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map ** 2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch,  height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(1).unsqueeze(-1)
        data["sensitivity_map"] = T.safe_divide(sensitivity_map, sensitivity_map_norm)

        with autocast(enabled=self.mixed_precision):

            output_kspace = self.model(
                masked_kspace=data["masked_kspace"],
                sampling_mask=data["sampling_mask"],
                sensitivity_map=data["sensitivity_map"],
            )

            output_image = T.root_sum_of_squares(
                self.backward_operator(output_kspace, dim=self._spatial_dims),
                dim=self._coil_dim,
            )  # shape (batch, height,  width)

            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
            regularizer_dict = {
                k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
            }

            for key, value in loss_dict.items():
                loss_dict[key] = value + loss_fns[key](
                    output_image,
                    **data,
                    reduction="mean",
                )

            for key, value in regularizer_dict.items():
                regularizer_dict[key] = value + regularizer_fns[key](
                    output_image,
                    **data,
                )

            loss = sum(loss_dict.values()) + sum(regularizer_dict.values())

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dicts.append(detach_dict(loss_dict))
        regularizer_dicts.append(
            detach_dict(regularizer_dict)
        )  # Need to detach dict as this is only used for logging.

        # Add the loss dicts.
        loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum")
        regularizer_dict = reduce_list_of_dicts(regularizer_dicts, mode="sum")

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict},
        )

    def build_loss(self, **kwargs) -> Dict:
        def get_resolution(**data):
            """Be careful that this will use the cropping size of the FIRST sample in the batch."""
            return self.compute_resolution(self.cfg.training.loss.crop, data.get("reconstruction_size", None))

        def l1_loss(source, reduction="mean", **data):
            """Calculate L1 loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, complex=2, height, width)
            Data: torch.Tensor
                Contains key "target" with value a tensor of shape (batch, height, width)
            """
            resolution = get_resolution(**data)
            l1_loss = F.l1_loss(*self.cropper(source, data["target"], resolution), reduction=reduction)

            return l1_loss

        def l2_loss(source, reduction="mean", **data):
            """Calculate L2 loss (MSE) given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, complex=2, height, width)
            Data: torch.Tensor
                Contains key "target" with value a tensor of shape (batch, height, width)
            """
            resolution = get_resolution(**data)
            l2_loss = F.mse_loss(*self.cropper(source, data["target"], resolution), reduction=reduction)

            return l2_loss

        def ssim_loss(source, reduction="mean", **data):
            """Calculate SSIM loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, complex=2, height, width)
            Data: torch.Tensor
                Contains key "target" with value a tensor of shape (batch, height, width)
            """
            resolution = get_resolution(**data)
            if reduction != "mean":
                raise AssertionError(
                    f"SSIM loss can only be computed with reduction == 'mean'." f" Got reduction == {reduction}."
                )

            source_abs, target_abs = self.cropper(source, data["target"], resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = SSIMLoss().to(source_abs.device).forward(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:  # type: ignore
            loss_fn = curr_loss.function
            if loss_fn == "l1_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l1_loss)
            elif loss_fn == "l2_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l2_loss)
            elif loss_fn == "ssim_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, ssim_loss)
            else:
                raise ValueError(f"{loss_fn} not permissible.")

        return loss_dict

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]],
        regularizer_fns: Optional[Dict[str, Callable]] = None,
        crop: Optional[str] = None,
        is_validation_process: bool = True,
    ):
        """Validation process. Assumes that each batch only contains slices of the same volume *AND* that these are
        sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: Dict[str, Callable], optional
        regularizer_fns: Dict[str, Callable], optional
        crop: str, optional
        is_validation_process: bool

        Returns
        -------
        loss_dict, all_gathered_metrics, visualize_slices, visualize_target
        """
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Variables required for evaluation.
        volume_metrics = self.build_metrics(self.cfg.validation.metrics)  # type: ignore

        # filenames can be in the volume_indices attribute of the dataset
        num_for_this_process = None
        all_filenames = None
        if hasattr(data_loader.dataset, "volume_indices"):
            all_filenames = list(data_loader.dataset.volume_indices.keys())
            num_for_this_process = len(list(data_loader.batch_sampler.sampler.volume_indices.keys()))
            self.logger.info(
                f"Reconstructing a total of {len(all_filenames)} volumes. "
                f"This process has {num_for_this_process} volumes (world size: {communication.get_world_size()})."
            )

        filenames_seen = 0
        reconstruction_output: DefaultDict = defaultdict(list)
        if is_validation_process:
            targets_output: DefaultDict = defaultdict(list)
        val_losses = []
        val_volume_metrics: Dict[PathLike, Dict] = defaultdict(dict)
        last_filename = None

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices: List[np.ndarray] = []
        visualize_target: List[np.ndarray] = []
        # visualizations = {}

        extra_visualization_keys = (
            self.cfg.logging.log_as_image if self.cfg.logging.log_as_image else []  # type: ignore
        )

        # Loop over dataset. This requires the use of direct.data.sampler.DistributedSequentialSampler as this sampler
        # splits the data over the different processes, and outputs the slices linearly. The implicit assumption here is
        # that the slices are outputted from the Dataset *sequentially* for each volume one by one, and each batch only
        # contains data from one volume.
        time_start = time.time()

        for iter_idx, data in enumerate(data_loader):
            filenames = data.pop("filename")
            if len(set(filenames)) != 1:
                raise ValueError(
                    f"Expected a batch during validation to only contain filenames of one case. "
                    f"Got {set(filenames)}."
                )

            slice_nos = data.pop("slice_no")
            scaling_factors = data["scaling_factor"]

            resolution = self.compute_resolution(
                key=self.cfg.validation.crop,  # type: ignore
                reconstruction_size=data.get("reconstruction_size", None),
            )

            # Compute output and loss.
            iteration_output = self._do_iteration(data, loss_fns, regularizer_fns=regularizer_fns)
            output = iteration_output.output_image
            loss_dict = iteration_output.data_dict

            loss_dict = detach_dict(loss_dict)
            output = output.detach()
            val_losses.append(loss_dict)

            # Output is complex-valued, and has to be cropped. This holds for both output and target.
            # Output has shape (batch, complex, height, width)
            output_abs = self.process_output(
                output,
                scaling_factors,
                resolution=resolution,
            )

            if is_validation_process:
                # Target has shape (batch,  height, width)
                target_abs = self.process_output(
                    data["target"].detach(),
                    scaling_factors,
                    resolution=resolution,
                )
                for key in extra_visualization_keys:
                    curr_data = data[key].detach()
                    # Here we need to discover which keys are actually normalized or not
                    # this requires a solution to issue #23: https://github.com/NKI-AI/direct/issues/23

            del output  # Explicitly call delete to clear memory.

            # Aggregate volumes to be able to compute the metrics on complete volumes.
            for idx, filename in enumerate(filenames):
                if last_filename is None:
                    last_filename = filename  # First iteration last_filename is not set.

                curr_slice = output_abs[idx].detach()
                slice_no = int(slice_nos[idx].numpy())

                reconstruction_output[filename].append((slice_no, curr_slice.cpu()))

                if is_validation_process:
                    targets_output[filename].append((slice_no, target_abs[idx].cpu()))

                is_last_element_of_last_batch = iter_idx + 1 == len(data_loader) and idx + 1 == len(data["target"])
                reconstruction_conditions = [
                    filename != last_filename,
                    is_last_element_of_last_batch,
                ]
                for condition in reconstruction_conditions:
                    if condition:
                        filenames_seen += 1

                        # Now we can ditch the reconstruction dict by reconstructing the volume,
                        # will take too much memory otherwise.
                        volume = torch.stack([_[1] for _ in reconstruction_output[last_filename]])
                        if is_validation_process:
                            target = torch.stack([_[1] for _ in targets_output[last_filename]])
                            curr_metrics = {
                                metric_name: metric_fn(target, volume)
                                for metric_name, metric_fn in volume_metrics.items()
                            }
                            val_volume_metrics[last_filename] = curr_metrics
                            # Log the center slice of the volume
                            if len(visualize_slices) < self.cfg.logging.tensorboard.num_images:  # type: ignore
                                visualize_slices.append(volume[volume.shape[0] // 2])
                                visualize_target.append(target[target.shape[0] // 2])

                            # Delete outputs from memory, and recreate dictionary.
                            # This is not needed when not in validation as we are actually interested
                            # in the iteration output.
                            del targets_output[last_filename]
                            del reconstruction_output[last_filename]

                        if all_filenames:
                            log_prefix = f"{filenames_seen} of {num_for_this_process} volumes reconstructed:"
                        else:
                            log_prefix = f"{iter_idx + 1} of {len(data_loader)} slices reconstructed:"

                        self.logger.info(
                            f"{log_prefix} {last_filename}"
                            f" (shape = {list(volume.shape)}) in {time.time() - time_start:.3f}s."
                        )
                        # restart timer
                        time_start = time.time()
                        last_filename = filename

        # Average loss dict
        loss_dict = reduce_list_of_dicts(val_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(val_volume_metrics))
        if not is_validation_process:
            return loss_dict, reconstruction_output

        return loss_dict, all_gathered_metrics, visualize_slices, visualize_target

    def process_output(self, data, scaling_factors=None, resolution=None):
        # data is of shape (batch, complex=2, height, width)
        if scaling_factors is not None:
            data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(data.device)

        data = T.modulus_if_complex(data)

        if len(data.shape) == 3:  # (batch, height, width)
            data = data.unsqueeze(1)  # Added channel dimension.

        if resolution is not None:
            data = T.center_crop(data, resolution).contiguous()

        return data

    @staticmethod
    def compute_resolution(key, reconstruction_size):
        if key == "header":
            # This will be of the form [tensor(x_0, x_1, ...), tensor(y_0, y_1,...), tensor(z_0, z_1, ...)] over
            # batches.
            resolution = [_.detach().cpu().numpy().tolist() for _ in reconstruction_size]
            # The volume sampler should give validation indices belonging to the *same* volume, so it should be
            # safe taking the first element, the matrix size are in x,y,z (we work in z,x,y).
            resolution = [_[0] for _ in resolution][:-1]
        elif key == "training":
            resolution = key
        elif not key:
            resolution = None
        else:
            raise ValueError(
                "Cropping should be either set to `header` to get the values from the header or "
                "`training` to take the same value as training."
            )
        return resolution

    def cropper(self, source, target, resolution):
        """2D source/target cropper.

        Parameters
        ----------
        source: torch.Tensor
            Has shape (batch, height, width)
        target: torch.Tensor
            Has shape (batch, height, width)
        """

        if not resolution or all(_ == 0 for _ in resolution):
            return source.unsqueeze(1), target.unsqueeze(1)  # Added channel dimension.

        source_abs = T.center_crop(source, resolution).unsqueeze(1)  # Added channel dimension.
        target_abs = T.center_crop(target, resolution).unsqueeze(1)  # Added channel dimension.

        return source_abs, target_abs

    def compute_model_per_coil(self, model_name, data):
        """Computes model per coil."""
        # data is of shape (batch, coil, complex=2, height, width)
        output = []

        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.models[model_name](subselected_data))
        output = torch.stack(output, dim=self._coil_dim)

        # output is of shape (batch, coil, complex=2, height, width)
        return output
