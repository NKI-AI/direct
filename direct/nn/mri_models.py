# coding=utf-8
# Copyright (c) DIRECT Contributors

import time
from collections import defaultdict
from os import PathLike
from typing import Callable, DefaultDict, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput, Engine
from direct.functionals import SSIMLoss
from direct.utils import communication, detach_dict, merge_list_of_dicts, multiply_function, reduce_list_of_dicts
from direct.utils.communication import reduce_tensor_dict


class MRIModelEngine(Engine):
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

    def _do_iteration(
        self,
        data: Dict[str, Union[List, torch.Tensor]],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        pass

    def build_loss(self, **kwargs) -> Dict:
        # TODO: Cropper is a processing output tool.
        def get_resolution(**data):
            """Be careful that this will use the cropping size of the FIRST sample in the batch."""
            return _compute_resolution(self.cfg.training.loss.crop, data.get("reconstruction_size", None))

        # TODO(jt) Ideally this is also configurable:
        # - Do in steps (use insertation order)
        # Crop -> then loss.

        def l1_loss(source, reduction="mean", **data):
            """Calculate L1 loss given source and target.

            Parameters
            ----------
            Source:  shape (batch, complex=2, height, width)
            Data: Contains key "target" with value a tensor of shape (batch, height, width)
            """
            resolution = get_resolution(**data)
            l1_loss = F.l1_loss(
                *self.cropper(T.modulus_if_complex(source), data["target"], resolution), reduction=reduction
            )

            return l1_loss

        def l2_loss(source, reduction="mean", **data):
            """Calculate L2 loss (MSE) given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, complex=2, height, width)
            data: torch.Tensor
                Contains key "target" with value a tensor of shape (batch, height, width)
            """
            resolution = get_resolution(**data)
            l2_loss = F.mse_loss(
                *self.cropper(T.modulus_if_complex(source), data["target"], resolution), reduction=reduction
            )

            return l2_loss

        def ssim_loss(source, reduction="mean", **data):
            """Calculate SSIM loss given source and target.

            Parameters
            ----------
            Source:  shape (batch, complex=2, height, width)
            Data: Contains key "target" with value a tensor of shape (batch, height, width)
            """
            resolution = get_resolution(**data)
            if reduction != "mean":
                raise AssertionError(
                    f"SSIM loss can only be computed with reduction == 'mean'." f" Got reduction == {reduction}."
                )

            source_abs, target_abs = self.cropper(T.modulus_if_complex(source), data["target"], resolution)
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

    def compute_sensitivity_map(self, sensitivity_map):
        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if "sensitivity_model" in self.models:
            # Move channels to first axis
            sensitivity_map = sensitivity_map.permute(
                (0, 1, 4, 2, 3)
            )  # shape (batch, coil, complex=2, height,  width)

            sensitivity_map = self.compute_model_per_coil(
                self.models, "sensitivity_model", sensitivity_map, self._coil_dim
            ).permute(
                (0, 1, 3, 4, 2)
            )  # has channel last: shape (batch, coil, height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map ** 2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch, [slice], height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

    @torch.no_grad()
    def reconstruct_volumes(
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]],
        regularizer_fns: Optional[Dict[str, Callable]] = None,
        add_target: bool = True,
    ):
        """Validation process. Assumes that each batch only contains slices of the same volume *AND* that these are
        sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        add_target: bool
            If true, will add the target to the output

        Returns
        -------
        torch.Tensor, Optional[torch.Tensor]

        # TODO(jt): visualization should be a namedtuple or a dict or so
        """
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Let us inspect this data
        all_filenames = list(data_loader.dataset.volume_indices.keys())
        num_for_this_process = len(list(data_loader.batch_sampler.sampler.volume_indices.keys()))
        self.logger.info(
            f"Reconstructing a total of {len(all_filenames)} volumes. "
            f"This process has {num_for_this_process} volumes (world size: {communication.get_world_size()})."
        )

        last_filename = None  # At the start of evaluation, there are no filenames.
        curr_volume = None
        curr_target = None
        volume_size = 0
        filenames_seen = 0

        # Loop over dataset. This requires the use of direct.data.sampler.DistributedSequentialSampler as this sampler
        # splits the data over the different processes, and outputs the slices linearly. The implicit assumption here is
        # that the slices are outputted from the Dataset *sequentially* for each volume one by one, and each batch only
        # contains data from one volume.
        time_start = time.time()
        loss_dict_list = []
        for iter_idx, data in enumerate(data_loader):
            filename = _get_filename_from_batch(data)
            if last_filename is None:
                last_filename = filename  # First iteration last_filename is not set.

            slice_nos = data["slice_no"]
            scaling_factors = data["scaling_factor"]
            resolution = _compute_resolution(
                key=self.cfg.validation.crop,  # type: ignore
                reconstruction_size=data.get("reconstruction_size", None),
            )
            # Compute output
            iteration_output = self._do_iteration(data, loss_fns=loss_fns, regularizer_fns=regularizer_fns)
            output = iteration_output.output_image
            loss_dict = iteration_output.data_dict

            # Output is complex-valued, and has to be cropped. This holds for both output and target.
            # Output has shape (batch, complex, [slice], height, width)
            output_abs = _process_output(
                output,
                scaling_factors,
                resolution=resolution,
            )

            if add_target:
                # Target has shape (batch, [slice], height, width)
                target_abs = _process_output(
                    data["target"].detach(),
                    scaling_factors,
                    resolution=resolution,
                )

            if not curr_volume:
                volume_size = data_loader.batch_sampler.sampler.volume_indices[filename]
                curr_volume = torch.zeros((volume_size, *(output_abs.size[1:])), dtype=output_abs.dtype)
                loss_dict_list.append(loss_dict)
                if add_target:
                    curr_target = curr_volume.copy()

            curr_volume[slice_nos[0] : slice_nos[-1], ...] = output_abs
            if add_target:
                curr_target[slice_nos[0]: slice_nos[-1], ...] = target_abs

            # Check if we had the last batch
            if slice_nos[-1] + 1 == volume_size:
                if all_filenames:
                    log_prefix = f"{filenames_seen} of {num_for_this_process} volumes reconstructed:"
                else:
                    log_prefix = f"{iter_idx + 1} of {len(data_loader)} slices reconstructed:"
                self.logger.info(
                    f"{log_prefix} {last_filename}"
                    f" (shape = {list(curr_volume.shape)}) in {time.time() - time_start:.3f}s."
                )
                yield curr_volume, curr_target, reduce_list_of_dicts(loss_dict_list), filename

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]],
        regularizer_fns: Optional[Dict[str, Callable]] = None,
        crop: Optional[str] = None,
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
        # TODO(jt): visualization should be a namedtuple or a dict or so
        """
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        volume_metrics = self.build_metrics(self.cfg.validation.metrics)  # type: ignore
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

        for volume_idx, output in enumerate(self.reconstruct_volumes(data_loader, add_target=True)):
            volume, target, volume_loss_dict, filename = output
            curr_metrics = {
                metric_name: metric_fn(target, volume).clone()
                for metric_name, metric_fn in volume_metrics.items()
            }
            val_volume_metrics[filename] = curr_metrics
            val_losses.append(volume_loss_dict)

            # Log the center slice of the volume
            if len(visualize_slices) < self.cfg.logging.tensorboard.num_images:  # type: ignore
                visualize_slices.append(volume[volume.shape[0] // 2])
                visualize_target.append(target[target.shape[0] // 2])

        # Average loss dict
        loss_dict = reduce_list_of_dicts(val_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        # TODO: Does not work yet with normal gather.
        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(val_volume_metrics))
        return loss_dict, all_gathered_metrics, visualize_slices, visualize_target

    @staticmethod
    def cropper(source, target, resolution):
        """2D source/target cropper.

        Parameters
        ----------
        source: torch.Tensor
            Has shape (batch, height, width)
        target: torch.Tensor
            Has shape (batch, height, width)
        resolution: tuple
            Target resolution.
        """

        if not resolution or all(_ == 0 for _ in resolution):
            return source.unsqueeze(1), target.unsqueeze(1)  # Added channel dimension.

        source_abs = T.center_crop(source, resolution).unsqueeze(1)  # Added channel dimension.
        target_abs = T.center_crop(target, resolution).unsqueeze(1)  # Added channel dimension.

        return source_abs, target_abs

    @staticmethod
    def compute_model_per_coil(models, model_name, data, coil_dim):
        """Computes model per coil."""
        # data is of shape (batch, coil, complex=2, height, width)
        output = []

        for idx in range(data.size(coil_dim)):
            subselected_data = data.select(coil_dim, idx)
            output.append(models[model_name](subselected_data))
        output = torch.stack(output, dim=coil_dim)

        # output is of shape (batch, coil, complex=2, height, width)
        return output


def _process_output(data, scaling_factors=None, resolution=None):
    # data is of shape (batch, complex=2, height, width)
    if scaling_factors is not None:
        data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(data.device)

    data = T.modulus_if_complex(data)

    if len(data.shape) == 3:  # (batch, height, width)
        data = data.unsqueeze(1)  # Added channel dimension.

    if resolution is not None:
        data = T.center_crop(data, resolution).contiguous()

    return data


def _compute_resolution(key, reconstruction_size):
    """Computes resolution.

    Parameters
    ----------
    key: str
        Can be 'header', 'training' or None.
    reconstruction_size: tuple
        Reconstruction size.

    Returns
    -------
    resolution: tuple
        Resolution of reconstruction.
    """
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


def _get_filename_from_batch(data):
    filenames = data.pop("filename")
    if len(set(filenames)) != 1:
        raise ValueError(
            f"Expected a batch during validation to only contain filenames of one case. " f"Got {set(filenames)}."
        )
    return filenames[0]
