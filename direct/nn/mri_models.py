# coding=utf-8
# Copyright (c) DIRECT Contributors

"""MRI model engine of DIRECT."""

import gc
import pathlib
import time
from abc import abstractmethod
from collections import defaultdict
from os import PathLike
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput, Engine
from direct.functionals import SSIMLoss
from direct.utils import communication, merge_list_of_dicts, multiply_function, reduce_list_of_dicts
from direct.utils.communication import reduce_tensor_dict


def _crop_volume(source, target, resolution):
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


class MRIModelEngine(Engine):
    """Engine for MRI models.

    Each child class should implement their own :meth:`_do_iteration` method.
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
            Device. Can be "cuda" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        models: nn.Module
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
        self._complex_dim = -1
        self._coil_dim = 1

    @abstractmethod
    def _do_iteration(
        self,
        data: Dict[str, Union[List, torch.Tensor]],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """To be implemented by child class.

        Should output a :meth:`DoIterationOutput` object with `output_image`, `sensitivity_map` and
        `data_dict` attributes.
        """

    def build_loss(self) -> Dict:
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
            source: torch.Tensor
                Has shape (batch, [complex=2,] height, width)
            data: Dict[str, torch.Tensor]
                Contains key "target" with value a tensor of shape (batch, height, width)

            Returns
            -------
            l1_loss: torch.Tensor
                L1 loss.
            """
            resolution = get_resolution(**data)
            l1_loss = F.l1_loss(
                *_crop_volume(T.modulus_if_complex(source), data["target"], resolution), reduction=reduction
            )

            return l1_loss

        def l2_loss(source, reduction="mean", **data):
            """Calculate L2 loss (MSE) given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, [complex=2,] height, width)
            data: Dict[str, torch.Tensor]
                Contains key "target" with value a tensor of shape (batch, height, width)

            Returns
            -------
            l2_loss: torch.Tensor
                L2 loss.
            """
            resolution = get_resolution(**data)
            l2_loss = F.mse_loss(
                *_crop_volume(T.modulus_if_complex(source), data["target"], resolution), reduction=reduction
            )

            return l2_loss

        def ssim_loss(source, reduction="mean", **data):
            """Calculate SSIM loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, [complex=2,] height, width)
            data: Dict[str, torch.Tensor]
                Contains key "target" with value a tensor of shape (batch, height, width)

            Returns
            -------
            ssim_loss: torch.Tensor
                SSIM loss.
            """
            resolution = get_resolution(**data)
            if reduction != "mean":
                raise AssertionError(
                    f"SSIM loss can only be computed with reduction == 'mean'." f" Got reduction == {reduction}."
                )

            source_abs, target_abs = _crop_volume(T.modulus_if_complex(source), data["target"], resolution)
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

    def compute_sensitivity_map(self, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Computes sensitivity maps :math:`\{S^k\}_{k=1}^{n_c}` if `sensitivity_model` is available.

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
        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if "sensitivity_model" in self.models:
            # Move channels to first axis
            sensitivity_map = sensitivity_map.permute(
                (0, 1, 4, 2, 3)
            )  # shape (batch, coil, complex=2, height,  width)

            sensitivity_map = self.compute_model_per_coil("sensitivity_model", sensitivity_map).permute(
                (0, 1, 3, 4, 2)
            )  # has channel last: shape (batch, coil, height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch, height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

    @torch.no_grad()
    def reconstruct_volumes(
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
        # TODO(jt): visualization should be a namedtuple or a dict or so
        """
        # pylint: disable=too-many-locals, arguments-differ
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Let us inspect this data
        all_filenames = list(data_loader.dataset.volume_indices.keys())
        num_for_this_process = len(list(data_loader.batch_sampler.sampler.volume_indices.keys()))
        self.logger.info(
            "Reconstructing a total of %s volumes. This process has %s volumes (world size: %s).",
            len(all_filenames),
            num_for_this_process,
            communication.get_world_size(),
        )

        last_filename = None  # At the start of evaluation, there are no filenames.
        curr_volume = None
        curr_target = None
        slice_counter = 0
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
                slice_counter = 0
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
            )

            if add_target:
                target_abs = _process_output(
                    data["target"].detach().clone(),
                    scaling_factors,
                    resolution=resolution,
                )

            if curr_volume is None:
                volume_size = len(data_loader.batch_sampler.sampler.volume_indices[filename])
                curr_volume = torch.zeros(*(volume_size, *output_abs.shape[1:]), dtype=output_abs.dtype)
                loss_dict_list.append(loss_dict)
                if add_target:
                    curr_target = curr_volume.clone()

            curr_volume[slice_counter : slice_counter + output_abs.shape[0], ...] = output_abs.cpu()
            if add_target:
                curr_target[slice_counter : slice_counter + output_abs.shape[0], ...] = target_abs.cpu()

            slice_counter += output_abs.shape[0]

            # Check if we had the last batch
            if slice_counter == volume_size:
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
                yield (curr_volume, curr_target, reduce_list_of_dicts(loss_dict_list), filename) if add_target else (
                    curr_volume,
                    reduce_list_of_dicts(loss_dict_list),
                    filename,
                )

    @torch.no_grad()
    def evaluate(
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
        # pylint: disable=arguments-differ

        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        volume_metrics = self.build_metrics(self.cfg.validation.metrics)  # type: ignore
        val_losses = []
        val_volume_metrics: Dict[PathLike, Dict] = defaultdict(dict)

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices: List[np.ndarray] = []
        visualize_target: List[np.ndarray] = []

        for _, output in enumerate(
            self.reconstruct_volumes(
                data_loader, loss_fns=loss_fns, add_target=True, crop=self.cfg.validation.crop  # type: ignore
            )
        ):
            volume, target, volume_loss_dict, filename = output
            curr_metrics = {
                metric_name: metric_fn(target, volume).clone() for metric_name, metric_fn in volume_metrics.items()
            }
            curr_metrics_string = ", ".join([f"{x}: {float(y)}" for x, y in curr_metrics.items()])
            self.logger.info("Metrics for %s: %s", filename, curr_metrics_string)
            # TODO: Path can be tricky if it is not unique (e.g. image.h5)
            val_volume_metrics[filename.name] = curr_metrics
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
        output = torch.stack(output, dim=self._coil_dim)
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
    # This can be fixed when there is a custom collate_fn
    return pathlib.Path(filenames[0])
