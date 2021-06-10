# coding=utf-8
# Copyright (c) DIRECT Contributors
import time
from collections import defaultdict
from os import PathLike
from typing import Callable, DefaultDict, Dict, List, Optional, Union

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


class RIMEngine(Engine):
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

        # The first input_image in the iteration is the input_image with the mask applied and no first hidden state.
        input_image = None
        hidden_state = None
        output_image = None
        loss_dicts = []
        regularizer_dicts = []

        data = dict_to_device(data, self.device)
        # TODO(jt): keys=['sampling_mask', 'sensitivity_map', 'target', 'masked_kspace', 'scaling_factor']

        # sensitivity_map of shape (batch, coil, [slice], height,  width, complex=2)
        sensitivity_map = data["sensitivity_map"]

        if "noise_model" in self.models:
            raise NotImplementedError()

        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if "sensitivity_model" in self.models:

            # Move channels to first axis
            sensitivity_map = data["sensitivity_map"].permute(
                (0, 1, 4, 2, 3) if self.ndim == 2 else (0, 1, 5, 2, 3, 4)
            ) # shape (batch, coil, complex=2, [slice], height,  width)

            sensitivity_map = (
                self.compute_model_per_coil("sensitivity_model", sensitivity_map)
                .permute((0, 1, 3, 4, 2) if self.ndim == 2 else (0, 1, 3, 4, 5, 2))
            ) # has channel last: shape (batch, coil, [slice], height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        complex_dim, coil_dim = -1, 1
        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map ** 2).sum(complex_dim)).sum(coil_dim)
        ) # shape (batch, [slice], height, width)
        # sensitivity_map_norm = sensitivity_map_norm.unsqueeze(1).unsqueeze(-1)

        data["sensitivity_map"] = T.safe_divide(sensitivity_map, sensitivity_map_norm)

        if self.cfg.model.scale_loglikelihood:  # type: ignore
            scaling_factor = 1.0 * self.cfg.model.scale_loglikelihood / (data["scaling_factor"] ** 2)  # type: ignore
            scaling_factor = scaling_factor.reshape(-1, 1) # shape (batch, complex=1)
            self.logger.debug(f"Scaling factor is: {scaling_factor}")
        else:
            # Needs fixing.
            scaling_factor = torch.tensor([1.0]).to(sensitivity_map.device) # shape (complex=1, )

        for _ in range(self.cfg.model.steps):  # type: ignore
            with autocast(enabled=self.mixed_precision):
                if input_image is not None:
                    # TODO(gy): is this print here needed?
                    print(input_image.shape, input_image.names, "input_image")
                    input_image = input_image.permute(
                        (0, 2, 3, 4, 1) if input_image.ndim == 5 else (0, 2, 3, 1)
                    )
                reconstruction_iter, hidden_state = self.model(
                    **data,
                    input_image=input_image,
                    hidden_state=hidden_state,
                    loglikelihood_scaling=scaling_factor,
                )
                # reconstruction_iter: list with tensors of shape (batch, complex=2, [slice,] height, width)
                # hidden_state has shape: (batch, num_hidden_channels, [slice,] height, width, depth)

                output_image = reconstruction_iter[-1] # shape (batch, complex=2, [slice,] height,  width)

                loss_dict = {
                    k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()
                }
                regularizer_dict = {
                    k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
                }

                # TODO: This seems too similar not to be able to do this, perhaps a partial can help here
                for output_image_iter in reconstruction_iter:
                    for k, v in loss_dict.items():
                        loss_dict[k] = v + loss_fns[k](
                            output_image_iter,
                            **data,
                            reduction="mean",
                        )

                    for k, v in regularizer_dict.items():
                        regularizer_dict[k] = (
                            v
                            + regularizer_fns[k](
                                output_image_iter,
                                **data,
                            )
                        )

                loss_dict = {k: v / len(reconstruction_iter) for k, v in loss_dict.items()}
                regularizer_dict = {k: v / len(reconstruction_iter) for k, v in regularizer_dict.items()}

                loss = sum(loss_dict.values()) + sum(regularizer_dict.values())

            if self.model.training:
                # TODO(gy): With steps >= 1, calling .backward(retain_grad=False) caused problems.
                #  Check with Jonas if it's ok.
                if (self.cfg.model.steps > 1) and (_ < self.cfg.model.steps - 1):
                    self._scaler.scale(loss).backward(retain_graph=True)
                else:
                    self._scaler.scale(loss).backward()

            # Detach hidden state from computation graph, to ensure loss is only computed per RIM block.
            hidden_state = hidden_state.detach() # shape: (batch, num_hidden_channels, [slice,] height, width, depth)
            input_image = output_image.detach() # shape (batch, complex=2, [slice,] height,  width)

            loss_dicts.append(detach_dict(loss_dict))
            regularizer_dicts.append(
                detach_dict(regularizer_dict)
            )  # Need to detach dict as this is only used for logging.

        # Add the loss dicts together over RIM steps, divide by the number of steps.
        loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum", divisor=self.cfg.model.steps)  # type: ignore
        regularizer_dict = reduce_list_of_dicts(regularizer_dicts, mode="sum", divisor=self.cfg.model.steps)  # type: ignore

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict},
        )

    def build_loss(self, **kwargs) -> Dict:
        # TODO: Cropper is a processing output tool.
        def get_resolution(**data):
            """Be careful that this will use the cropping size of the FIRST sample in the batch."""
            return self.compute_resolution(self.cfg.training.loss.crop, data.get("reconstruction_size", None))

        # TODO(jt) Ideally this is also configurable:
        # - Do in steps (use insertation order)
        # Crop -> then loss.

        def l1_loss(source, reduction="mean", **data):
            """
            Calculate L1 loss given source and target.

            Parameters:
            -----------
                Source:  shape (batch, complex=2, height, width)
                Data: Contains key "target" with value a tensor of shape (batch, height, width)

            """
            resolution = get_resolution(**data)
            l1_loss = F.l1_loss(*self.cropper(source, data["target"], resolution), reduction=reduction)

            return l1_loss

        def ssim_loss(source, reduction="mean", **data):
            """
            Calculate SSIM loss given source and target.

            Parameters:
            -----------
                Source:  shape (batch, complex=2, height, width)
                Data: Contains key "target" with value a tensor of shape (batch, height, width)

            """
            resolution = get_resolution(**data)
            if reduction != "mean":
                raise AssertionError(f"SSIM loss can only be computed with reduction == 'mean'."
                                     f" Got reduction == {reduction}.")

            source_abs, target_abs = self.cropper(source, data["target"], resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = SSIMLoss().to(source_abs.device)(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:  # type: ignore
            loss_fn = curr_loss.function
            if loss_fn == "l1_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l1_loss)
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
        is_validation_process=True,
    ):

        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Variables required for evaluation.
        # TODO(jt): Consider if this needs to be in the main engine.py or here. Might be possible we have different
        # types needed, perhaps even a FastMRI engine or something similar depending on the metrics.
        volume_metrics = self.build_metrics(self.cfg.validation.metrics)  # type: ignore

        # filenames can be in the volume_indices attribute of the dataset
        num_for_this_process = None
        if hasattr(data_loader.dataset, "volume_indices"):
            all_filenames = list(data_loader.dataset.volume_indices.keys())
            num_for_this_process = len(list(data_loader.batch_sampler.sampler.volume_indices.keys()))
            self.logger.info(
                f"Reconstructing a total of {len(all_filenames)} volumes. "
                f"This process has {num_for_this_process} volumes (world size: {communication.get_world_size()})."
            )

        filenames_seen = 0

        reconstruction_output: DefaultDict = defaultdict(list)
        targets_output: DefaultDict = defaultdict(list)
        val_losses = []
        val_volume_metrics: Dict[PathLike, Dict] = defaultdict(dict)
        last_filename = None

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices: List[np.ndarray] = []
        visualize_target: List[np.ndarray] = []
        # visualizations = {}

        extra_visualization_keys = self.cfg.logging.log_as_image if self.cfg.logging.log_as_image else []  # type: ignore

        # Loop over dataset. This requires the use of direct.data.sampler.DistributedSequentialSampler as this sampler
        # splits the data over the different processes, and outputs the slices linearly. The implicit assumption here is
        # that the slices are outputted from the Dataset *sequentially* for each volume one by one.
        time_start = time.time()

        for iter_idx, data in enumerate(data_loader):
            # data = AddNames()(data)
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
            # sensitivity_map = iteration_output.sensitivity_map

            loss_dict = detach_dict(loss_dict)
            output = output.detach()
            val_losses.append(loss_dict)

            # Output is complex-valued, and has to be cropped. This holds for both output and target.
            # Output has shape (batch, complex, [slice], height, width)
            output_abs = self.process_output(
                output,
                scaling_factors,
                resolution=resolution,
            )

            if is_validation_process:
                # Target has shape (batch, [slice], height, width)
                target_abs = self.process_output(
                    data["target"].detach(),
                    scaling_factors,
                    resolution=resolution,
                )
                for key in extra_visualization_keys:
                    curr_data = data[key].detach()
                    # Here we need to discover which keys are actually normalized or not
                    # this requires a solution to issue #23: https://github.com/directgroup/direct/issues/23

            del output  # Explicitly call delete to clear memory.
            # TODO: Is a hack.

            # Aggregate volumes to be able to compute the metrics on complete volumes.
            for idx, filename in enumerate(filenames):
                if last_filename is None:
                    last_filename = filename  # First iteration last_filename is not set.

                # If the new filename is not the previous one, then we can reconstruct the volume as the sampling
                # is linear.
                # For the last case we need to check if we are at the last batch *and* at the last element in the batch.
                is_last_element_of_last_batch = iter_idx + 1 == len(data_loader) and idx + 1 == len(data["target"])
                if filename != last_filename or is_last_element_of_last_batch:
                    filenames_seen += 1
                    # Now we can ditch the reconstruction dict by reconstructing the volume,
                    # will take too much memory otherwise.
                    # TODO: Stack does not support named tensors.
                    volume = torch.stack([_[1] for _ in reconstruction_output[last_filename]])
                    if is_validation_process:
                        target = torch.stack([_[1] for _ in targets_output[last_filename]])
                        curr_metrics = {
                            metric_name: metric_fn(target, volume) for metric_name, metric_fn in volume_metrics.items()
                        }
                        val_volume_metrics[last_filename] = curr_metrics
                        # Log the center slice of the volume
                        if len(visualize_slices) < self.cfg.logging.tensorboard.num_images:  # type: ignore
                            visualize_slices.append(volume[volume.shape[0] // 2])
                            visualize_target.append(target[target.shape[0] // 2])

                        # Delete outputs from memory, and recreate dictionary.
                        # This is not needed when not in validation as we are actually interested
                        # in the iteration output.
                        del targets_output
                        targets_output = defaultdict(list)
                        del reconstruction_output
                        reconstruction_output = defaultdict(list)

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

                curr_slice = output_abs[idx].detach()
                slice_no = int(slice_nos[idx].numpy())

                # TODO: CPU?
                reconstruction_output[filename].append((slice_no, curr_slice.cpu()))

                if is_validation_process:
                    targets_output[filename].append((slice_no, target_abs[idx].cpu()))

        # Average loss dict
        loss_dict = reduce_list_of_dicts(val_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        # TODO: Does not work yet with normal gather.
        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(val_volume_metrics))
        if not is_validation_process:
            return loss_dict, reconstruction_output

        # TODO: Apply named tuples where applicable
        # TODO: Several functions have multiple DoIterationOutput values, in many cases
        # TODO: it would be more convenient to convert this to namedtuples.
        return loss_dict, all_gathered_metrics, visualize_slices, visualize_target

    # TODO: WORK ON THIS.
    # def do_something_with_the_noise(self, data):
    #     # Seems like a better idea to compute noise in image space
    #     masked_kspace = data["masked_kspace"]
    #     sensitivity_map = data["sensitivity_map"]
    #     masked_image_forward = self.backward_operator(masked_kspace)
    #     masked_image_forward = masked_image_forward.align_to(
    #         *self.complex_names(add_coil=True)
    #     )
    #     noise_vector = self.compute_model_per_coil("noise_model", masked_image_forward)
    #
    #     # Create a complex noise vector
    #     noise_vector = torch.view_as_complex(
    #         noise_vector.reshape(
    #             noise_vector.shape[0],
    #             noise_vector.shape[1],
    #             noise_vector.shape[-1] // 2,
    #             2,
    #         )
    #     )
    #
    #     # Apply prewhitening
    #     # https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.1241
    #     noise_int = noise_vector.reshape(
    #         noise_vector.shape[0], noise_vector.shape[1], -1
    #     )
    #     noise_int *= 1 / (noise_int.shape[-1] - 1)
    #
    #     phi = T.complex_bmm(noise_int, noise_int.conj().transpose(1, 2))
    #     # TODO(jt): No cholesky nor inverse on GPU yet...
    #     new_basis = torch.inverse(torch.cholesky(phi.cpu())).to(phi.device) / np.sqrt(
    #         2.0
    #     )
    #
    #     # TODO(jt): Likely we need something a bit more elaborate e.g. percentile
    #     masked_kspace_max = masked_kspace.max()
    #     masked_kspace = self.view_as_complex(masked_kspace)
    #     prewhitened_kspace = (
    #         T.complex_bmm(
    #             new_basis,
    #             masked_kspace.rename(None).reshape(
    #                 masked_kspace.shape[0], masked_kspace.shape[1], -1
    #             ),
    #         )
    #         .reshape(masked_kspace.shape)
    #         .refine_names(*masked_kspace.names)
    #     )
    #     prewhitened_kspace = self.view_as_real(prewhitened_kspace)
    #
    #     # kspace has different values after whitening, lets map back
    #     prewhitened_kspace = (
    #         prewhitened_kspace / prewhitened_kspace.max() * masked_kspace_max
    #     )
    #     data["masked_kspace"] = prewhitened_kspace
    #
    #     sensitivity_map = self.view_as_complex(sensitivity_map)
    #     prewhitened_sensitivity_map = (
    #         T.complex_bmm(
    #             new_basis,
    #             sensitivity_map.rename(None).reshape(
    #                 masked_kspace.shape[0], masked_kspace.shape[1], -1
    #             ),
    #         )
    #         .reshape(masked_kspace.shape)
    #         .refine_names(*sensitivity_map.names)
    #     )
    #     sensitivity_map = self.view_as_real(prewhitened_sensitivity_map)

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
                f"`training` to take the same value as training."
            )
        return resolution

    def cropper(self, source, target, resolution):
        """
        2D source/target cropper

        Parameters:
        -----------
            Source has shape (batch, complex=2, height, width)
            Target has shape (batch, height, width)

        """
        source_abs = T.modulus(source) # shape (batch, height, width)

        if not resolution or all(_ == 0 for _ in resolution):
            return source_abs.unsqueeze(1), target.unsqueeze(1)

        source_abs = T.center_crop(source_abs, resolution).unsqueeze(1)
        target_abs = T.center_crop(target, resolution).unsqueeze(1)

        return source_abs, target_abs

    def complex_to_channel(self, names):
        # TODO(jt): This only works in the CHW order
        if self.ndim == 2:
            channel_axis = names.index("height")
        elif self.ndim == 3:
            channel_axis = names.index("slice")
        else:
            raise NotImplementedError

        complex_index = names.index("complex")
        names = list(names)
        names.pop(complex_index)
        names.insert(channel_axis, "complex")

        return names

    @staticmethod
    def complex_to_last(names):
        if names[-1] == "complex":
            return names
        index = names.index("complex")
        names = list(names)

        names.pop(index)
        names.insert(-1, "complex")
        return names

    def compute_model_per_coil(self, model_name, data):
        # data is of shape (batch, coil, complex=2, [slice], height, width)
        output = []

        coil_index = 1
        for idx in range(data.size(coil_index)):
            subselected_data = data.select(coil_index, idx)
            output.append(self.models[model_name](subselected_data))
        output = torch.stack(output, dim=coil_index)

        # output is of shape (batch, coil, complex=2, [slice], height, width)
        return output


class RIM3dEngine(RIMEngine):
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

    def process_output(self, data, scaling_factors=None, resolution=None):
        # Data has shape (batch, complex, slice, height, width)
        # TODO(gy): verify shape

        slice_dim = -3
        center_slice = data.size(slice_dim) // 2

        if scaling_factors is not None:
            data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(data.device)

        data = T.modulus_if_complex(data).select(slice_dim, center_slice)

        if len(data.shape) == 3:  # (batch, height, width)
            data = data.unsqueeze(1)  # Added channel dimension.

        if resolution is not None:
            data = T.center_crop(data, resolution).contiguous()

        return data

    def cropper(self, source, target, resolution=(320, 320)):
        """
        2D source/target cropper

        Parameters:
        -----------
            Source has shape (batch, complex=2, slice, height, width)
            Target has shape (batch, slice, height, width)

        """
        # TODO(gy): Verify target shape
        slice_index = -3

        # TODO(gy): Why is this set to True and then have an if statement?
        use_center_slice = True
        if use_center_slice:
            # Source and target have a different number of slices when trimming in depth
            source = source.select(
                slice_index,
                source.size(slice_index) // 2
            ) # shape (batch, complex=2, height, width)
            target = target.select(
                slice_index,
                target.size(slice_index) // 2
            ).unsqueeze(1) # shape (batch, complex=1, height, width)
        # else:
        #     source = source.permute(0, 2, 3, 4, 1) # shape (batch *slice, height, width, complex=2)
        #     source = source.flatten(0, 1).permute(0, 3, 1, 2) # shape (batch *slice, complex=2, height, width)
        #     target = target.flatten(0, 1).unsqueeze(1) # shape (batch*slice, 1, height, width)

        source_abs = T.modulus(source) # shape (batch, height, width)

        if not resolution or all(_ == 0 for _ in resolution):
            return source_abs.unsqueeze(1), target

        source_abs = T.center_crop(source_abs, resolution).unsqueeze(1)
        target_abs = T.center_crop(target, resolution)
        return source_abs, target_abs
