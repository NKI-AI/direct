# coding=utf-8
# Copyright (c) DIRECT Contributors
from collections import defaultdict
from typing import Dict, Callable, Tuple, Optional

import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from direct.config import BaseConfig
from direct.data.mri_transforms import AddNames
from direct.data.transforms import modulus_if_complex, center_crop, modulus, safe_divide
from direct.engine import Engine
from direct.utils import (
    dict_to_device,
    reduce_list_of_dicts,
    detach_dict,
    merge_list_of_dicts,
)
from direct.utils import (
    normalize_image,
    multiply_function,
    communication,
)
from direct.utils.communication import reduce_tensor_dict
from direct.functionals import SSIMLoss


class RIMEngine(Engine):
    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: int,
        mixed_precision: bool = False,
        **models: Dict[str, nn.Module],
    ):
        super().__init__(cfg, model, device, mixed_precision, **models)

    def _do_iteration(
        self, data: Dict[str, torch.Tensor], loss_fns: Optional[Dict[str, Callable]]
    ) -> Tuple[torch.Tensor, Dict]:

        # loss_fns can be done, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        # TODO(jt): Target is not needed in the model input, but in the loss computation. Keep it here for now.
        target = data["target"].align_to(*self.complex_names).to(self.device)  # type: ignore
        # The first input_image in the iteration is the input_image with the mask applied and no first hidden state.
        input_image = data.pop("masked_image").to(self.device)  # type: ignore
        hidden_state = None
        output_image = None
        loss_dicts = []

        # TODO: Target might not need to be copied.
        data = dict_to_device(data, self.device)
        # TODO(jt): keys=['sampling_mask', 'sensitivity_map', 'target', 'masked_kspace', 'scaling_factor']

        sensitivity_map = data["sensitivity_map"]
        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if "sensitivity_model" in self.models:
            sensitivity_map = self.compute_model_per_coil(
                self.models["sensitivity_model"], sensitivity_map
            )

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1
        sensitivity_map_norm = modulus(sensitivity_map).sum("coil")
        data["sensitivity_map"] = safe_divide(sensitivity_map, sensitivity_map_norm)

        for rim_step in range(self.cfg.model.steps):
            with autocast(enabled=self.mixed_precision):
                reconstruction_iter, hidden_state = self.model(
                    **data,
                    input_image=input_image,
                    hidden_state=hidden_state,
                )
                # TODO: Unclear why this refining is needed.

                output_image = reconstruction_iter[-1].refine_names(*self.complex_names)

                loss_dict = {
                    k: torch.tensor([0.0], dtype=target.dtype).to(self.device)
                    for k in loss_fns.keys()
                }
                for output_image_iter in reconstruction_iter:
                    for k, v in loss_dict.items():
                        loss_dict[k] = v + loss_fns[k](
                            output_image_iter,
                            target,
                            reduction="mean",
                        )

                loss_dict = {
                    k: v / len(reconstruction_iter) for k, v in loss_dict.items()
                }
                loss = sum(loss_dict.values())

            if self.model.training:
                self._scaler.scale(loss).backward()

            # Detach hidden state from computation graph, to ensure loss is only computed per RIM block.
            hidden_state = hidden_state.detach()
            input_image = output_image.detach()

            loss_dicts.append(
                detach_dict(loss_dict)
            )  # Need to detach dict as this is only used for logging.

        # Add the loss dicts together over RIM steps, divide by the number of steps.
        loss_dict = reduce_list_of_dicts(
            loss_dicts, mode="sum", divisor=self.cfg.model.steps
        )
        return output_image, loss_dict

    def build_loss(self, **kwargs) -> Dict:
        # TODO: Cropper is a processing output tool.
        resolution = self.cfg.training.loss.crop

        # TODO(jt) Ideally this is also configurable:
        # - Do in steps (use insertation order)
        # Crop -> then loss.

        def l1_loss(source, target, reduction="mean"):
            return F.l1_loss(
                *self.cropper(source, target, resolution), reduction=reduction
            )

        def ssim_loss(source, target, reduction="mean"):
            assert reduction == "mean"
            source_abs, target_abs = self.cropper(source, target, resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)
            return SSIMLoss().to(source_abs.device)(
                source_abs, target_abs, data_range=data_range
            )

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:
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
        crop: Optional[str] = None,
        is_validation_process=True,
    ):

        # TODO(jt): Also log other models output (e.g. sensitivity map).
        # TODO(jt): This can be simplified as the sampler now only outputs batches belonging to the same volume.
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Variables required for evaluation.
        # TODO(jt): Consider if this needs to be in the main engine.py or here. Might be possible we have different
        # types needed, perhaps even a FastMRI engine or something similar depending on the metrics.
        volume_metrics = self.build_metrics(self.cfg.validation.metrics)

        reconstruction_output = defaultdict(list)
        targets_output = defaultdict(list)
        val_losses = []
        val_volume_metrics = defaultdict(dict)
        last_filename = None

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices = []
        visualize_target = []

        # Loop over dataset. This requires the use of direct.data.sampler.DistributedSequentialSampler as this sampler
        # splits the data over the different processes, and outputs the slices linearly. The implicit assumption here is
        # that the slices are outputted from the Dataset *sequentially* for each volume one by one.
        for iter_idx, data in enumerate(data_loader):
            self.log_process(iter_idx, len(data_loader))
            data = AddNames()(data)
            filenames = data.pop("filename")
            if len(set(filenames)) != 1:
                raise ValueError(
                    f"Expected a batch during validation to only contain filenames of one case. "
                    f"Got {set(filenames)}."
                )

            slice_nos = data.pop("slice_no")
            scaling_factors = data.pop("scaling_factor")

            # Check if reconstruction size is the data
            if self.cfg.validation.crop == "header":
                # This will be of the form [tensor(x_0, x_1, ...), tensor(y_0, y_1,...), tensor(z_0, z_1, ...)] over
                # batches.
                resolution = [
                    _.cpu().numpy().tolist() for _ in data["reconstruction_size"]
                ]
                # The volume sampler should give validation indices belonging to the *same* volume, so it should be
                # safe taking the first element, the matrix size are in x,y,z (we work in z,x,y).
                resolution = [_[0] for _ in resolution][:-1]
            elif self.cfg.validation.crop == "training":
                resolution = self.cfg.training.loss.crop
            elif not self.cfg.validation.loss.crop:
                resolution = None
            else:
                raise ValueError(
                    f"Cropping should be either set to `header` to get the values from the header or "
                    f"`training` to take the same value as training."
                )

            # Compute output and loss.
            output, loss_dict = self._do_iteration(data, loss_fns)
            val_losses.append(loss_dict)

            # Output is complex-valued, and has to be cropped. This holds for both output and target.
            output_abs = self.process_output(
                output.refine_names(*self.complex_names).detach(),
                scaling_factors,
                resolution=resolution,
            )

            if is_validation_process:
                target_abs = self.process_output(
                    data["target"].refine_names(*self.real_names).detach(),
                    scaling_factors,
                    resolution=resolution,
                )
            del output  # Explicitly call delete to clear memory.
            # TODO: Is a hack.

            # Aggregate volumes to be able to compute the metrics on complete volumes.
            for idx, filename in enumerate(filenames):
                if last_filename is None:
                    last_filename = (
                        filename  # First iteration last_filename is not set.
                    )
                # If the new filename is not the previous one, then we can reconstruct the volume as the sampling
                # is linear.
                # For the last case we need to check if we are at the last batch *and* at the last element in the batch.
                if filename != last_filename or (
                    iter_idx + 1 == len(data_loader) and idx + 1 == len(data["target"])
                ):
                    # Now we can ditch the reconstruction dict by reconstructing the volume,
                    # will take too much memory otherwise.
                    # TODO: Stack does not support named tensors.
                    volume = torch.stack(
                        [
                            _[1].rename(None)
                            for _ in reconstruction_output[last_filename]
                        ]
                    )
                    self.logger.info(
                        f"Reconstructed {last_filename} (shape = {list(volume.shape)})."
                    )
                    if is_validation_process:
                        target = torch.stack(
                            [_[1].rename(None) for _ in targets_output[last_filename]]
                        )
                        curr_metrics = {
                            metric_name: metric_fn(volume, target)
                            for metric_name, metric_fn in volume_metrics.items()
                        }
                        val_volume_metrics[last_filename] = curr_metrics
                        # Log the center slice of the volume
                        if len(visualize_slices) < self.cfg.tensorboard.num_images:
                            visualize_slices.append(
                                normalize_image(volume[volume.shape[0] // 2])
                            )
                            visualize_target.append(
                                normalize_image(target[target.shape[0] // 2])
                            )

                        # Delete outputs from memory, and recreate dictionary. This is not needed when not in validation
                        # as we are actually interested in the output
                        del targets_output
                        targets_output = defaultdict(list)
                        del reconstruction_output
                        reconstruction_output = defaultdict(list)

                    last_filename = filename

                curr_slice = output_abs[idx]
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

        # TODO(jt): Does not work yet with normal gather.
        all_gathered_metrics = merge_list_of_dicts(
            communication.all_gather(val_volume_metrics)
        )

        if not is_validation_process:
            return loss_dict, reconstruction_output

        # TODO(jt): Make named tuple
        return loss_dict, all_gathered_metrics, visualize_slices, visualize_target

    def process_output(self, data, scaling_factors=None, resolution=None):
        if scaling_factors is not None:
            data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(
                data.device
            )
        data = modulus_if_complex(data).rename(None)
        if len(data.shape) == 3:  # (batch, height, width)
            data = data.unsqueeze(1)  # Added channel dimension.

        if resolution is not None:
            data = center_crop(data, resolution).contiguous()

        return data

    def cropper(self, source, target, resolution=(320, 320)):
        source = source.rename(None)
        target = target.rename(None)
        source_abs = modulus(source.refine_names(*self.complex_names))
        if not resolution or all([_ == 0 for _ in resolution]):
            return source_abs.rename(None).unsqueeze(1), target

        source_abs = center_crop(source_abs, resolution).rename(None).unsqueeze(1)
        target_abs = center_crop(target, resolution)
        return source_abs, target_abs

    def compute_model_per_coil(self, model, data):
        output = []
        coil_index = data.names.index("coil")
        for idx in range(data.size("coil")):
            subselected_data = data.select("coil", idx)
            output.append(
                model(subselected_data.align_to(*self.complex_names).rename(None))
                .refine_names(*self.complex_names)  # noqa
                .align_to(*self.complex_names_complex_last)
                .rename(None)
            )

        output = torch.stack(output, dim=coil_index).refine_names(*data.names)
        return output


class RIM3dEngine(RIMEngine):
    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: int,
        mixed_precision: bool = False,
        **models: Dict[str, nn.Module],
    ):
        super().__init__(cfg, model, device, mixed_precision, **models)

    def process_output(self, data, scaling_factors=None, resolution=None):
        center_slice = data.size("slice") // 2

        if scaling_factors is not None:
            data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(
                data.device
            )
        data = modulus_if_complex(data).select("slice", center_slice).rename(None)
        if len(data.shape) == 3:  # (batch, height, width)
            data = data.unsqueeze(1)  # Added channel dimension.

        if resolution is not None:
            data = center_crop(data, resolution).contiguous()

        return data

    def cropper(self, source, target, resolution=(320, 320)):
        # Can also do reshaping and compute over the full volume
        slice_index = target.names.index("slice")

        use_center_slice = True
        if use_center_slice:
            center_slice = target.size("slice") // 2
            source = source.select(slice_index, center_slice)
            target = target.select("slice", center_slice).rename(None)
        else:
            source = source.refine_names(*target.names)
            source = source.flatten(["batch", "slice"], "batch").rename(None)
            target = target.flatten(["batch", "slice"], "batch").rename(None)

        complex_names = self.complex_names.copy()
        complex_names.pop(slice_index)

        source_abs = modulus(source.refine_names(*complex_names))
        if not resolution or all([_ == 0 for _ in resolution]):
            return source_abs.rename(None).unsqueeze(1), target

        source_abs = center_crop(source_abs, resolution).rename(None).unsqueeze(1)
        target_abs = center_crop(target, resolution)
        return source_abs, target_abs
