# coding=utf-8
# Copyright (c) DIRECT Contributors
from collections import defaultdict
from typing import Dict, Callable, Tuple, Optional

import torch
from apex import amp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from direct.config import BaseConfig
from direct.data.mri_transforms import AddNames
from direct.data.transforms import modulus_if_complex, center_crop, modulus, safe_divide
from direct.engine import Engine
from direct.utils import dict_to_device, reduce_list_of_dicts, detach_dict, str_to_class
from direct.utils import normalize_image, multiply_function, communication
from direct.utils.communication import reduce_tensor_dict
from direct.functionals import batch_psnr, SSIM


class RIMEngine(Engine):
    def __init__(self, cfg: BaseConfig,
                 model: nn.Module,
                 device: int, mixed_precision: bool = False, **models: Dict[str, nn.Module]):
        super().__init__(cfg, model, device, mixed_precision, **models)

    def _do_iteration(self,
                      data: Dict[str, torch.Tensor],
                      loss_fns: Optional[Dict[str, Callable]]) -> Tuple[torch.Tensor, Dict]:

        # loss_fns can be done, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        # Target is not needed in the model input, but in the loss computation. Keep it here for now.
        target = data['target'].align_to('batch', 'complex', 'height', 'width').to(self.device)  # type: ignore
        # The first input_image in the iteration is the input_image with the mask applied and no first hidden state.
        input_image = data.pop('masked_image').to(self.device)  # type: ignore
        hidden_state = None
        output_image = None
        loss_dicts = []
        for rim_step in range(self.cfg.model.steps):
            # TODO: Target might not need to be copied.
            data = dict_to_device(data, self.device)
            sensitivity_map = data['sensitivity_map']
            # Some things can be done with the sensitivity map here, e.g. apply a u-net
            if 'sensitivity_model' in self.models:
                sensitivity_map = self.models['sensitivity_model'](sensitivity_map)
            # The sensitivity map needs to be normalized such that
            # So \sum_{i \in \text{coils}} S_i S_i^* = 1
            sensitivity_map_norm = modulus(sensitivity_map).sum('coil')
            data['sensitivity_map'] = safe_divide(sensitivity_map, sensitivity_map_norm)

            reconstruction_iter, hidden_state = self.model(
                **data,
                input_image=input_image,
                hidden_state=hidden_state,
            )
            # TODO: Unclear why this refining is needed.
            output_image = reconstruction_iter[-1].refine_names('batch', 'complex', 'height', 'width')

            loss_dict = {k: torch.tensor([0.], dtype=target.dtype).to(self.device) for k in loss_fns.keys()}
            for output_image_iter in reconstruction_iter:
                for k, v in loss_dict.items():
                    loss_dict[k] = v + loss_fns[k](
                        output_image_iter.rename(None), target.rename(None), reduction='mean'
                    )

            loss_dict = {k: v / len(reconstruction_iter) for k, v in loss_dict.items()}
            loss = sum(loss_dict.values())

            if self.model.training:
                if self.mixed_precision:
                    with amp.scale_loss(loss, self.__optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()  # type: ignore

            # Detach hidden state from computation graph, to ensure loss is only computed per RIM block.
            hidden_state = hidden_state.detach()
            input_image = output_image.detach()

            loss_dicts.append(detach_dict(loss_dict))  # Need to detach dict as this is only used for logging.

        # Add the loss dicts together over RIM steps, divide by the number of steps.
        loss_dict = reduce_list_of_dicts(loss_dicts, mode='sum', divisor=self.cfg.model.steps)
        return output_image, loss_dict

    def build_metrics(self) -> Dict:
        if not self.cfg.validation.metrics:
            self.logger.warning(f'No metrics defined in the config. Continuing without.')
            return {}

        metrics_dict = {
            curr_metric:
                str_to_class('direct.functionals', curr_metric) for curr_metric in self.cfg.validation.metrics
        }
        return metrics_dict

    def build_loss(self, **kwargs) -> Dict:
        # TODO: Cropper is a processing output tool.
        resolution = self.cfg.training.loss.crop
        # TODO(jt) Ideally this is also configurable:
        # - Do in steps (use insertation order)
        # Crop -> then loss.

        def l1_loss(source, target, reduction='mean'):
            return F.l1_loss(*self.cropper(source, target, resolution), reduction=reduction)

        def ssim_loss(source, target, reduction='mean'):
            source_abs, target_abs = self.cropper(source, target, resolution)
            return -SSIM(
                data_range=target_abs.max(), channel=1, reduction=reduction)(source_abs, target_abs)

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:
            loss_fn = curr_loss.function
            if loss_fn == 'l1_loss':
                loss_dict[curr_loss.function] = multiply_function(curr_loss.multiplier, l1_loss)
            elif loss_fn == 'ssim_loss':
                loss_dict[curr_loss.function] = multiply_function(curr_loss.multiplier, ssim_loss)
            else:
                raise ValueError(f'{loss_fn} not permissible.')

        return loss_dict

    @torch.no_grad()
    def evaluate(self,
                 data_loader: DataLoader,
                 loss_fns: Optional[Dict[str, Callable]],
                 volume_metrics: Optional[Dict[str, Callable]] = None,
                 crop=None, is_validation_process=True):

        self.model.eval()
        for curr_model in self.models:
            self.models[curr_model].eval()

        torch.cuda.empty_cache()

        # Variables required for evaluation.
        volume_metrics = volume_metrics if volume_metrics is not None else self.build_metrics()

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
            filenames = data.pop('filename')
            slice_nos = data.pop('slice_no')
            scaling_factors = data.pop('scaling_factor')

            # Compute output and loss.
            output, loss_dict = self._do_iteration(data, loss_fns)
            val_losses.append(loss_dict)

            # Output is complex-valued, and has to be cropped. This holds for both output and target.
            output_abs = self.process_output(
                output.refine_names('batch', 'complex', 'height', 'width').detach(),
                scaling_factors, crop)

            if is_validation_process:
                target_abs = self.process_output(
                    data['target'].refine_names('batch', 'height', 'width').detach(),
                    scaling_factors, crop)
            del output  # Explicitly call delete to clear memory.
            # TODO: Is a hack.

            # Aggregate volumes to be able to compute the metrics on complete volumes.
            for idx, filename in enumerate(filenames):
                if last_filename is None:
                    last_filename = filename  # First iteration last_filename is not set.
                # If the new filename is not the previous one, then we can reconstruct the volume as the sampling
                # is linear.
                # For the last case we need to check if we are at the last batch *and* at the last element in the batch.
                if filename != last_filename or (iter_idx + 1 == len(data_loader) and idx + 1 == len(data['target'])):
                    # Now we can ditch the reconstruction dict by reconstructing the volume,
                    # will take too much memory otherwise.
                    # TODO: Stack does not support named tensors.
                    volume = torch.stack([_[1].rename(None) for _ in reconstruction_output[last_filename]])
                    self.logger.info(f'Reconstructed {last_filename} (shape = {list(volume.shape)}).')
                    if is_validation_process:
                        target = torch.stack([_[1].rename(None) for _ in targets_output[last_filename]])
                        curr_metrics = {
                            metric_name: metric_fn(volume, target) for metric_name, metric_fn in volume_metrics.items()}
                        val_volume_metrics[last_filename] = curr_metrics
                        # Log the center slice of the volume
                        if len(visualize_slices) < self.cfg.tensorboard.num_images:
                            visualize_slices.append(normalize_image(volume[volume.shape[0] // 2]))
                            visualize_target.append(normalize_image(target[target.shape[0] // 2]))

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

        if not is_validation_process:
            return loss_dict, reconstruction_output

        return loss_dict, visualize_slices, visualize_target

    def process_output(self, data, scaling_factors=None, resolution=None):
        if scaling_factors is not None:
            data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(data.device)
        data = modulus_if_complex(data).rename(None)
        if len(data.shape) == 3:  # (batch, height, width)
            data = data.unsqueeze(1)  # Added channel dimension.

        if resolution is not None:
            data = center_crop(data, resolution).contiguous()

        return data

    @staticmethod
    def cropper(source, target, resolution=(320, 320)):
        source_abs = modulus(source.refine_names('batch', 'complex', 'height', 'width'))
        if resolution is not None or all([_ != 0 for _ in resolution]):
            source_abs = center_crop(source_abs, resolution).rename(None).unsqueeze(1)
            target_abs = center_crop(target, resolution)
        return source_abs, target_abs
