# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import autocast

from direct.config import BaseConfig
from direct.data.transforms import modulus
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.utils import detach_dict, dict_to_device, reduce_list_of_dicts


class RIMEngine(MRIModelEngine):
    """Recurrent Inference Machine Engine."""

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
        """Inits :class:`RIMEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )
        self._complex_dim = 1

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

        # sensitivity_map of shape (batch, coil, height,  width, complex=2)
        sensitivity_map = data["sensitivity_map"].clone()

        if "noise_model" in self.models:
            raise NotImplementedError()

        data["sensitivity_map"] = self.compute_sensitivity_map(sensitivity_map)

        if self.cfg.model.scale_loglikelihood:  # type: ignore
            scaling_factor = 1.0 * self.cfg.model.scale_loglikelihood / (data["scaling_factor"] ** 2)  # type: ignore
            scaling_factor = scaling_factor.reshape(-1, 1)  # shape (batch, complex=1)
            self.logger.debug(f"Scaling factor is: {scaling_factor}")
        else:
            # Needs fixing.
            scaling_factor = torch.tensor([1.0]).to(sensitivity_map.device)  # shape (complex=1, )

        for _ in range(self.cfg.model.steps):  # type: ignore
            with autocast(enabled=self.mixed_precision):
                if input_image is not None:
                    input_image = input_image.permute(0, 2, 3, 1)
                reconstruction_iter, hidden_state = self.model(
                    **data,
                    input_image=input_image,
                    hidden_state=hidden_state,
                    loglikelihood_scaling=scaling_factor,
                )
                # reconstruction_iter: list with tensors of shape (batch, complex=2, height, width)
                # hidden_state has shape: (batch, num_hidden_channels, height, width, depth)

                output_image = reconstruction_iter[-1]  # shape (batch, complex=2, height, width)
                output_image = modulus(output_image, complex_axis=self._complex_dim)  # shape (batch, height,  width)

                loss_dict = {
                    k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()
                }
                regularizer_dict = {
                    k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
                }

                # TODO: This seems too similar not to be able to do this, perhaps a partial can help here
                for output_image_iter in reconstruction_iter:
                    for key, value in loss_dict.items():
                        loss_dict[key] = value + loss_fns[key](
                            modulus(output_image_iter, complex_axis=self._complex_dim),
                            **data,
                            reduction="mean",
                        )

                    for key, value in regularizer_dict.items():
                        regularizer_dict[key] = value + regularizer_fns[key](
                            modulus(output_image_iter, complex_axis=self._complex_dim),
                            **data,
                        )

                loss_dict = {k: v / len(reconstruction_iter) for k, v in loss_dict.items()}
                regularizer_dict = {k: v / len(reconstruction_iter) for k, v in regularizer_dict.items()}

                loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore

            if self.model.training:
                # TODO(gy): With steps >= 1, calling .backward(retain_grad=False) caused problems.
                #  Check with Jonas if it's ok.

                if (self.cfg.model.steps > 1) and (_ < self.cfg.model.steps - 1):  # type: ignore
                    self._scaler.scale(loss).backward(retain_graph=True)
                else:
                    self._scaler.scale(loss).backward()

            # Detach hidden state from computation graph, to ensure loss is only computed per RIM block.
            hidden_state = hidden_state.detach()  # shape: (batch, num_hidden_channels, [slice,] height, width, depth)
            input_image = output_image.detach()  # shape (batch, complex=2, [slice,] height,  width)

            loss_dicts.append(detach_dict(loss_dict))
            regularizer_dicts.append(
                detach_dict(regularizer_dict)
            )  # Need to detach dict as this is only used for logging.

        # Add the loss dicts together over RIM steps, divide by the number of steps.
        loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum", divisor=self.cfg.model.steps)  # type: ignore
        regularizer_dict = reduce_list_of_dicts(
            regularizer_dicts,
            mode="sum",
            divisor=self.cfg.model.steps,  # type: ignore
        )

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict},
        )
