# Copyright (c) DIRECT Contributors

"""SSL MRI model engines of DIRECT."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.cuda.amp import autocast

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device, normalize_image
from direct.utils.events import get_event_storage

__all__ = ["SSLMRIModelEngine"]


class SSLMRIModelEngine(MRIModelEngine):
    r"""Base Engine for SSL MRI models.

    This engine is used for training models that are trained with self-supervised learning. During training,
    the loss is computed as follows:

    .. math::

        \mathcal{L}\big(\mathcal{A}_{\text{tar}}(x_{\text{out}}), y_{\text{tar}}\big)

    where :math:`x_{\text{out}}=f_{\theta}(y_{\text{inp}})` and :math:`y_{\text{inp}} + y_{\text{tar}}=\tilde{y}`
    are splits of the original measured k-space :math:`\tilde{y}` via two (disjoint or not) sub-sampling operators
    :math:`y_{\text{inp}}=U_{\text{inp}}(\tilde{y})` and :math:`y_{\text{tar}}=U_{\text{tar}}(\tilde{y})` and
    :math:`U_{\text{inp}} + U_{\text{tar}} = U`, where :math:`U` is the original sub-sampling operator.

    During inference, output is computed as :math:`(\mathbb{1} - U)f_{\theta}(\tilde{y}) + \tilde{y}`.

    Note
    ----
    This engine also implements the `log_first_training_example_and_model` method to log the first training example
    which differs from the corresponding method of the base :class:`MRIModelEngine`.
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
        """Inits :class:`SSLMRIModelEngine`.

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
        **models: nn.Module
            Additional models.
        """
        super().__init__(
            cfg=cfg,
            model=model,
            device=device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def log_first_training_example_and_model(self, data: dict[str, Any]) -> None:
        """Logs the first training example for SSL-based MRI models.

        This differs from the corresponding method of the base :class:`MRIModelEngine` as it requires the input
        and target sampling masks to be logged as well and to create the actual sampling mask.

        Parameters
        ----------
        data: dict[str, Any]
            Dictionary containing the data. The dictionary should contain the following keys:
            - "filename": Filename of the data.
            - "slice_no": Slice number of the data.
            - "input_sampling_mask": Sampling mask for the input k-space.
            - "target_sampling_mask": Sampling mask for the target k-space.
            - "target": Target image. This is the reconstruction of the target k-space (i.e. subsampled using
              the target_sampling_mask).
            - "initial_image": Initial image.
        """
        storage = get_event_storage()

        self.logger.info(f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}.")

        first_input_sampling_mask = data["input_sampling_mask"][0][0]
        first_target_sampling_mask = data["target_sampling_mask"][0][0]
        storage.add_image("train/input_mask", first_input_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image("train/target_mask", first_target_sampling_mask[..., 0].unsqueeze(0))
        first_sampling_mask = first_target_sampling_mask | first_input_sampling_mask

        first_target = data["target"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            num_slices = first_target.shape[0]
            first_target = first_target[: num_slices // 2]
            first_target = torch.cat([first_target[_] for _ in range(first_target.shape[0])], dim=-1)
        elif self.ndim > 3:
            raise NotImplementedError

        storage.add_image("train/mask", first_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image(
            "train/target",
            normalize_image(first_target.unsqueeze(0)),
        )
        self.write_to_logs()

    @abstractmethod
    def forward_function(self, data: dict[str, Any]) -> tuple[TensorOrNone, TensorOrNone]:
        """Must be implemented by child classes.

        Parameters
        ----------
        data: dict[str, Any]

        Raises
        ------
        NotImplementedError
            Must be implemented by child class.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: Optional[dict[str, Callable]] = None,
        regularizer_fns: Optional[dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """This function is a base `_do_iteration` method for SSL-based MRI models.

        It assumes that the `forward_function` is implemented by the child class which should return the output
        image and/or output k-space.

        It assumes different behavior for training and inference. During training, it expects the input data to contain
        keys "input_kspace" and "input_sampling_mask" and during inference, it expects the input data to contain
        keys "masked_kspace" and "sampling_mask".

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary. The dictionary should contain the following keys:
            - "input_kspace" if training, otherwise "masked_kspace".
            - "input_sampling_mask" if training, otherwise "sampling_mask".
            - "target_sampling_mask": Sampling mask for the target k-space if training.
            - "sensitivity_map": Sensitivity map.
            - "target": Target image.
            - "padding": Padding, optionally.
        loss_fns : Optional[dict[str, Callable]], optional
            Loss functions, optional.
        regularizer_fns : Optional[dict[str, Callable]], optional
            Regularizer functions, optional.

        Returns
        -------
        DoIterationOutput
            Output of the iteration.

        Raises
        ------
        ValueError
            If both output_image and output_kspace from the forward function are None.
        """

        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        data = dict_to_device(data, self.device)

        # Get the k-space and mask which differ during training and inference for SSL
        kspace = data["input_kspace"] if self.model.training else data["masked_kspace"]
        mask = data["input_sampling_mask"] if self.model.training else data["sampling_mask"]

        # Initialize loss and regularizer dictionaries
        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
        regularizer_dict = {
            k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
        }

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            # Compute sensitivity map
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])
            # Forward pass via the forward function of the model engine
            output_image, output_kspace = self.forward_function(data)
            # Some models output images, so transform them to k-space domain if they are not already there
            if output_kspace is None:
                if output_image is None:
                    raise ValueError(
                        "Both output_image and output_kspace cannot be None. "
                        "The `forward_function` must return at least one of them."
                    )
                # Predict only on unmeasured locations
                output_kspace = self._forward_operator(output_image, data["sensitivity_map"], ~mask)
            else:
                # Predict only on unmeasured locations
                output_kspace = T.apply_mask(output_kspace, ~mask, return_mask=False)
            # Data consistency followed by padding if it exists
            output_kspace = T.apply_padding(kspace + output_kspace, padding=data.get("padding", None))

            if self.model.training:
                # SSL: project the predicted k-space to target k-space, i.e. predict locations only in target k-space
                output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)

            # Compute loss and regularizer in k-space domain
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, None, output_kspace)
            regularizer_dict = self.compute_loss_on_data(regularizer_dict, regularizer_fns, data, None, output_kspace)

            # Compute image via SENSE reconstruction
            output_image = T.modulus(
                T.reduce_operator(
                    self.backward_operator(output_kspace, dim=self._spatial_dims),
                    data["sensitivity_map"],
                    self._coil_dim,
                )
            )

            # Compute loss and regularizer loss in image domain
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, None)
            regularizer_dict = self.compute_loss_on_data(regularizer_dict, regularizer_fns, data, output_image, None)

            # Compute total loss
            loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore

            # Backward pass
            if self.model.training:
                self._scaler.scale(loss).backward()

            loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.
            regularizer_dict = detach_dict(regularizer_dict)

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict} if self.model.training else {},
        )


class JSSLMRIModelEngine(MRIModelEngine):
    r"""Base Engine for JSSL MRI models.

    This engine is used for training models that are trained with joint supervised and self-supervised learning (JSSL).
    During training, for self-supervised samples the loss is computed as in :class:`SSLMRIModelEngine` and for
    supervised samples the loss is computed as normal supervised MRI learning.

    During inference, output is computed as :math:`(\mathbb{1} - U)f_{\theta}(\tilde{y}) + \tilde{y}`.

    Note
    ----
    This engine also implements the `log_first_training_example_and_model` method to log the first training example
    which differs from the corresponding method of the base :class:`MRIModelEngine`.
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
        """Inits :class:`JSSLMRIModelEngine`.

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
        **models: nn.Module
            Additional models.
        """
        super().__init__(
            cfg=cfg,
            model=model,
            device=device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def log_first_training_example_and_model(self, data: dict[str, Any]) -> None:
        """Logs the first training example for SSL-based MRI models.

        This differs from the corresponding method of the base :class:`MRIModelEngine` as it requires the input
        and target sampling masks to be logged as well and to create the actual sampling mask.

        Parameters
        ----------
        data: dict[str, Any]
            Dictionary containing the data. The dictionary should contain the following keys:
            - "filename": Filename of the data.
            - "slice_no": Slice number of the data.
            - "input_sampling_mask": Sampling mask for the input k-space.
            - "target_sampling_mask": Sampling mask for the target k-space.
            - "target": Target image. This is the reconstruction of the target k-space (i.e. subsampled using
              the target_sampling_mask).
            - "initial_image": Initial image.
        """
        storage = get_event_storage()

        self.logger.info(f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}.")

        if "input_sampling_mask" in data:
            first_input_sampling_mask = data["input_sampling_mask"][0][0]
            first_target_sampling_mask = data["target_sampling_mask"][0][0]
            storage.add_image("train/input_mask", first_input_sampling_mask[..., 0].unsqueeze(0))
            storage.add_image("train/target_mask", first_target_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = first_target_sampling_mask | first_input_sampling_mask

        else:
            first_sampling_mask = data["sampling_mask"][0][0]

        first_target = data["target"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            num_slices = first_target.shape[0]
            first_target = first_target[: num_slices // 2]
            first_target = torch.cat([first_target[_] for _ in range(first_target.shape[0])], dim=-1)
        elif self.ndim > 3:
            raise NotImplementedError

        storage.add_image("train/mask", first_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image(
            "train/target",
            normalize_image(first_target.unsqueeze(0)),
        )
        self.write_to_logs()

    @abstractmethod
    def forward_function(self, data: dict[str, Any]) -> tuple[TensorOrNone, TensorOrNone]:
        """Must be implemented by child classes.

        Parameters
        ----------
        data: dict[str, Any]

        Raises
        ------
        NotImplementedError
            Must be implemented by child class.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: dict[str, Any],
        loss_fns: Optional[dict[str, Callable]] = None,
        regularizer_fns: Optional[dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """This function is a base `_do_iteration` method for JSSL-based MRI models.

        Returns
        -------
        DoIterationOutput
            Output of the iteration.

        It assumes that the `forward_function` is implemented by the child class which should return the output
        image and/or output k-space.

        It assumes different behavior for training and inference. During SSL training, it expects the input data
        to contain keys "input_kspace" and "input_sampling_mask", otherwise, it expects the input data to contain
        keys "masked_kspace" and "sampling_mask".

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary. The dictionary should contain the following keys:
            - "is_ssl_training": Boolean indicating if the sample is for SSL training.
            - "input_kspace" if SSL training, otherwise "masked_kspace".
            - "input_sampling_mask" if SSL training, otherwise "sampling_mask".
            - "target_sampling_mask": Sampling mask for the target k-space if SSL training.
            - "sensitivity_map": Sensitivity map.
            - "target": Target image.
            - "padding": Padding, optionally.
        loss_fns : Optional[dict[str, Callable]], optional
            Loss functions, optional.
        regularizer_fns : Optional[dict[str, Callable]], optional
            Regularizer functions, optional.

        Raises
        ------
        ValueError
            If both output_image and output_kspace from the forward function are None.
        """

        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        data = dict_to_device(data, self.device)

        # Get a boolean indicating if the sample is for SSL training
        # This will expect the input data to contain the keys "input_kspace" and "input_sampling_mask" if SSL training
        is_ssl_training = data["is_ssl_training"][0]

        # Get the k-space and mask which differ if SSL training or supervised training
        # The also differ during training and inference for SSL
        if is_ssl_training and self.model.training:
            kspace, mask = data["input_kspace"], data["input_sampling_mask"]
        else:
            kspace, mask = data["masked_kspace"], data["sampling_mask"]

        # Initialize loss and regularizer dictionaries
        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
        regularizer_dict = {
            k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
        }

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            # Compute sensitivity map
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])
            # Forward pass via the forward function of the model engine
            output_image, output_kspace = self.forward_function(data)

            # Some models output images, so transform them to k-space domain if they are not already there
            if output_kspace is None:
                if output_image is None:
                    raise ValueError(
                        "Both output_image and output_kspace cannot be None. "
                        "The `forward_function` must return at least one of them."
                    )
                # Predict only on unmeasured locations using output image if output k-space is None
                output_kspace = self._forward_operator(output_image, data["sensitivity_map"], ~mask)
            else:
                # Predict only on unmeasured locations by applying the complement of the mask if output k-space exists
                output_kspace = T.apply_mask(output_kspace, ~mask, return_mask=False)
            # Data consistency (followed by padding if it exists)
            output_kspace = T.apply_padding(kspace + output_kspace, padding=data.get("padding", None))

            if self.model.training and is_ssl_training:
                # SSL: project the predicted k-space to target k-space
                output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)

            # Compute loss and regularizer loss in k-space domain
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, None, output_kspace)
            regularizer_dict = self.compute_loss_on_data(regularizer_dict, regularizer_fns, data, None, output_kspace)

            # Compute image via SENSE reconstruction
            output_image = T.modulus(
                T.reduce_operator(
                    self.backward_operator(output_kspace, dim=self._spatial_dims),
                    data["sensitivity_map"],
                    self._coil_dim,
                )
            )

            # Compute loss and regularizer loss in image domain
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, None)
            regularizer_dict = self.compute_loss_on_data(regularizer_dict, regularizer_fns, data, output_image, None)

            # Compute total loss
            loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore

            # Backward pass
            if self.model.training:
                self._scaler.scale(loss).backward()

            loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.
            regularizer_dict = detach_dict(regularizer_dict)

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict} if self.model.training else {},
        )
