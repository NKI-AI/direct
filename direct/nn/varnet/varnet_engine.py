# Copyright (c) DIRECT Contributors

"""Engines for End-to-End Variational Network model.

Includes supervised, self-supervised and joint supervised and self-supervised learning engines.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import JSSLMRIModelEngine, SSLMRIModelEngine


class EndToEndVarNetEngine(MRIModelEngine):
    """End-to-End Variational Network Engine.

    Parameters
    ----------
    cfg: BaseConfig
        Configuration file.
    model: nn.Module
        Model.
    device: str
        Device. Can be "cuda:{idx}" or "cpu".
    forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The forward operator. Default: None.
    backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The backward operator. Default: None.
    mixed_precision: bool
        Use mixed precision. Default: False.
    **models: nn.Module
        Additional models.
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
        """Inits :class:`EndToEndVarNetEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )  # shape (batch, height,  width)

        return output_image, output_kspace


class EndToEndVarNetSSLEngine(SSLMRIModelEngine):
    """Self-supervised Learning End-to-End Variational Network Engine.

    Used for supplementary experiments for End-to-End Variational Network model with SLL in the JSSL paper [1].

    Parameters
    ----------
    cfg: BaseConfig
        Configuration file.
    model: nn.Module
        Model.
    device: str
        Device. Can be "cuda:{idx}" or "cpu".
    forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The forward operator. Default: None.
    backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The backward operator. Default: None.
    mixed_precision: bool
        Use mixed precision. Default: False.
    **models: nn.Module
        Additional models.

    References
    ----------
    .. [1] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
        Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
        https://doi.org/10.48550/arXiv.2311.15856.
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
        """Inits :class:`EndToEndVarNetSSLEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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

    def forward_function(self, data: dict[str, Any]) -> tuple[None, torch.Tensor]:
        """Forward function for :class:`EndToEndVarNetSSLEngine`.

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary. Should contain the following keys:
            - "input_kspace" if training, "masked_kspace" if inference
            - "input_sampling_mask" if training, "sampling_mask" if inference
            - "sensitivity_map"

        Returns
        -------
        tuple[None, torch.Tensor]
            None for image and output k-space.
        """

        kspace = data["input_kspace"] if self.model.training else data["masked_kspace"]
        mask = data["input_sampling_mask"] if self.model.training else data["sampling_mask"]

        output_kspace = self.model(
            masked_kspace=kspace,
            sampling_mask=mask,
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = None

        return output_image, output_kspace


class EndToEndVarNetJSSLEngine(JSSLMRIModelEngine):
    """Joint Supervised and Self-supervised Learning End-to-End Variational Network Engine.

    Used for supplementary experiments for End-to-End Variational Network model with JSLL in the JSSL paper [1].

    Parameters
    ----------
    cfg: BaseConfig
        Configuration file.
    model: nn.Module
        Model.
    device: str
        Device. Can be "cuda:{idx}" or "cpu".
    forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The forward operator. Default: None.
    backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The backward operator. Default: None.
    mixed_precision: bool
        Use mixed precision. Default: False.
    **models: nn.Module
        Additional models.

    References
    ----------
    .. [1] Yiasemis, G., Moriakov, N., Sánchez, C.I., Sonke, J.-J., Teuwen, J.: JSSL: Joint Supervised and
        Self-supervised Learning for MRI Reconstruction, http://arxiv.org/abs/2311.15856, (2023).
        https://doi.org/10.48550/arXiv.2311.15856.
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
        """Inits :class:`EndToEndVarNetJSSLEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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

    def forward_function(self, data: dict[str, Any]) -> tuple[None, torch.Tensor]:
        """Forward function for :class:`EndToEndVarNetJSSLEngine`.

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary. Should contain the following keys:
            - "is_ssl" boolean tensor indicating if training is SSL
            - "input_kspace" if training and training is SSL, "masked_kspace" if inference
            - "input_sampling_mask" if training and training is SSL, "sampling_mask" if inference
            - "sensitivity_map"

        Returns
        -------
        tuple[None, torch.Tensor]
            None for image and output k-space.
        """

        if data["is_ssl"][0] and self.model.training:
            kspace, mask = data["input_kspace"], data["input_sampling_mask"]
        else:
            kspace, mask = data["masked_kspace"], data["sampling_mask"]

        output_kspace = self.model(
            masked_kspace=kspace,
            sampling_mask=mask,
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = None

        return output_image, output_kspace


class EndToEndVarNet3DEngine(MRIModelEngine):
    """End-to-End Variational Network Engine for 3D data.

    Parameters
    ----------
    cfg: BaseConfig
        Configuration file.
    model: nn.Module
        Model.
    device: str
        Device. Can be "cuda:{idx}" or "cpu".
    forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The forward operator. Default: None.
    backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
        The backward operator. Default: None.
    mixed_precision: bool
        Use mixed precision. Default: False.
    **models: nn.Module
        Additional models.
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
        """Inits :class:`EndToEndVarNet3DEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
            The forward operator. Default: None.
        backward_operator: Callable[[tuple[Any, ...]], torch.Tensor], optional
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

        self._spatial_dims = (3, 4)

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )  # shape (batch, slice/time, height,  width)

        return output_image, output_kspace
