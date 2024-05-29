# Copyright (c) DIRECT Contributors

"""Unet2d Models Engines for direct.

This module contains engines for Unet2d models, both for supervised and self-supervised learning.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import JSSLMRIModelEngine, SSLMRIModelEngine


class Unet2dEngine(MRIModelEngine):
    """Unet2d Model Engine.

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
        """Inits :class:`Unet2dEngine`.

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

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, None]:
        """Forward function for :class:`Unet2dEngine`.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary containing the following keys: "masked_kspace" and "sensitivity_map"
            if image initialization is "sense".

        Returns
        -------
        tuple[torch.Tensor, None]
            Prediction of image and None for k-space.
        """

        sensitity_map = (
            data["sensitivity_map"] if self.cfg.model.image_initialization == "sense" else None  # type: ignore
        )

        output_image = self.model(masked_kspace=data["masked_kspace"], sensitivity_map=sensitity_map)
        output_image = T.modulus(output_image)

        output_kspace = None

        return output_image, output_kspace


class Unet2dSSLEngine(SSLMRIModelEngine):
    """SSL Unet2d Model Engine.

    Used for supplementary experiments for U-Net model with SLL in the JSSL paper [1].

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
        """Inits :class:`Unet2dSSLEngine`.

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

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, None]:
        """Forward function for :class:`Unet2dSSLEngine`.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary containing the following keys: "input_kspace" if training,
            otherwise "masked_kspace". Also contains "sensitivity_map" if image initialization is "sense".

        Returns
        -------
        tuple[torch.Tensor, None]
            Prediction of image and None for k-space.
        """
        # Get the k-space and mask which differ during training and inference for SSL
        kspace = data["input_kspace"] if self.model.training else data["masked_kspace"]

        sensitity_map = (
            data["sensitivity_map"] if self.cfg.model.image_initialization == "sense" else None  # type: ignore
        )

        output_image = self.model(masked_kspace=kspace, sensitivity_map=sensitity_map)
        output_kspace = None

        return output_image, output_kspace


class Unet2dJSSLEngine(JSSLMRIModelEngine):
    """JSSL Unet2d Model Engine.

    Used for supplementary experiments for U-Net model with JSLL in the JSSL paper [1].

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
        """Inits :class:`Unet2dSSLEngine`.

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

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, None]:
        """Forward function for :class:`Unet2dJSSLEngine`.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary containing the following keys: "is_ssl" indicating SSL sample, "input_kspace" if SSL
            training, otherwise "masked_kspace". Also contains "sensitivity_map" if image initialization is "sense".

        Returns
        -------
        tuple[torch.Tensor, None]
            Prediction of image and None for k-space.
        """
        is_ssl = data["is_ssl"][0]

        # Get the k-space and mask which differ if SSL training or supervised training
        # The also differ during training and inference for SSL
        if is_ssl and self.model.training:
            kspace = data["input_kspace"]
        else:
            kspace = data["masked_kspace"]

        sensitity_map = (
            data["sensitivity_map"] if self.cfg.model.image_initialization == "sense" else None  # type: ignore
        )

        output_image = self.model(masked_kspace=kspace, sensitivity_map=sensitity_map)
        output_kspace = None

        return output_image, output_kspace


class Unet3dEngine(MRIModelEngine):
    """Unet3d Model Engine.

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
        """Inits :class:`Unet3dEngine`.

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

    def forward_function(self, data: dict[str, Any]) -> tuple[torch.Tensor, None]:
        """Forward function for :class:`Unet3dEngine`.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary containing the following keys: "masked_kspace" and "sensitivity_map"
            if image initialization is "sense".

        Returns
        -------
        tuple[torch.Tensor, None]
            Prediction of image and None for k-space.
        """

        sensitity_map = (
            data["sensitivity_map"] if self.cfg.model.image_initialization == "sense" else None  # type: ignore
        )

        output_image = self.model(masked_kspace=data["masked_kspace"], sensitivity_map=sensitity_map)
        output_image = T.modulus(output_image)

        output_kspace = None

        return output_image, output_kspace
