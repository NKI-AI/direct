# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine


class ImageDomainMRIViTEngine(MRIModelEngine):
    """MRI ViT Model Engine for Image Domain."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`ImageDomainMRIViT`.

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
        """Forward function for ImageDomainMRIViT.

        Parameters
        ----------
        data : dict[str, Any]
            Input data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output image and output k-space.
        """
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

        data = self.perform_sampling(data)

        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, slice/time, height,  width, complex[=2])

        output_kspace = data["masked_kspace"] + T.apply_mask(
            T.apply_padding(
                self.forward_operator(
                    T.expand_operator(output_image, data["sensitivity_map"], dim=self._coil_dim),
                    dim=self._spatial_dims,
                ),
                padding=data.get("padding", None),
            ),
            ~data["sampling_mask"],
            return_mask=False,
        )

        return output_image, output_kspace


class ImageDomainMRIViT2DEngine(ImageDomainMRIViTEngine):
    """MRI ViT Model Engine for Image Domain 2D."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`ImageDomainMRIViT2D`.

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

        self._spatial_dims = (2, 3)


class ImageDomainMRIViT3DEngine(ImageDomainMRIViTEngine):
    """MRI ViT Model Engine for Image Domain 3D."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`ImageDomainMRIViT3D`.

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


class KSpaceDomainMRIViTEngine(MRIModelEngine):
    """MRI ViT Model Engine for K-Space Domain."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`KSpaceDomainMRIViT`.

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
        """Forward function for KSpaceDomainMRIViT.

        Parameters
        ----------
        data : dict[str, Any]
            Input data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output image and output k-space.
        """
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

        data = self.perform_sampling(data)

        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["sampling_mask"],
        )  # shape (batch, slice/time, height,  width, complex[=2])

        output_kspace = data["masked_kspace"] + T.apply_mask(
            T.apply_padding(
                self.forward_operator(
                    T.expand_operator(output_image, data["sensitivity_map"], dim=self._coil_dim),
                    dim=self._spatial_dims,
                ),
                padding=data.get("padding", None),
            ),
            ~data["sampling_mask"],
            return_mask=False,
        )

        return output_image, output_kspace


class KSpaceDomainMRIViT2DEngine(KSpaceDomainMRIViTEngine):
    """MRI ViT Model Engine for K-Space Domain 2D."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`KSpaceDomainMRIViT2D`.

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

        self._spatial_dims = (2, 3)


class KSpaceDomainMRIViT3DEngine(KSpaceDomainMRIViTEngine):
    """MRI ViT Model Engine for K-Space Domain 3D."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        backward_operator: Optional[Callable[[tuple[Any, ...]], torch.Tensor]] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`KSpaceDomainMRIViT3D`.

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
