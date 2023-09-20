# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine


class VariationalUFormerEngine(MRIModelEngine):
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
        """Inits :class:`VariationalUFormerEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
            padding=data.get("padding", None),
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims), dim=self._coil_dim
        )
        return output_image, output_kspace


class MRIUFormerEngine(MRIModelEngine):
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
        """Inits :class:`MRIUFormerEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
            padding=data.get("padding", None),
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims), dim=self._coil_dim
        )
        return output_image, output_kspace


class ImageDomainUFormerEngine(MRIUFormerEngine):
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
        """Inits :class:`ImageDomainUFormerEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )


class KSpaceDomainUFormerEngine(MRIUFormerEngine):
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
        """Inits :class:`KSpaceDomainUFormerEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )


class MRITransformerEngine(MRIModelEngine):
    """MRI Transformer Engine."""

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
        """Inits :class:`ImageDomainTransformerEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_kspace = T.apply_padding(
            self.backward_operator(
                T.expand_operator(data=output_image, sensitivity_map=data["sensitivity_map"], dim=self._coil_dim),
                dim=self._spatial_dims,
            ),
            data.get("padding", None),
        )
        output_kspace = torch.where(data["sampling_mask"] == 0, output_kspace, data["masked_kspace"])

        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )

        return output_image, output_kspace


class ImageDomainVisionTransformerEngine(MRIModelEngine):
    """Image Domain Transformer Engine."""

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
        """Inits :class:`ImageDomainTransformerEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_kspace = T.apply_padding(
            self.backward_operator(
                T.expand_operator(data=output_image, sensitivity_map=data["sensitivity_map"], dim=self._coil_dim),
                dim=self._spatial_dims,
            ),
            data.get("padding", None),
        )
        output_kspace = torch.where(data["sampling_mask"] == 0, output_kspace, data["masked_kspace"])

        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )

        return output_image, output_kspace
