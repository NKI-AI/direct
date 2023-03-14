.. highlight:: shell

=====================
Adding your own model
=====================

To add a new model follow the steps below:

- Implement your custom model under :code:`direct/nn/<model_name>/<model_name>.py`. For example:

.. code-block:: python

    import torch
    from torch import nn
    from torch.nn import functional as F

    class MyMRIModel(nn.Module):
        """My custom MRI model."""

        def __init__(self, param1: param1_type, ...):
            """Inits :class:`MyMRIModel`.

            Parameters
            ----------
            param1 : param1_type
                ...
            ...
            """
            super().__init__()

        def my_method(self, ...) -> ...:
            pass

        @staticmethod
        def my_static_method(...) -> ...:
            pass

        def forward(
            self,
            masked_kspace: torch.Tensor,
            sampling_mask: torch.Tensor,
            sensitivity_map: torch.Tensor,
            ...
        ) -> torch.Tensor:
            """Computes forward pass of :class:`MyMRIModel`.

            Parameters
            ----------
            masked_kspace: torch.Tensor
                Masked k-space of shape (N, coil, height, width, complex=2).
            sampling_mask: torch.Tensor
                Sampling mask of shape (N, 1, height, width, 1).
            sensitivity_map: torch.Tensor
                Sensitivity map of shape (N, coil, height, width, complex=2).
            ...

            Returns
            -------
            out_image: torch.Tensor
                Output image of shape (N, height, width, complex=2).
            ...
            """

- Implement your custom model's engine under :code:`direct/nn/<model_name>/<model_name>_engine.py`. For example:

.. code-block:: python

    from __future__ import annotations

    from typing import Any, Callable, Dict, Optional, Tuple

    import torch
    from torch import nn

    from direct.config import BaseConfig
    from direct.nn.mri_models import MRIModelEngine


    class MyMRIModelEngine(MRIModelEngine):
        """:class:`MyMRIModel` Engine."""

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
            """Inits :class:`MyMRIModel`."""
            super().__init__(
                cfg,
                model,
                device,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                mixed_precision=mixed_precision,
                **models,
            )

        def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
            output_image = self.model(
                masked_kspace=data["masked_kspace"],
                sampling_mask=data["sampling_mask"],
                sensitivity_map=data["sensitivity_map"],
                ...=...
            )
            # ΟR
            output_kspace = self.model(
                masked_kspace=data["masked_kspace"],
                sampling_mask=data["sampling_mask"],
                sensitivity_map=data["sensitivity_map"],
                ...=...
            )
            ...
            return output_image, output_kspace


- Implement your custom model's config under :code:`direct/nn/<model_name>/config.py`. For example:

.. code-block:: python

    from dataclasses import dataclass

    from direct.config.defaults import ModelConfig


    @dataclass
    class MyMRIModelConfig(ModelConfig):
        param1: param1_type = param1_default_value
        ...
