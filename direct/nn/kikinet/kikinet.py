# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn

import direct.data.transforms as T
from direct.nn.conv.conv import Conv2d
from direct.nn.conv.modulated import ModConvActivation, ModConvType, ModulationParams
from direct.nn.crossdomain.multicoil import MultiCoil
from direct.nn.didn.didn import DIDN
from direct.nn.mwcnn.mwcnn import MWCNN
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.types import FFTOperator


class KIKINet(nn.Module):
    """Based on KIKINet implementation [1]_. Modified to work with multi-coil k-space data.

    Supports conditional weight modulation as proposed in [2]_.

    References
    ----------

    .. [1] Eo, Taejoon, et al. "KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled
        Magnetic Resonance Images." Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188-201.
        https://doi.org/10.1002/mrm.27201.

    .. [2] Moriakov, N., Yiasemis, G., Sonke, J.-J. & Teuwen, J. (2026). Conditional Learned Reconstruction for
        Medical Imaging. Proceedings of The 9th International Conference on Medical Imaging with Deep Learning,
        PMLR 315:754-780. https://proceedings.mlr.press/v315/moriakov26a.html
    """

    def __init__(
        self,
        forward_operator: FFTOperator,
        backward_operator: FFTOperator,
        image_model_architecture: str = "MWCNN",
        kspace_model_architecture: str = "DIDN",
        num_iter: int = 2,
        normalize: bool = False,
        conv_modulation: ModConvType = ModConvType.NONE,
        aux_in_features: int | None = None,
        fc_hidden_features: tuple[int] | int | None = None,
        fc_groups: int = 1,
        fc_activation: ModConvActivation = ModConvActivation.SIGMOID,
        num_weights: int | None = None,
        **kwargs,
    ):
        """Inits :class:`KIKINet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        image_model_architecture: str
            Image model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        kspace_model_architecture: str
            Kspace model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        num_iter: int
            Number of unrolled iterations.
        normalize: bool
            If true, input is normalised based on input scaling_factor.
        conv_modulation : ModConvType
            Modulation type for convolutional layers. Default: ModConvType.NONE.
        aux_in_features : int, optional
            Number of features in the auxiliary input for modulation.
        fc_hidden_features : int or tuple of int, optional
            Hidden features in the modulation MLP.
        fc_groups : int
            Groups for modulation MLP output. Default: 1.
        fc_activation : ModConvActivation
            Activation after modulation MLP. Default: ModConvActivation.SIGMOID.
        num_weights : int, optional
            Number of weight bases for ModConvType.SUM.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        super().__init__()

        modulation_params = ModulationParams(
            modulation=conv_modulation,
            aux_in_features=aux_in_features,
            fc_hidden_features=fc_hidden_features,
            fc_groups=fc_groups,
            fc_activation=fc_activation,
            num_weights=num_weights,
        )

        image_model: nn.Module
        if image_model_architecture == "MWCNN":
            image_model = MWCNN(
                input_channels=2,
                first_conv_hidden_channels=kwargs.get("image_mwcnn_hidden_channels", 32),
                num_scales=kwargs.get("image_mwcnn_num_scales", 4),
                bias=kwargs.get("image_mwcnn_bias", False),
                batchnorm=kwargs.get("image_mwcnn_batchnorm", False),
                modulation_params=modulation_params,
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if image_model_architecture == "UNET" else NormUnetModel2d
            image_model = unet(
                in_channels=2,
                out_channels=2,
                num_filters=kwargs.get("image_unet_num_filters", 8),
                num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("image_unet_dropout_probability", 0.0),
                modulation_params=modulation_params,
            )
        else:
            raise NotImplementedError(
                f"KIKINet is currently implemented only with image_model_architecture == 'MWCNN', 'UNET' or 'NORMUNET'."
                f" Got {image_model_architecture}."
            )

        kspace_model: nn.Module
        if kspace_model_architecture == "CONV":
            kspace_model = Conv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=kwargs.get("kspace_conv_hidden_channels", 16),
                n_convs=kwargs.get("kspace_conv_n_convs", 4),
                batchnorm=kwargs.get("kspace_conv_batchnorm", False),
                modulation_params=modulation_params,
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=kwargs.get("kspace_didn_hidden_channels", 16),
                num_dubs=kwargs.get("kspace_didn_num_dubs", 6),
                num_convs_recon=kwargs.get("kspace_didn_num_convs_recon", 9),
                modulation_params=modulation_params,
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if kspace_model_architecture == "UNET" else NormUnetModel2d
            kspace_model = unet(
                in_channels=2,
                out_channels=2,
                num_filters=kwargs.get("kspace_unet_num_filters", 8),
                num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("kspace_unet_dropout_probability", 0.0),
                modulation_params=modulation_params,
            )
        else:
            raise NotImplementedError(
                f"KIKINet is currently implemented for kspace_model_architecture == 'CONV', 'DIDN',"
                f" 'UNET' or 'NORMUNET'. Got kspace_model_architecture == {kspace_model_architecture}."
            )

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

        self.image_model_list = nn.ModuleList([image_model] * num_iter)
        self.kspace_model_list = nn.ModuleList([MultiCoil(kspace_model, self._coil_dim)] * num_iter)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.normalize = normalize
        self.conv_modulation = conv_modulation

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        scaling_factor: torch.Tensor | None = None,
        auxiliary_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`KIKINet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        scaling_factor: Optional[torch.Tensor]
            Scaling factor of shape (N,). If None, no scaling is applied. Default: None.
        auxiliary_data: torch.Tensor, optional
            Auxiliary data for modulation of shape (N, aux_in_features).

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """

        kspace = masked_kspace.clone()
        if self.normalize and scaling_factor is not None:
            kspace = kspace / (scaling_factor**2).view(-1, 1, 1, 1, 1)

        for idx in range(self.num_iter):
            kspace_permuted = kspace.permute(0, 1, 4, 2, 3)
            if self.conv_modulation != ModConvType.NONE:
                kspace = self.kspace_model_list[idx](kspace_permuted, auxiliary_data).permute(0, 1, 3, 4, 2)
            else:
                kspace = self.kspace_model_list[idx](kspace_permuted).permute(0, 1, 3, 4, 2)

            image = T.reduce_operator(
                self.backward_operator(
                    torch.where(
                        sampling_mask == 0,
                        torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                        kspace,
                    ).contiguous(),
                    self._spatial_dims,
                ),
                sensitivity_map,
                self._coil_dim,
            )

            if self.conv_modulation != ModConvType.NONE:
                image = self.image_model_list[idx](image.permute(0, 3, 1, 2), auxiliary_data).permute(0, 2, 3, 1)
            else:
                image = self.image_model_list[idx](image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            if idx < self.num_iter - 1:
                kspace = torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=image.dtype).to(image.device),
                    self.forward_operator(
                        T.expand_operator(image, sensitivity_map, self._coil_dim),
                        dim=self._spatial_dims,
                    ),
                )

        if self.normalize and scaling_factor is not None:
            image = image * (scaling_factor**2).view(-1, 1, 1, 1)

        return image
