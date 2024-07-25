"""Registration models for direct registration."""

import torch
import torch.nn as nn

from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d
from direct.registration.registration import DISCPLACEMENT_FIELD_2D_DIMENSIONS
from direct.registration.warp import warp


class UnetRegistrationModel(nn.Module):

    def __init__(
        self,
        max_seq_len: int,
        unet_num_filters: int = 16,
        unet_num_pool_layers: int = 4,
        unet_dropout_probability: float = 0.0,
        unet_normalized: bool = False,
        warp_num_itegration_steps: int = 1,
    ) -> None:
        """Inits :class:`UnetRegistrationModel`.

        Parameters
        ----------
        max_seq_len : int
            Maximum sequence length expected in the moving image.
        unet_num_filters : int
            Number of filters in the first layer of the UNet. Default: 16.
        unet_num_pool_layers : int
            Number of pooling layers in the UNet. Default: 4.
        unet_dropout_probability : float
            Dropout probability. Default: 0.0.
        unet_normalized : bool
            Whether to use normalization in the UNet. Default: False.
        warp_num_itegration_steps : int
            Number of integration steps to perform when warping the moving image. Default: 1.
        """
        super().__init__()

        self.max_seq_len = max_seq_len

        self.model = (UnetModel2d if not unet_normalized else NormUnetModel2d)(
            in_channels=max_seq_len + 1,
            out_channels=max_seq_len * DISCPLACEMENT_FIELD_2D_DIMENSIONS,
            num_filters=unet_num_filters,
            num_pool_layers=unet_num_pool_layers,
            dropout_probability=unet_dropout_probability,
        )
        self.warp_num_itegration_steps = warp_num_itegration_steps

    def forward(self, moving_image: torch.Tensor, reference_image: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`UnetRegistrationModel`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Moving image tensor of shape (batch_size, seq_len, height, width).
        reference_image : torch.Tensor
            Reference image tensor of shape (batch_size, height, width).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the warped image tensor of shape (batch_size, seq_len, height, width)
            and the displacement field tensor of shape (batch_size, seq_len, 2, height, width).
        """
        batch_size, seq_len, height, width = moving_image.shape

        # Pad the moving image to the maximum sequence length
        x = nn.functional.pad(moving_image, (0, 0, 0, 0, 0, self.max_seq_len - moving_image.shape[1]))
        # Add the reference image as the first channel
        x = torch.cat((reference_image.unsqueeze(1), x), dim=1)

        # Forward pass through the model
        displacement_field = self.model(x)

        # Model outputs the displacement field for each time step with 2 channels (x and y displacements)
        displacement_field = displacement_field.reshape(
            batch_size, self.max_seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )  # (batch_size, max_seq_len, 2, height, width)

        # Crop the displacement field to the actual sequence length
        displacement_field = displacement_field[:, :seq_len]  # (batch_size, seq_len, 2, height, width)

        # Reshape the displacement field and moving image to be compatible with the warp module
        displacement_field = displacement_field.reshape(
            batch_size * seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width
        )
        moving_image = moving_image.reshape(batch_size * seq_len, 1, height, width)

        # Warp the moving image
        warped_image = warp(moving_image, displacement_field, num_steps=self.warp_num_itegration_steps)
        return (
            warped_image.reshape(batch_size, seq_len, height, width),
            displacement_field.reshape(batch_size, seq_len, DISCPLACEMENT_FIELD_2D_DIMENSIONS, height, width),
        )
