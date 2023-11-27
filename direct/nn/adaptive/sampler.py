# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.sampler module."""

import functools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.nn.conv.conv import CWNConv2d, CWNConv3d
from direct.nn.types import ActivationType, DirectEnum


class SingleConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        padding: int = 1,
        drop_prob: float = 0,
        pool_size: int = 2,
        cwn_conv: bool = False,
    ):
        """Inits :class:`SingleConvBlock`.

        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        kernel_size : int
            Kernel size. Default: 3.
        padding : int
            Padding. Default: 1.
        drop_prob : float
            Dropout probability. Default: 0.
        pool_size : int
            Size of 2D max-pooling operator. Default: 2.
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_prob = drop_prob
        self.pool_size = pool_size
        self.cwn_conv = cwn_conv

        layers = [
            (CWNConv2d if cwn_conv else nn.Conv2d)(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        ]

        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SingleConvBlock`.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape [batch_size, self.in_chans, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, self.out_chans, height, width].
        """
        return self.layers(inp)

    def __repr__(self):
        return (
            f"SingleConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"kernel_size={self.kernel_size}, padding={self.padding}, "
            f"drop_prob={self.drop_prob}, max_pool_size={self.pool_size}, "
            f"cwn_conv={self.cwn_conv})"
        )


class SingleConv3dBlock(nn.Module):
    """
    A 3D Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        padding: int = 1,
        drop_prob: float = 0,
        pool_size: int = 2,
        cwn_conv: bool = False,
    ):
        """Inits :class:`SingleConv3dBlock`.

        Parameters
        ----------
        in_chans : int
            Number of channels in the input.
        out_chans : int
            Number of channels in the output.
        kernel_size : int
            Kernel size. Default: 3.
        padding : int
            Padding. Default: 1.
        drop_prob : float
            Dropout probability. Default: 0.
        pool_size : int
            Size of 2D max-pooling operator. Default: 2.
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_prob = drop_prob
        self.pool_size = pool_size
        self.cwn_conv = cwn_conv

        layers = [
            (CWNConv3d if cwn_conv else nn.Conv3d)(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.InstanceNorm3d(out_chans),
            nn.ReLU(),
            nn.Dropout3d(drop_prob),
        ]

        if pool_size > 1:
            layers.append(nn.MaxPool3d(pool_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SingleConvBlock`.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape [batch_size, self.in_chans, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, self.out_chans, height, width].
        """
        return self.layers(inp)

    def __repr__(self):
        return (
            f"SingleConv3dBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"kernel_size={self.kernel_size}, padding={self.padding}, "
            f"drop_prob={self.drop_prob}, max_pool_size={self.pool_size}, "
            f"cwn_conv={self.cwn_conv})"
        )


class LineConvSamplerDim(DirectEnum):
    TWO_D = 2
    THREE_D = 3


class LineConvSampler(nn.Module):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        kernel_size: int = 3,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        activation: ActivationType = ActivationType.LEAKYRELU,
        cwn_conv: bool = False,
        dim: LineConvSamplerDim = LineConvSamplerDim.TWO_D,
    ):
        """Inits :class:`ImageLineConvSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, H, W).
        kernel_size : int
            Convolution kernel size. Padding is computed as kernel_size // 2. Default: 3.
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        activation : ActivationType
            Activation function to use: ActivationType.LEAKYRELU or ActivationType.ELU.
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        dim : LineConvSamplerDim
            Can be either `LineConvSamplerDim.TWO_D` for 2D sampling,
            or `LineConvSamplerDim.THREE_D` for 3D sampling. Default: `LineConvSamplerDim.TWO_D`.
        """
        super().__init__()

        if dim == LineConvSamplerDim.TWO_D:
            if len(input_dim) != 3:
                raise ValueError(
                    f"`input_dim` should have length equal to 3 for 2D sampling."
                    f" Received: `input_dim`={input_dim}."
                )
            conv_block = SingleConvBlock
        else:
            if len(input_dim) != 4:
                raise ValueError(
                    f"`input_dim` should have length equal to 4 for 3D sampling."
                    f" Received: `input_dim`={input_dim}."
                )
            conv_block = SingleConv3dBlock

        self.input_dim = input_dim
        self.in_chans = input_dim[0]
        self.num_actions = input_dim[-1]
        self.chans = chans

        self.num_pool_layers = num_pool_layers
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.pool_size = 2
        self.slope = slope
        self.use_softplus = use_softplus
        self.num_fc_layers = num_fc_layers
        self.cwn_conv = cwn_conv
        self.activation = activation

        # Initial from in_chans to chans
        self.channel_layer = conv_block(
            self.in_chans,
            chans,
            kernel_size,
            kernel_size // 2,
            drop_prob,
            pool_size=1,
            cwn_conv=cwn_conv,
        )

        # Downsampling convolution
        # These are num_pool_layers layers where each layers 2x2 max pools, and doubles
        # the number of channels.
        self.down_sample_layers = nn.ModuleList(
            [
                conv_block(
                    chans * 2**i,
                    chans * 2 ** (i + 1),
                    kernel_size,
                    kernel_size // 2,
                    drop_prob,
                    pool_size=self.pool_size,
                    cwn_conv=cwn_conv,
                )
                for i in range(num_pool_layers)
            ]
        )

        self.feature_extractor = nn.Sequential(self.channel_layer, *self.down_sample_layers)

        # Dynamically determinte size of fc_layer
        self.flattened_size = functools.reduce(
            operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape)
        )

        fc_out: list[nn.Module] = []
        for layer in range(self.num_fc_layers):
            in_features = fc_size
            out_features = fc_size
            if layer == 0:
                in_features = self.flattened_size
            if layer + 1 == self.num_fc_layers:
                out_features = self.num_actions
            fc_out.append(nn.Linear(in_features=in_features, out_features=out_features))

            if layer + 1 < self.num_fc_layers:
                act: nn.Module
                if activation == ActivationType.LEAKYRELU:
                    act = nn.LeakyReLU()
                elif activation == ActivationType.ELU:
                    act = nn.ELU()
                else:
                    raise RuntimeError(
                        f"Invalid activation function {activation}. "
                        f"Should be ActivationType.LEAKY_RELU or ActivationType.ELU."
                    )
                fc_out.append(act)

        self.fc_out = nn.Sequential(*fc_out)


class ImageLineConvSampler(LineConvSampler):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        cwn_conv: bool = False,
        activation: ActivationType = ActivationType.LEAKYRELU,
        dim: LineConvSamplerDim = LineConvSamplerDim.TWO_D,
    ):
        """Inits :class:`ImageLineConvSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, H, W) or (C, S, H, W).
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        activation : str
            Activation function to use: leakyrelu or elu.
        dim : LineConvSamplerDim
            Can be either `LineConvSamplerDim.TWO_D` for 2D sampling,
            or `LineConvSamplerDim.THREE_D` for 3D sampling. Default: `LineConvSamplerDim.TWO_D`.
        """
        super().__init__(
            input_dim=input_dim,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            drop_prob=drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            num_fc_layers=num_fc_layers,
            cwn_conv=cwn_conv,
            activation=activation,
            dim=dim,
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Input tensor of shape [batch_size, self.in_chans, [slice,] height, width].
        mask : torch.Tensor
            Input tensor of shape [resolution], containing 0s and 1s.

        Returns
        -------
        torch.Tensor
            prob_mask [batch_size, num_actions=width] corresponding to all actions at the
            given observation. Gives probabilities of sampling a particular action.
        """

        # Image embedding

        image_emb = self.feature_extractor(image)

        # flatten all but batch dimension before FC layers
        image_emb = image_emb.flatten(start_dim=1)

        out = self.fc_out(image_emb)

        if self.use_softplus:
            # Softplus to make positive
            out = F.softplus(out, beta=self.slope)
            # Make sure max probability is 1, but ignore already sampled rows for this normalisation, since
            #  those get masked out later anyway.
            prob_mask = out / torch.max((1 - mask.reshape(out.shape[0], out.shape[1])) * out, dim=1)[0].reshape(-1, 1)
        else:
            prob_mask = torch.sigmoid(self.slope * out)
        # Mask out already sampled rows
        prob_mask = prob_mask * (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1]))
        assert len(prob_mask.shape) == 2
        return prob_mask


class ImageLineConv2dSampler(ImageLineConvSampler):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        cwn_conv: bool = False,
        activation: ActivationType = ActivationType.LEAKYRELU,
    ):
        """Inits :class:`ImageLineConv2dSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, H, W).
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        activation : str
            Activation function to use: leakyrelu or elu.
        """
        super().__init__(
            input_dim=input_dim,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            drop_prob=drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            num_fc_layers=num_fc_layers,
            cwn_conv=cwn_conv,
            activation=activation,
            dim=LineConvSamplerDim.TWO_D,
        )


class ImageLineConv3dSampler(ImageLineConvSampler):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        cwn_conv: bool = False,
        activation: ActivationType = ActivationType.LEAKYRELU,
    ):
        """Inits :class:`ImageLineConv3dSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, S, H, W).
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        activation : str
            Activation function to use: leakyrelu or elu.
        """
        super().__init__(
            input_dim=input_dim,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            drop_prob=drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            num_fc_layers=num_fc_layers,
            cwn_conv=cwn_conv,
            activation=activation,
            dim=LineConvSamplerDim.THREE_D,
        )


class KSpaceLineConvSampler(LineConvSampler):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        cwn_conv: bool = False,
        activation: ActivationType = ActivationType.LEAKYRELU,
        dim: LineConvSamplerDim = LineConvSamplerDim.TWO_D,
    ):
        """Inits :class:`KSpaceLineConvSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, H, W) or (C, S, H, W).
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        activation : str
            Activation function to use: leakyrelu or elu.
        dim : LineConvSamplerDim
            Can be either `LineConvSamplerDim.TWO_D` for 2D sampling,
            or `LineConvSamplerDim.THREE_D` for 3D sampling. Default: `LineConvSamplerDim.TWO_D`.
        """
        super().__init__(
            input_dim=input_dim,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            kernel_size=7,
            drop_prob=drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            num_fc_layers=num_fc_layers,
            cwn_conv=cwn_conv,
            activation=activation,
            dim=dim,
        )
        self.coil_dim = 1

    def forward(self, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        kspace : torch.Tensor
            Input tensor of shape [batch_size, coils, self.in_chans, [slice,] height, width].
        mask : torch.Tensor
            Input tensor of shape [resolution], containing 0s and 1s.

        Returns
        -------
        torch.Tensor
            prob_mask [batch_size, num_actions=width] corresponding to all actions at the
            given observation. Gives probabilities of sampling a particular action.
        """

        # Kspace embeddings, aggregated via summation

        n_coils = kspace.shape[self.coil_dim]

        embds = []
        for i in range(n_coils):
            coil_emb = self.feature_extractor(kspace[:, i])
            # flatten all but batch dimension before FC layers
            coil_emb = coil_emb.flatten(start_dim=1)
            embds.append(coil_emb)

        out = self.fc_out(sum(embds))

        if self.use_softplus:
            # Softplus to make positive
            out = F.softplus(out, beta=self.slope)
            # Make sure max probability is 1, but ignore already sampled rows for this normalisation, since
            #  those get masked out later anyway.
            prob_mask = out / torch.max((1 - mask.reshape(out.shape[0], out.shape[1])) * out, dim=1)[0].reshape(-1, 1)
        else:
            prob_mask = torch.sigmoid(self.slope * out)
        # Mask out already sampled rows
        prob_mask = prob_mask * (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1]))
        assert len(prob_mask.shape) == 2
        return prob_mask


class KSpaceLineConv2dSampler(KSpaceLineConvSampler):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        cwn_conv: bool = False,
        activation: ActivationType = ActivationType.LEAKYRELU,
    ):
        """Inits :class:`KSpaceLineConv2dSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, H, W).
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        activation : str
            Activation function to use: leakyrelu or elu.
        """
        super().__init__(
            input_dim=input_dim,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            drop_prob=drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            num_fc_layers=num_fc_layers,
            cwn_conv=cwn_conv,
            activation=activation,
            dim=LineConvSamplerDim.TWO_D,
        )


class KSpaceLineConv3dSampler(KSpaceLineConvSampler):
    def __init__(
        self,
        input_dim: tuple,
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        cwn_conv: bool = False,
        activation: ActivationType = ActivationType.LEAKYRELU,
    ):
        """Inits :class:`KSpaceLineConv3dSampler`.

        Parameters
        ----------
        chans : int
            Number of input channels.
        input_dim : tuple
            Input size of reconstructed images (C, S, H, W).
        chans : int
            Number of output channels of the first convolution layer.
        num_pool_layers : int
            Number of down-sampling layers.
        fc_size : int
            Number of hidden neurons for the fully connected layers.
        drop_prob : float
            Dropout probability.
        num_fc_layers : int
            Number of fully connected layers to use after convolutional part.
        use_softplus : bool
            Whether to use softplus as final activation (otherwise sigmoid).
        cwn_conv : bool
            If True will use Convolutional layer with Centered Weight Normalization. Default: False.
        activation : str
            Activation function to use: leakyrelu or elu.
        """
        super().__init__(
            input_dim=input_dim,
            chans=chans,
            num_pool_layers=num_pool_layers,
            fc_size=fc_size,
            drop_prob=drop_prob,
            slope=slope,
            use_softplus=use_softplus,
            num_fc_layers=num_fc_layers,
            cwn_conv=cwn_conv,
            activation=activation,
            dim=LineConvSamplerDim.THREE_D,
        )
