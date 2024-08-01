# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.parameterized module."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.nn.adaptive.binarizer import ThresholdSigmoidMask
from direct.nn.adaptive.types import PolicySamplingDimension, PolicySamplingType
from direct.nn.adaptive.utils import (
    normalize_masked_probabilities,
    reshape_acquisitions_post_sampling,
    reshape_mask_pre_sampling,
)

__all__ = [
    "Parameterized2dPolicy",
    "Parameterized3dPolicy",
    "ParameterizedDynamic2dPolicy",
    "ParameterizedMultislice2dPolicy",
]


class ParameterizedPolicy(nn.Module):
    """Base class for Parameterized policy models."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        sampling_type: PolicySamplingType = PolicySamplingType.STATIC,
        num_time_steps: Optional[int] = None,
        num_slices: Optional[int] = None,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        acceleration: Optional[float] = None,
    ):
        """Inits :class:`ParameterizedPolicy`.

        Parameters
        ----------
        kspace_shape : tuple[int, ...]
            The shape of the k-space data used in the policy.
        sampling_dimension : PolicySamplingDimension
            The sampling dimension for the policy, either `ONE_D` or `TWO_D`.
        sampling_type : PolicySamplingType, optional
            The sampling type for the policy, default is `STATIC`.
        num_time_steps : int, optional
            The number of time steps (required if `sampling_type` is `DYNAMIC_2D_2D`).
        num_slices : int, optional
            The number of slices (required if `sampling_type` is `MULTISLICE_2D`).
        use_softplus : bool, optional
            Flag indicating whether softplus function should be used, default is `True`.
        slope : float, optional
            The slope parameter used in the policy, default is `10`.
        fix_sign_leakage : bool, optional
            Flag indicating whether sign leakage should be fixed, default is `True`.
        st_slope : float, optional
            The slope parameter used in threshold sigmoid mask, default is `10`.
        st_clamp : bool, optional
            Flag indicating whether clamping should be applied in threshold sigmoid mask, default is `False`.

        Raises
        ------
        ValueError
            If the input dimension of the policy is not 1, 2, or 3.
            If `num_time_steps` is `None` but `sampling_type` is set to 'DYNAMIC_2D_2D'.
        """
        super().__init__()

        if len(kspace_shape) not in [1, 2, 3]:
            raise ValueError(
                f"Input dimension of the policy should have length of 1, 2, or 3. Received `input_dim`={kspace_shape}."
            )
        if sampling_dimension == PolicySamplingDimension.ONE_D:
            self.num_actions = kspace_shape[-1]  # num_actions = width
        elif sampling_dimension == PolicySamplingDimension.TWO_D:
            self.num_actions = np.prod(kspace_shape[-2:])  # num_actions = height * width
        else:
            raise ValueError(f"Sampling dimension can be `1D` or `2D`.")

        self.kspace_shape = kspace_shape
        self.sampling_dimension = sampling_dimension

        if sampling_type in [PolicySamplingType.DYNAMIC_2D, PolicySamplingType.DYNAMIC_2D_NON_UNIFORM]:
            if num_time_steps is None:
                raise ValueError(
                    f"Received None for `num_time_steps` but `sampling_type` is set to 'DYNAMIC_2D' or `DYNAMIC_2D_NON_UNIFORM`."
                )
            self.steps = num_time_steps
        if sampling_type in [PolicySamplingType.MULTISLICE_2D, PolicySamplingType.MULTISLICE_2D_NON_UNIFORM]:
            if num_slices is None:
                raise ValueError(
                    f"Received None for `num_slices` but `sampling_type` is set to 'MULTISLICE_2D' or `MULTISLICE_2D_NON_UNIFORM`."
                )
            self.steps = num_slices

        if sampling_type == PolicySamplingType.STATIC:
            ones = torch.ones(1, self.num_actions)
        else:
            ones = torch.ones(
                1,
                num_time_steps if "dynamic" in sampling_type else num_slices,
                self.num_actions,
            )

        if use_softplus:
            # Softplus will be applied
            self.sampler = nn.Parameter(torch.normal(ones.clone(), ones.clone() / 10))
        else:
            # Sigmoid will be applied
            self.sampler = nn.Parameter(ones.clone())

        self.use_softplus = use_softplus
        self.slope = slope
        self.st_slope = st_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp
        self.binarizer = ThresholdSigmoidMask(self.st_slope, self.st_clamp)

        self.sampling_type = sampling_type

        self.acceleration = acceleration


class ParameterizedStaticPolicy(ParameterizedPolicy):
    """Base Parameterized policy model for non dynamic 2D or 3D data."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        acceleration: Optional[float] = None,
    ):
        """Inits :class:`ParameterizedStaticPolicy`.

        Parameters
        ----------
        kspace_shape : tuple[int, ...]
            The shape of the k-space data used in the policy.
        sampling_dimension : PolicySamplingDimension
            The sampling dimension for the policy, either `ONE_D` or `TWO_D`.
        use_softplus : bool, optional
            Flag indicating whether softplus function should be used, default is `True`.
        slope : float, optional
            The slope parameter used in the policy, default is `10`.
        fix_sign_leakage : bool, optional
            Flag indicating whether sign leakage should be fixed, default is `True`.
        st_slope : float, optional
            The slope parameter used in the threshold sigmoid mask, default is `10`.
        st_clamp : bool, optional
            Flag indicating whether clamping should be applied in the threshold sigmoid mask, default is `False`.
        """
        super().__init__(
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=PolicySamplingType.STATIC,
            num_time_steps=None,
            num_slices=None,
            use_softplus=use_softplus,
            slope=slope,
            fix_sign_leakage=fix_sign_leakage,
            st_slope=st_slope,
            st_clamp=st_clamp,
            acceleration=acceleration,
        )

    @abstractmethod
    def dim_check(self, kspace: torch.Tensor) -> None:
        """Abstract method to check k-space dimensions."""
        raise NotImplementedError(f"Must be implemented by child class.")

    def forward(
        self,
        mask: torch.Tensor,
        kspace: torch.Tensor,
        acceleration: float | torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass of :class:`ParameterizedStaticPolicy`.

        Reshapes mask according to sampling dimension and target k-space shape, performs sampling, applies mask to
        k-space, and performs forward propagation.

        Parameters
        ----------
        mask : torch.Tensor
            The mask tensor.
        kspace : torch.Tensor
            The k-space data tensor.
        acceleration : float or torch.Tensor
            Desired acceleration. If not a number, this should be a tensor matching the batch of k-space.
        padding : torch.Tensor, optional
            Padding tensor. If not None, locations present in padding will not be included in the resulting mask.
            Default is `None`.

        Returns
        -------
        tuple
            Tuple containing masked k-space data, masks, and final probability mask.
        """
        self.dim_check(kspace)

        batch_size = kspace.shape[0]

        masks = [mask]

        # Reshape initial mask to [batch, num_actions]
        mask, padding = reshape_mask_pre_sampling(self.sampling_dimension, mask, padding, kspace.shape)

        if self.acceleration is not None:
            acceleration = self.acceleration
        else:
            if acceleration is None:
                raise ValueError(
                    f"Argument `acceleration` received None for a value. "
                    f"This should not be None when `StraightThroughPolicy` is initialized "
                    f"with `acceleration` with None value."
                )
            else:
                if not isinstance(acceleration, (int, float, torch.Tensor)):
                    raise ValueError(f"Invalid `acceleration` type. Received `acceleration`={acceleration}.")
                acceleration = acceleration.squeeze()
                if acceleration.ndim > 1:
                    raise ValueError(
                        f"Tensor accelerations should be 1-dimensional. "
                        f"Received `acceleration` of shape ={acceleration.shape}."
                    )

        sampled_fraction = torch.tensor(
            [mask[i].sum().item() / np.prod(mask[i].shape) for i in range(mask.shape[0])]
        ).to(mask.device)
        budget = self.num_actions * (1 / acceleration - sampled_fraction)

        budget = budget.round().int()

        # Expand sampler to [batch, num_actions]
        sampler_out = self.sampler.expand(batch_size, self.num_actions)

        if self.use_softplus:
            # Softplus to make positive
            prob_mask = F.softplus(sampler_out, beta=self.slope)
            prob_mask = prob_mask / torch.max((1 - mask) * prob_mask, dim=1)[0].reshape(-1, 1)
        else:
            # Sigmoid to make positive
            prob_mask = torch.sigmoid(self.slope * sampler_out)

        # Mask out already sampled rows
        masked_prob_mask = prob_mask * (1 - mask)

        # Mask out padded areas
        if padding is not None:
            masked_prob_mask = masked_prob_mask * (1 - padding)

        masked_prob_mask = normalize_masked_probabilities(mask, masked_prob_mask, budget)

        # Binarize the mask
        flat_bin_mask = self.binarizer(masked_prob_mask)

        acquisitions, final_prob_mask, mask = reshape_acquisitions_post_sampling(
            self.sampling_dimension, flat_bin_mask, masked_prob_mask, mask, kspace.shape
        )

        mask = mask + acquisitions
        masks.append(mask)

        masked_kspace = mask * kspace

        # Note that since masked_kspace = mask * kspace, this masked_kspace will leak sign information
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask

        return masked_kspace, masks, [final_prob_mask]


class Parameterized2dPolicy(ParameterizedStaticPolicy):
    """Parameterized policy model for 2D data."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        acceleration: Optional[float] = None,
    ):
        super().__init__(
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            use_softplus=use_softplus,
            slope=slope,
            fix_sign_leakage=fix_sign_leakage,
            st_slope=st_slope,
            st_clamp=st_clamp,
            acceleration=acceleration,
        )

    def dim_check(self, kspace: torch.Tensor) -> None:
        if kspace.ndim != 5:
            raise ValueError(f"Expected shape of k-space to have 5 dimensions, but got shape={kspace.shape}.")


class Parameterized3dPolicy(ParameterizedStaticPolicy):
    """Parameterized policy model for 3D data."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        acceleration: Optional[float] = None,
    ):
        super().__init__(
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            use_softplus=use_softplus,
            slope=slope,
            fix_sign_leakage=fix_sign_leakage,
            st_slope=st_slope,
            st_clamp=st_clamp,
            acceleration=acceleration,
        )

    def dim_check(self, kspace: torch.Tensor) -> None:
        if kspace.ndim != 6:
            raise ValueError(f"Expected shape of k-space to have 6 dimensions, but got shape={kspace.shape}.")


class ParameterizedDynamicOrMultislice2dPolicy(ParameterizedPolicy):
    """Parameterized policy for dynamic or multislice 2D data model."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        sampling_type: PolicySamplingType,
        num_time_steps: Optional[int] = None,
        num_slices: Optional[int] = None,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        acceleration: Optional[float] = None,
    ):
        """Inits :class:`ParameterizedDynamicOrMultislice2dPolicy`.

        Parameters
        ----------
        kspace_shape : tuple[int, ...]
            The shape of the k-space data used in the policy.
        sampling_dimension : PolicySamplingDimension
            The sampling dimension for the policy, either `ONE_D` or `TWO_D`.
        sampling_type : PolicySamplingType
            The sampling type for the policy.
        num_time_steps : int, optional
            The number of time steps for the dynamic policy. Ignored if sampling_type is not `DYNAMIC_2D`.
        num_slices : int, optional
            The number of slices for the multislice policy. Ignored if sampling_type is not `MULTISLICE_2D`.
        use_softplus : bool, optional
            Flag indicating whether softplus function should be used, default is `True`.
        slope : float, optional
            The slope parameter used in the policy, default is `10`.
        fix_sign_leakage : bool, optional
            Flag indicating whether sign leakage should be fixed, default is `True`.
        st_slope : float, optional
            The slope parameter used in the threshold sigmoid mask, default is `10`.
        st_clamp : bool, optional
            Flag indicating whether clamping should be applied in the threshold sigmoid mask, default is `False`.
        """
        super().__init__(
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=sampling_type,
            num_time_steps=num_time_steps,
            num_slices=num_slices,
            use_softplus=use_softplus,
            slope=slope,
            fix_sign_leakage=fix_sign_leakage,
            st_slope=st_slope,
            st_clamp=st_clamp,
            acceleration=acceleration,
        )

    def forward(
        self,
        mask: torch.Tensor,
        kspace: torch.Tensor,
        acceleration: float | torch.Tensor,
        padding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass of :class:`ParameterizedDynamicOrMultislice2dPolicy`.

        Reshapes mask according to sampling dimension and target k-space shape, performs sampling per time-step or
        slice, applies mask to k-space, and performs forward propagation.

        Parameters
        ----------
        mask : torch.Tensor
            The mask tensor of shape (batch, coils, 1 or time/slices, height, width, complex).
        kspace : torch.Tensor
            The k-space data tensor of shape (batch, coils, time/slices, height, width, complex).
        acceleration : float or torch.Tensor
            Desired acceleration. If not a number, this should be a tensor matching the batch of k-space.
        padding : torch.Tensor, optional
            Padding tensor. If not None, locations present in padding will not be included in the resulting mask.
            Default is `None`.

        Returns
        -------
        tuple
            Tuple containing masked k-space data, masks, and final probability mask.
        """
        batch_size, _, slices, height, width, _ = kspace.shape  # batch, coils, time, height, width, complex
        masks = [mask.expand(batch_size, 1, slices, height, width, 1)]

        if self.acceleration is not None:
            acceleration = self.acceleration
        else:
            if acceleration is None:
                raise ValueError(
                    f"Argument `acceleration` received None for a value. "
                    f"This should not be None when `StraightThroughPolicy` is initialized "
                    f"with `acceleration` with None value."
                )
            if not isinstance(acceleration, (int, float, torch.Tensor)):
                raise ValueError(f"Invalid `acceleration` type. Received `acceleration`={acceleration}.")
            if isinstance(acceleration, torch.Tensor):
                if acceleration.ndim > 2:
                    raise ValueError(
                        f"Tensor accelerations should be 1 or 2-dimensional for "
                        f"`sampling_type`={self.sampling_type}. "
                        f"Received `acceleration`={acceleration}."
                    )
                elif acceleration.ndim == 2:
                    if acceleration.shape[1] != kspace.shape[2]:
                        raise ValueError(
                            f"Acceleration second dimension should match k-space 3rd dimension. "
                            f"Received acceleration of shape={acceleration.shape} and k-space "
                            f"of shape={kspace.shape}."
                        )

        if "non_uniform" not in self.sampling_type:
            sampled_fraction = []
            for i in range(mask.shape[0]):
                sampled_fraction.append(
                    torch.tensor(
                        [mask[i, :, j].sum().item() / np.prod(mask[i, :, j].shape) for j in range(mask.shape[2])]
                    )
                )
            sampled_fraction = torch.stack(sampled_fraction, 0)

            if isinstance(acceleration, torch.Tensor) and acceleration.ndim == 1:
                acceleration = acceleration.unsqueeze(1)
        else:
            sampled_fraction = torch.tensor(
                [mask[i].sum().item() / np.prod(mask[i].shape) for i in range(mask.shape[0])]
            )
        sampled_fraction = sampled_fraction.to(mask.device)
        budget = (
            self.num_actions
            * (1 if "non_uniform" not in self.sampling_type else self.steps)
            * (1 / acceleration - sampled_fraction)
        )

        budget = budget.round().int()

        if "non_uniform" not in self.sampling_type:
            output_mask = []
            final_prob_mask = []

            for step in range(self.steps):
                if self.sampling_dimension == PolicySamplingDimension.ONE_D:
                    mask_step = mask[:, :, 0, 0, :, :].reshape(batch_size, width)
                else:
                    mask_step = mask[:, :, 0].reshape(batch_size, height * width)

                sampler_out = self.sampler[:, step].expand(batch_size, self.num_actions)

                if self.use_softplus:
                    # Softplus to make positive
                    prob_mask = F.softplus(sampler_out, beta=self.slope)
                    prob_mask = prob_mask / torch.max((1 - mask_step) * prob_mask, dim=1)[0].reshape(-1, 1)
                else:
                    # Sigmoid to make positive
                    prob_mask = torch.sigmoid(self.slope * sampler_out)

                # Mask out already sampled rows
                masked_prob_mask = prob_mask * (1 - mask_step)

                # Mask out padded areas
                if padding is not None:
                    if self.sampling_dimension == PolicySamplingDimension.ONE_D:
                        padding = padding[:, :, 0, 0, :, :].reshape(batch_size, width)
                    else:
                        padding = padding[:, :, 0].reshape(batch_size, height * width)
                    masked_prob_mask = masked_prob_mask * (1 - padding)

                masked_prob_mask = normalize_masked_probabilities(mask_step, masked_prob_mask, budget[:, step])

                # Binarize the mask
                flat_bin_mask = self.binarizer(masked_prob_mask)

                if self.sampling_dimension == PolicySamplingDimension.ONE_D:
                    acquisitions = flat_bin_mask.reshape(batch_size, 1, 1, width, 1).expand(
                        batch_size, 1, height, width, 1
                    )
                    final_prob_mask = masked_prob_mask.reshape(batch_size, 1, 1, width, 1).expand(
                        batch_size, 1, height, width, 1
                    )
                    mask_step = mask_step.reshape(batch_size, 1, 1, width, 1).expand(batch_size, 1, height, width, 1)
                else:
                    acquisitions = flat_bin_mask.reshape(batch_size, 1, height, width, 1)
                    final_prob_mask = masked_prob_mask.reshape(batch_size, 1, height, width, 1)
                    mask_step = mask_step.reshape(batch_size, 1, height, width, 1)

                mask_step = mask_step + acquisitions
                output_mask.append(mask_step)

            output_mask = torch.stack(output_mask, 2)
        else:
            if mask.shape[2] < self.steps:
                mask = mask.expand(batch_size, 1, self.steps, height, width, 1)
            if self.sampling_dimension == PolicySamplingDimension.ONE_D:
                mask = mask[:, :, :, 0, :, :].reshape(batch_size, self.steps * width)
            else:
                mask = mask.reshape(batch_size, self.steps * height * width)

            sampler_out = self.sampler.reshape(batch_size, self.steps * self.num_actions)

            if self.use_softplus:
                # Softplus to make positive
                prob_mask = F.softplus(sampler_out, beta=self.slope)
                prob_mask = prob_mask / torch.max((1 - mask) * prob_mask, dim=1)[0].reshape(-1, 1)
            else:
                # Sigmoid to make positive
                prob_mask = torch.sigmoid(self.slope * sampler_out)

            # Mask out already sampled rows
            masked_prob_mask = prob_mask * (1 - mask)

            # Mask out padded areas
            if padding is not None:
                if self.sampling_dimension == PolicySamplingDimension.ONE_D:
                    padding = padding[:, :, :, 0, :, :].reshape(batch_size, self.steps * width)
                else:
                    padding = padding.reshape(batch_size, self.steps * height * width)
                masked_prob_mask = masked_prob_mask * (1 - padding)

            masked_prob_mask = normalize_masked_probabilities(mask, masked_prob_mask, budget)

            # Binarize the mask
            flat_bin_mask = self.binarizer(masked_prob_mask)

            if self.sampling_dimension == PolicySamplingDimension.ONE_D:
                acquisitions = flat_bin_mask.reshape(batch_size, 1, self.steps, 1, width, 1).expand(
                    batch_size, 1, self.steps, height, width, 1
                )
                final_prob_mask = masked_prob_mask.reshape(batch_size, 1, self.steps, 1, width, 1).expand(
                    batch_size, 1, self.steps, height, width, 1
                )
                mask = mask.reshape(batch_size, 1, self.steps, 1, width, 1).expand(
                    batch_size, 1, self.steps, height, width, 1
                )
            else:
                acquisitions = flat_bin_mask.reshape(batch_size, 1, self.steps, height, width, 1)
                final_prob_mask = masked_prob_mask.reshape(batch_size, 1, self.steps, height, width, 1)
                mask = mask.reshape(batch_size, 1, self.steps, height, width, 1)

            output_mask = mask + acquisitions

        masks.append(output_mask)
        masked_kspace = output_mask * kspace

        # Note that since masked_kspace = output_mask * kspace, this masked_kspace will leak sign information
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(torch.bitwise_and(kspace < 0.0, output_mask == 0.0), -1.0, 1.0)
            masked_kspace = masked_kspace * fix_sign_leakage_mask

        return masked_kspace, masks, [final_prob_mask]


class ParameterizedDynamic2dPolicy(ParameterizedDynamicOrMultislice2dPolicy):
    """Parameterized policy for dynamic 2D data model."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        num_time_steps: int,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        non_uniform: bool = False,
        acceleration: Optional[float] = None,
    ):
        """Inits :class:`ParameterizedDynamic2dPolicy`.

        Parameters
        ----------
        kspace_shape : tuple[int, ...]
            The shape of the k-space data used in the policy.
        sampling_dimension : PolicySamplingDimension
            The sampling dimension for the policy, either `ONE_D` or `TWO_D`.
        num_time_steps : int
            The number of time steps for the dynamic policy.
        use_softplus : bool, optional
            Flag indicating whether softplus function should be used, default is `True`.
        slope : float, optional
            The slope parameter used in the policy, default is `10`.
        fix_sign_leakage : bool, optional
            Flag indicating whether sign leakage should be fixed, default is `True`.
        st_slope : float, optional
            The slope parameter used in the threshold sigmoid mask, default is `10`.
        st_clamp : bool, optional
            Flag indicating whether clamping should be applied in the threshold sigmoid mask, default is `False`.
        """
        super().__init__(
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=(
                PolicySamplingType.DYNAMIC_2D if not non_uniform else PolicySamplingType.DYNAMIC_2D_NON_UNIFORM
            ),
            num_time_steps=num_time_steps,
            use_softplus=use_softplus,
            slope=slope,
            fix_sign_leakage=fix_sign_leakage,
            st_slope=st_slope,
            st_clamp=st_clamp,
            acceleration=acceleration,
        )
        self.non_uniform = non_uniform


class ParameterizedMultislice2dPolicy(ParameterizedDynamicOrMultislice2dPolicy):
    """Parameterized policy for multislice 2D data model."""

    def __init__(
        self,
        kspace_shape: tuple[int, ...],
        sampling_dimension: PolicySamplingDimension,
        num_slices: int,
        use_softplus: bool = True,
        slope: float = 10,
        fix_sign_leakage: bool = True,
        st_slope: float = 10,
        st_clamp: bool = False,
        non_uniform: bool = False,
        acceleration: Optional[float] = None,
    ):
        """Inits :class:`ParameterizedMultislice2dPolicy`.

        Parameters
        ----------
        kspace_shape : tuple[int, ...]
            The shape of the k-space data used in the policy.
        sampling_dimension : PolicySamplingDimension
            The sampling dimension for the policy, either `ONE_D` or `TWO_D`.
        num_slices : int
            The number of slices for the policy.
        use_softplus : bool, optional
            Flag indicating whether softplus function should be used, default is `True`.
        slope : float, optional
            The slope parameter used in the policy, default is `10`.
        fix_sign_leakage : bool, optional
            Flag indicating whether sign leakage should be fixed, default is `True`.
        st_slope : float, optional
            The slope parameter used in the threshold sigmoid mask, default is `10`.
        st_clamp : bool, optional
            Flag indicating whether clamping should be applied in the threshold sigmoid mask, default is `False`.
        non_uniform: bool
            Flag indicating whether masks will contain uniform accelerations or not, default is `False`.
        """
        super().__init__(
            kspace_shape=kspace_shape,
            sampling_dimension=sampling_dimension,
            sampling_type=PolicySamplingType.MULTISLICE_2D,
            num_slices=num_slices,
            use_softplus=use_softplus,
            slope=slope,
            fix_sign_leakage=fix_sign_leakage,
            st_slope=st_slope,
            st_clamp=st_clamp,
            acceleration=acceleration,
        )
        self.non_uniform = non_uniform
