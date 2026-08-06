# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
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

"""Straight-through binarizers for adaptive k-space sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ["deterministic_binarizer", "ThresholdSigmoidMask"]


class ThresholdSigmoidMaskFunction(Function):
    """Straight-through estimator for stochastic mask binarization.

    The forward step stochastically binarizes the probability mask via rejection
    sampling. The backward step approximates the non-differentiable threshold
    operator with a sigmoid of large slope.
    """

    @staticmethod
    def forward(ctx, inputs, slope, clamp):
        batch_size = len(inputs)
        probs = []
        results = []

        for i in range(batch_size):
            x = inputs[i : i + 1]
            count = 0
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()
                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break
                count += 1
                if count > 1000:
                    raise RuntimeError(
                        "Rejection sampled exceeded number of tries. Probably this means all "
                        "sampling probabilities are 1 or 0 for some reason, leading to divide "
                        "by zero in rescale_probs()."
                    )
            probs.append(prob)
            results.append(result)
        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)

        slope = torch.tensor(slope, requires_grad=False)
        ctx.clamp = clamp
        ctx.save_for_backward(inputs, probs, slope)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        input, prob, slope = ctx.saved_tensors
        if ctx.clamp:
            grad_output = F.hardtanh(grad_output)
        # derivative of sigmoid function
        current_grad = (
            slope * torch.exp(-slope * (input - prob)) / torch.pow((torch.exp(-slope * (input - prob)) + 1), 2)
        )
        return current_grad * grad_output, None, None


class ThresholdSigmoidMask(nn.Module):
    """Module wrapper around :class:`ThresholdSigmoidMaskFunction`."""

    def __init__(self, slope: float, clamp: bool) -> None:
        """Inits :class:`ThresholdSigmoidMask`.

        Parameters
        ----------
        slope : float
            Slope of the sigmoid used in the straight-through backward pass.
        clamp : bool
            If ``True``, clamp gradients with :func:`torch.nn.functional.hardtanh`.
        """
        super().__init__()
        self.slope = slope
        self.clamp = clamp

        self.fun = ThresholdSigmoidMaskFunction.apply

    def forward(self, input_probs: torch.Tensor) -> torch.Tensor:
        """Stochastically binarize probability maps.

        Parameters
        ----------
        input_probs : torch.Tensor
            Probability map of shape ``(batch, num_actions)``.

        Returns
        -------
        torch.Tensor
            Binary mask with the same shape as ``input_probs``.
        """
        return self.fun(input_probs, self.slope, self.clamp)


def deterministic_binarizer(input_probs: torch.Tensor, budget: int) -> torch.Tensor:
    """Binarizes a tensor based on the highest probabilities within the budget.

    Parameters
    ----------
    input_probs : torch.Tensor:
        Input tensor of probabilities with shape (batch, max_lines).
    budget : int
        The number of lines to keep active (binarized).

    Returns
    -------
    binarized_tensor : torch.Tensor
        Binarized tensor with shape (batch, max_lines).
    """
    # Get the top-k values and indices along the last dimension
    _, top_indices = torch.topk(input_probs, k=budget, dim=-1)

    # Create a binary mask with the same shape as the input tensor
    binarized_tensor = torch.zeros_like(input_probs)

    # Set the indices corresponding to the top-k values to 1 in the binary mask
    binarized_tensor.scatter_(dim=-1, index=top_indices, value=1)

    return binarized_tensor
