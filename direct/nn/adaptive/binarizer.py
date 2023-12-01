# Copyright (c) DIRECT Contributors

"""direct.nn.adaptive.binarizer module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ThresholdSigmoidMaskFunction(Function):

    """
    Straight through estimator.
    The forward step stochastically binarizes the probability mask.
    The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
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
                    print(torch.mean(prob), torch.mean(result), torch.mean(x))
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
    def __init__(self, slope, clamp):
        super().__init__()
        self.slope = slope
        self.clamp = clamp

        self.fun = ThresholdSigmoidMaskFunction.apply

    def forward(self, input_probs: torch.Tensor):
        return self.fun(input_probs, self.slope, self.clamp)
