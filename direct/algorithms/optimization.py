# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import torch
from torch import nn


class Algorithm(nn.Module):
    def __init__(self, max_iter: int = 30):
        self.max_iter = max_iter
        self.iter = 0
        super().__init__()

    def _update(self):
        raise NotImplementedError

    def _done(self):
        return self.iter >= self.max_iter

    def update(self):
        self._update()
        self.iter += 1

    def done(self):
        return self._done()

    def forward(self):
        while not self.done():
            self.update()


class MaximumEigenvaluePowerMethod(Algorithm):
    def __init__(
        self,
        forward_operator: Callable,
        x: torch.Tensor,
        norm_func: Optional[Callable] = None,
        max_iter: int = 30,
    ):
        self.forward_operator = forward_operator
        self.x = x
        self.norm_func = norm_func
        super().__init__(max_iter)

    def _update(self):
        y = self.forward_operator(self.x)
        if self.norm_func is None:
            self.max_eig = (y * self.x.conj()).sum() / (self.x * self.x.conj()).sum()
        else:
            self.max_eig = self.norm_func(y)
        self.x = y / self.max_eig

    def _done(self):
        return self.iter >= self.max_iter


class GradientMethod(Algorithm):
    def __init__(
        self,
        gradient_operator: Callable,
        x: torch.Tensor,
        alpha: float,
        max_iter: int = 100,
        tol: float = 0.001,
    ):
        self.gradient_operator = gradient_operator
        self.alpha = alpha
        self.x = x
        self.tol = tol
        self.resid = torch.inf
        super().__init__(max_iter)

    def _update(self):
        x_old = self.x.clone()
        self.x = self.x - self.alpha / (self.iter + 1) ** 2 * self.gradient_operator(self.x)
        self.resid = torch.norm(self.x - x_old) / self.alpha

    def _done(self):
        return (self.iter >= self.max_iter) or self.resid <= self.tol


class ConjugateGradient(Algorithm):
    r"""Conjugate gradient method.
    Solves for:
    .. math:: A x = b
    where A is a Hermitian linear operator.
    Args:
        A (Linop or function): Linop or function to compute A.
        b (array): Observation.
        x (array): Variable.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping condition.
    """

    def __init__(
        self,
        A: Callable,
        b: torch.Tensor,
        x: torch.Tensor,
        max_iter: int = 100,
        tol: float = 0.0001,
    ):
        self.A = A
        self.b = b
        self.x = x
        self.tol = tol

        self.r = b - self.A(self.x)

        z = self.r

        if max_iter > 1:
            self.p = z.clone()
        else:
            self.p = z

        self.not_positive_definite = False
        self.rzold = torch.vdot(self.r, z).real
        self.resid = self.rzold.item() ** 0.5

        super().__init__(max_iter)

    def _update(self):
        Ap = self.A(self.p)
        pAp = torch.vdot(self.p, Ap).real
        if pAp <= 0:
            self.not_positive_definite = True
            return

        self.alpha = self.rzold / pAp
        self.x = self.alpha * self.p + self.x

        if self.iter < self.max_iter - 1:
            self.r = self.r - self.alpha * Ap
            z = self.r

            rznew = torch.vdot(self.r, z).real
            beta = rznew / self.rzold
            self.p = z + beta * self.p
            self.rzold = rznew

        self.resid = self.rzold**0.5

    def _done(self):
        return self.iter >= self.max_iter or self.not_positive_definite or self.resid <= self.tol
