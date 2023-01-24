# coding=utf-8
# Copyright (c) DIRECT Contributors

"""General mathematical optimization techniques."""

from typing import Callable, Optional

import torch
from torch import nn


class Algorithm(nn.Module):
    """Base class for implementing mathematical optimization algorithms."""

    def __init__(self, max_iter: int = 30):
        self.max_iter = max_iter
        self.iter = 0
        super().__init__()

    def _update(self):
        """Abstract method for updating the algorithm's parameters."""
        raise NotImplementedError

    def _done(self):
        """Abstract method for checking if the algorithm has ran for `max_iter`.

        Returns
        -------
        bool
        """
        return self.iter >= self.max_iter

    def update(self):
        """Update the algorithm's parameters and increment the iteration count."""
        self._update()
        self.iter += 1

    def done(self):
        """Check if the algorithm has converged.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return self._done()

    def forward(self):
        """Run the algorithm until convergence."""
        while not self.done():
            self.update()


class MaximumEigenvaluePowerMethod(Algorithm):
    """A class for solving the maximum eigenvalue problem using the Power Method."""

    def __init__(
        self,
        forward_operator: Callable,
        x: torch.Tensor,
        norm_func: Optional[Callable] = None,
        max_iter: int = 30,
    ):
        """Inits :class:`MaximumEigenvaluePowerMethod`.

        Parameters
        ----------
        forward_operator : Callable
            The forward operator for the problem.
        x : torch.Tensor
            The initial guess for the eigenvector.
        norm_func : Callable, optional
            An optional function for normalizing the eigenvector. Default: None.
        max_iter : int, optional
            Maximum number of iterations to run the algorithm. Default: 30.
        """
        self.forward_operator = forward_operator
        self.x = x
        self.norm_func = norm_func
        super().__init__(max_iter)

    def _update(self):
        """Perform a single update step of the algorithm.

        Updates maximum eigenvalue guess and corresponding eigenvector.
        """
        y = self.forward_operator(self.x)
        if self.norm_func is None:
            self.max_eig = (y * self.x.conj()).sum() / (self.x * self.x.conj()).sum()
        else:
            self.max_eig = self.norm_func(y)
        self.x = y / self.max_eig

    def _done(self):
        """Check if the algorithm is done.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return self.iter >= self.max_iter


class GradientDescentMethod(Algorithm):
    """A class for solving optimization problems using the Gradient descent method."""

    def __init__(
        self,
        gradient_operator: Callable,
        x: torch.Tensor,
        alpha: float,
        max_iter: int = 100,
        tol: float = 0.001,
    ):
        """Inits :class:`GradientDescentMethod`.

        Parameters
        ----------
        gradient_operator : Callable
            The gradient operator for the problem.
        x : torch.Tensor
            The initial guess for the solution.
        alpha : float
            The step size of the algorithm.
        max_iter : int, optional
            Maximum number of iterations to run the algorithm. Default: 100.
        tol : float, optional
            The tolerance for the residual error. Default: 0.001.
        """
        self.gradient_operator = gradient_operator
        self.alpha = alpha
        self.x = x
        self.tol = tol
        self.resid = torch.inf
        super().__init__(max_iter)

    def _update(self):
        r"""Performs a single update of the algorithm.

        It takes a step into the negative direction of the gradient with step size :math:`\alpha / (t+1)`.
        """
        x_old = self.x.clone()
        self.x = self.x - self.alpha / (self.iter + 1) ** 2 * self.gradient_operator(self.x)
        self.resid = torch.norm(self.x - x_old) / self.alpha

    def _done(self):
        """Check if the algorithm has converged.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return (self.iter >= self.max_iter) or self.resid <= self.tol
