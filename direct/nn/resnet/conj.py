# coding=utf-8
# Copyright (c) DIRECT Contributors

from enum import Enum

import torch

from direct.data.transforms import expand_operator, reduce_operator


class CGUpdateType(str, Enum):
    FR = "FR"
    PRP = "PRP"
    DY = "DY"
    BAN = "BAN"


class ConjGrad:
    def __init__(
        self,
        forward_operator,
        backward_operator,
        cg_param_update_type: CGUpdateType = CGUpdateType.FR,
        num_iters: int = 6,
    ):
        r"""
        Performs the Conjugate Gradient algorithm to approach a solution to:

        .. math ::

            \min_{x} f(x) = \min_{x} \fraq{1}{2} \big( ||A(x) - y||_2^2 + \mu ||x - z||_2^2 \big).

        The algorithm is as follows:

        .. math ::

            x^{k+1} = x^{k} + \alpha_{k} d^{k}

        where

        .. math ::

            g^{k} = - \nabla f(x_{k}), and d^{k} = g^{k} + \beta^{k} * d^{k-1} and d^{0} = g^{0}

        and :math:`\beta` is calculated using on of ["FR", "PRB", "DY", "BAN"] methods as shown in [1]_.
        For example, for "FR" we have:

        .. math ::

            \beta_{k} = \fraq{|| g^{k+1} ||_2^2}{|| g^{k} ||_2^2.

        We calculate the step size :math:`\alpha^{k}` using the secant method.

        References
        ----------

        .. [1] Ishaq, Adam Ajimoti, et al. “A STEP-LENGTH FORMULA FOR CONJUGATE GRADIENT METHODS.” MALAYSIAN JOURNAL OF
            COMPUTING, vol. 5, no. 1, Apr. 2020, p. 403. DOI: https://doi.org/10.24191/mjoc.v5i1.7715.
        """
        super().__init__()

        assert cg_param_update_type in ["FR", "PRB", "DY", "BAN"]
        self.cg_param_update_type = cg_param_update_type
        self.num_iters = num_iters

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._spatial_dims = (2, 3)
        self._coil_dim = 1

    def secant_method(self, x, y, z, sensitivity_map, sampling_mask, dk, mu, h=None):
        r"""Produces a step size using the secant method:

        .. math ::

            a^{k} = \frac{{\nabla f(x^{k})}^{*} d^{k}}{H(x{k})^{*} d^{k}}.

        Estimates Hessian as:

        .. math ::

            H(x^{k}) \ approx \frac{ \nabla f(x^{k} + h * d^{k}) - \nabla f(x^{k}) }{ h }

        for some small number :math:`h`.
        """
        if h is None:
            h = torch.ones(x.shape[0], device=x.device) * 1e-4

        g_k = self.grad(x, y, z, sensitivity_map, sampling_mask, mu)
        g_k_h = self.grad(
            x + h.reshape([-1] + torch.ones(x.ndim - 1, dtype=int).tolist()) * dk,
            y,
            z,
            sensitivity_map,
            sampling_mask,
            mu,
        )

        dim = torch.arange(1, x.ndim - 1).tolist()
        top_term = abs_dot_product(g_k, dk, dim)
        bottom_term = abs_dot_product(g_k_h - g_k, dk, dim)
        return h * (top_term / bottom_term)

    def _forward_operator(self, image, sensitivity_map, sampling_mask):
        return torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            self.forward_operator(expand_operator(image, sensitivity_map, dim=self._coil_dim), dim=self._spatial_dims),
        )

    def _backward_operator(self, kspace, sensitivity_map, sampling_mask):
        return reduce_operator(
            self.backward_operator(
                torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device), kspace),
                dim=self._spatial_dims,
            ),
            sensitivity_map,
            dim=self._coil_dim,
        )

    def fun(self, x, y, z, sensitivity_map, sampling_mask, mu):
        r"""
        Computes

        .. math::

            f(x) = \fraq{1}{2} \big( ||A(x) - y||_2^2 + \mu ||x - z||_2^2 \big)

        where :math:`x` and :math:`z` denote images of shape (N, height, width, 2), :math:`y` a kspace of shape (N, coils, height, width, 2),
        and `senstivity_map` and `sampling_mask` of shapes (N, coils, height, width, 2) and (N, 1, height, width, 1).
        """
        A_x = self._forward_operator(x, sensitivity_map, sampling_mask)

        term1 = torch.norm(torch.view_as_complex(A_x - y), p=2, dim=torch.arange(2, y.ndim - 1).tolist()).sum(
            self._coil_dim
        )
        term2 = mu * torch.norm(torch.view_as_complex(x - z), p=2, dim=torch.arange(1, x.ndim - 1).tolist())

        return 0.5 * (term1**2 + term2**2)

    def grad(self, x, y, z, sensitivity_map, sampling_mask, mu):
        r"""
        Computes

        .. math::

            \nabla_x f(x) = A^* \big( A(x) - y \big) + \mu (x - z)
        """
        x = x.clone()  # without clone it caused backpropagation issues
        term1 = self._backward_operator(
            self._forward_operator(x, sensitivity_map, sampling_mask) - y, sensitivity_map, sampling_mask
        )
        term2 = mu * (x - z)

        return term1 + term2

    def __call__(self, x, y, z, sensitivity_map, sampling_mask, mu):

        # g^0 = - \nabla f(x^0)
        g_k_new = -self.grad(x, y, z, sensitivity_map, sampling_mask, mu)
        # d^0 = g^0
        d_k = g_k_new.clone()
        # f(x^0)
        # initial step_size is a random small number
        a_k = None
        for i in range(self.num_iters):
            # TODO: Not the best way to calculate a_k.
            a_k = self.secant_method(x, y, z, sensitivity_map, sampling_mask, d_k, mu, a_k)

            x += a_k.reshape([-1] + torch.ones(x.ndim - 1, dtype=int).tolist()) * d_k

            g_k_old = g_k_new.clone()
            g_k_new = -self.grad(x, y, z, sensitivity_map, sampling_mask, mu)

            dim = torch.arange(1, x.ndim - 1).tolist()
            if self.cg_param_update_type == "FR":
                b_k = _FR(g_k_new, g_k_old, dim)
            elif self.cg_param_update_type == "PRB":
                b_k = _PRP(g_k_new, g_k_old, dim)
            elif self.cg_param_update_type == "DY":
                b_k = _DY(g_k_new, g_k_old, d_k, dim)
            else:
                b_k = _BAN(g_k_new, g_k_old, dim)

            b_k = b_k.reshape([-1] + torch.ones(x.ndim - 1, dtype=int).tolist())
            d_k = g_k_new + b_k * d_k

        return x


def abs_dot_product(A, B, dim):
    return (torch.view_as_complex(A).conj() * torch.view_as_complex(B)).sum(dim).abs()


def _FR(g_k_new, g_k_old, dim):
    return (
        torch.norm(torch.view_as_complex(g_k_new), dim=dim) ** 2
        / torch.norm(torch.view_as_complex(g_k_old), dim=dim) ** 2
    )


def _PRP(g_k_new, g_k_old, dim):
    y_k = g_k_new - g_k_old
    return abs_dot_product(g_k_new, y_k, dim) / torch.norm(torch.view_as_complex(g_k_old), dim=dim) ** 2


def _DY(g_k_new, g_k_old, d_k, dim):
    y_k = g_k_new - g_k_old
    return torch.norm(torch.view_as_complex(g_k_new), dim=dim) ** 2 / abs_dot_product(g_k_old, d_k, dim)


def _BAN(g_k_new, g_k_old, dim):
    y_k = g_k_new - g_k_old
    return abs_dot_product(g_k_new, y_k, dim) / abs_dot_product(g_k_old, y_k, dim)
