# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import List, Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal as normal


def simulate_sens_maps(shape: Union[List[int], Tuple[int]], num_coils: int, var: float = 1) -> np.ndarray:
    r"""Simulates coil sensitivities using bi-variate or tri-variate gaussian distribution.

    Parameters
    ----------
    shape: List[int] or Tuple[int]
        (nx, ny) or (nx, ny, nz).
    num_coils: int
        Number of coils to be simulated.
    var: float
        Variance.

    Returns
    -------
    sensitivity_map : nd.array
        Simulated coil sensitivity maps of shape (num_coils, *shape).

    Note
    ----
    Sensitivity maps are normalized such that:

    .. math::
        \sum_{k=1}^{n_c} {S^{k}}^{*}S^{k} = I.
    """
    if num_coils == 1:
        return np.ones(shape)[None] + 0.0j
    # X, Y are switched in np.meshgrid
    meshgrid = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape[:2][::-1] + shape[2:]])
    indices = np.stack(meshgrid, axis=-1)
    print(indices.shape)

    sensitivity_map = np.zeros((num_coils, *shape))
    print(sensitivity_map.shape)
    # Assume iid
    cov = np.zeros(len(shape))
    for ii in range(len(shape)):
        cov[ii] = var
    cov = np.diag(cov)
    print(cov)
    for coil_idx in range(num_coils):
        mu = [np.cos(coil_idx / num_coils * 2 * np.pi), np.sin(coil_idx / num_coils * 2 * np.pi)]
        if len(shape) == 3:
            mu += [0.0]
        print(normal(mu, cov).pdf(indices).shape)
        sensitivity_map[coil_idx] = normal(mu, cov).pdf(indices)

    sensitivity_map = sensitivity_map + 1.0j * sensitivity_map  # make complex
    # Normalize
    sensitivity_map_norm = np.sqrt((np.conj(sensitivity_map) * sensitivity_map).sum(0))[None]
    sensitivity_map = sensitivity_map / sensitivity_map_norm

    return sensitivity_map
