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
"""Probability distributions for sampling."""

from __future__ import annotations

import numpy as np


def triangular_distribution(
    a: float,
    b: float,
    n: int,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample from a triangular distribution on ``[a, b]`` with mode at ``b``.

    The probability density is proportional to ``x`` on ``[a, b]``, so values near
    the upper endpoint are sampled more often than values near the lower endpoint.

    Parameters
    ----------
    a : float
        Left endpoint of the distribution.
    b : float
        Right endpoint of the distribution.
    n : int
        Number of samples to draw.
    rng : np.random.RandomState, optional
        Random number generator. Default: None.

    Returns
    -------
    np.ndarray
        Array of ``n`` samples.
    """

    def inverse_cdf(u: np.ndarray) -> np.ndarray:
        return np.sqrt(u * (b**2 - a**2) + a**2)

    if rng is None:
        rng = np.random.RandomState()

    uniform_samples = rng.uniform(0, 1, n)
    return inverse_cdf(uniform_samples)
