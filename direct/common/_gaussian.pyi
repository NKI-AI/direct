"""Type stubs for the compiled :mod:`direct.common._gaussian` extension."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def gaussian_mask_1d(
    nonzero_count: int,
    n: int,
    center: int,
    std: float,
    mask: npt.NDArray[np.int64],
    seed: int,
) -> None: ...
def gaussian_mask_2d(
    nonzero_count: int,
    nrow: int,
    ncol: int,
    center_x: int,
    center_y: int,
    std: npt.NDArray[np.float64],
    mask: npt.NDArray[np.int64],
    seed: int,
) -> None: ...
