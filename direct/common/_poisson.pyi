"""Type stubs for the compiled :mod:`direct.common._poisson` extension."""

import numpy as np
import numpy.typing as npt

def poisson(
    nx: int,
    ny: int,
    max_attempts: int,
    mask: npt.NDArray[np.int64],
    radius_x: npt.NDArray[np.float64],
    radius_y: npt.NDArray[np.float64],
    seed: int,
) -> None: ...
