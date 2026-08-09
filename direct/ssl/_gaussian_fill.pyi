"""Type stubs for the compiled :mod:`direct.ssl._gaussian_fill` extension."""

import numpy as np
import numpy.typing as npt

def gaussian_fill(
    nonzero_mask_count: int,
    nrow: int,
    ncol: int,
    center_x: int,
    center_y: int,
    std_scale: float,
    mask: npt.NDArray[np.int64],
    output_mask: npt.NDArray[np.int64],
    seed: int,
) -> npt.NDArray[np.int64]: ...
