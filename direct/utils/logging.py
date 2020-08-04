# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import sys
from typing import Optional
from os import PathLike


def setup(
    use_stdout: Optional[bool] = True,
    filename: Optional[PathLike] = None,
    log_level: Optional[str] = "INFO",
) -> None:
    """
    Setup logging for DIRECT.

    Parameters
    ----------
    use_stdout : bool
        Write output to standard out.
    filename : PathLike
        Filename to write log to.
    log_level : str
        Logging level as in the `python.logging` library.

    Returns
    -------
    None
    """
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"]:
        raise ValueError(f"Unexpected log level got {log_level}.")

    logging.captureWarnings(True)
    log_level = getattr(logging, log_level)

    root = logging.getLogger("")
    root.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    if use_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
