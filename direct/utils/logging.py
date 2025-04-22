# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
import logging
import sys
from os import PathLike
from typing import Optional, Union


def setup(
    use_stdout: Optional[bool] = True,
    filename: Optional[PathLike] = None,
    log_level: Union[int, str] = "INFO",
) -> None:
    """Setup logging for DIRECT.

    Parameters
    ----------
    use_stdout: bool
        Write output to standard out.
    filename: PathLike
        Filename to write log to.
    log_level: str
        Logging level as in the `python.logging` library.

    Returns
    -------
    None
    """
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"]:
        raise ValueError(f"Unexpected log level got {log_level}.")

    logging.captureWarnings(True)
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)

    root = logging.getLogger()
    root.setLevel(log_level)

    for name in logging.root.manager.loggerDict:  # pylint: disable = E1101 # type: ignore
        if name.startswith("torch"):
            logging.getLogger(name).setLevel("WARNING")

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

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

    logging.warning("DIRECT is not intended for clinical use.")
