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
from __future__ import annotations

import logging
import pathlib
import sys
from os import PathLike

logger = logging.getLogger(__name__)

_ASCII_LOGO_FILENAME = "direct_logo_ascii.txt"


def _ascii_logo_path() -> pathlib.Path | None:
    """Resolve ``logo/direct_logo_ascii.txt`` from the repository root when available."""
    # direct/utils/logging.py -> repo root is parents[2]
    candidate = pathlib.Path(__file__).resolve().parents[2] / "logo" / _ASCII_LOGO_FILENAME
    if candidate.is_file():
        return candidate
    return None


def _load_ascii_logo() -> str | None:
    path = _ascii_logo_path()
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def _emit_ascii_logo(root: logging.Logger) -> None:
    """Write the ASCII logo at the start of logging (stdout / log files), unprefixed."""
    logo = _load_ascii_logo()
    if logo is None:
        return
    if not logo.endswith("\n"):
        logo = logo + "\n"
    for handler in root.handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None:
            stream.write(logo)
            stream.flush()
            continue
        # FileHandler exposes the path; append raw art so log files include it too.
        base = getattr(handler, "baseFilename", None)
        if base is not None:
            with open(base, "a", encoding="utf-8") as fh:
                fh.write(logo)


def setup(
    use_stdout: bool | None = True,
    filename: PathLike | None = None,
    log_level: int | str = "INFO",
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

    for name in logging.root.manager.loggerDict:  # pylint: disable = E1101
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

    logger.warning("DIRECT is not intended for clinical use.")
    _emit_ascii_logo(root)
