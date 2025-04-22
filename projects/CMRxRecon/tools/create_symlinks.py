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
import glob
import logging
import os
import pathlib
from typing import Union

logger = logging.getLogger("CreateCMRData")


def create_symlinks(base_path: Union[str, pathlib.Path], sym_base_path: Union[str, pathlib.Path], prefix: str = ""):
    """Creates symlinks of data from different directories data in a single directory.

    Paths should be provided in absolute form.

    Parameters
    ----------
    base_path : Union[str, pathlib.Path],
        Must point to directory with structure:
        Cine/Mapping
        - Type of Data
        -- P001
        -- P002
    sym_base_path : Union[str, pathlib.Path],
        Path to save symbolic data.
    prefix : str
        Prefix for symlink data.
    """

    base_path = pathlib.Path(base_path)

    sym_base_path = pathlib.Path(sym_base_path)

    if not sym_base_path.exists():
        logger.info(f"Creating symbolic path: {sym_base_path}...")
        sym_base_path.mkdir(parents=True, exist_ok=True)

    patients = glob.glob(str(base_path) + "/P*")

    for patient in patients:
        patient_name = pathlib.Path(patient).name
        mat_files = glob.glob(patient + "/*.mat")
        mat_files = [pathlib.Path(m) for m in mat_files if "mask" not in m]
        for mat_file in mat_files:
            mat_name = pathlib.Path(mat_file).name
            new_name = prefix + patient_name + "_" + mat_name

            sym_name = sym_base_path / new_name
            if not sym_name.exists():
                logger.info(f"Creating symbolic link for {mat_file.absolute()} at  {sym_name}...")
                os.symlink(mat_file.absolute(), sym_name)
            else:
                logger.info(f"Symbolic link {sym_name} exists. Skipping...")
