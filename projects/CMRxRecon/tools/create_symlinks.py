import glob
import os
import pathlib
from typing import Union


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

    if not os.path.exists(sym_base_path):
        print(f"Creating symbolic path: {sym_base_path}...")
        os.makedirs(sym_base_path)

    patients = glob.glob(str(base_path) + "/P*")

    for patient in patients:
        patient_name = pathlib.Path(patient).name
        mat_files = glob.glob(patient + "/*.mat")
        mat_files = [m for m in mat_files if "mask" not in m]
        for mat_file in mat_files:
            mat_name = pathlib.Path(mat_file).name
            new_name = prefix + patient_name + "_" + mat_name

            sym_name = sym_base_path / new_name
            os.symlink(os.path.abspath(mat_file), sym_name)
            print(f"Creating symbolic link {sym_name}...")
