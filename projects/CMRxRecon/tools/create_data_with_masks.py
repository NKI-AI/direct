import glob
import logging
import os
import pathlib
from typing import Union

import h5py

ACCELERATIONS = ["04", "08", "10"]

logger = logging.getLogger("CreateCMRData")


def create_data_with_masks(data_path: Union[str, pathlib.Path], save_path: Union[str, pathlib.Path]):
    """
    Parameters
    ----------
    data_path : str
        Must point to directory with structure:
        Cine/Mapping
        - AccFactor04
        -- P001
        -- P002
        - AccFactor08
        - AccFactor10
        - FullSample
        -- P001
        -- P002
    save_path : str
        Where to store data with masks.

    Example
    -------
    >>> create_data_with_masks(
            "../CMRxRecon/MICCAIChallenge2023/ChallegeData/MultiCoil/Mapping/TrainingSet/",
            "../CMRxRecon/MICCAIChallenge2023/ChallegeData/MultiCoil/Mapping/TrainingSet/WithMasks/"
        )

    """
    data_path = pathlib.Path(data_path)
    save_path = pathlib.Path(save_path)

    patients = glob.glob(str(data_path / "FullSample") + "/P*")

    for patient in patients:
        patient_name = pathlib.Path(patient).name
        patient_sub_dir = save_path / patient_name
        # Create new dir for patient
        if not os.path.exists(patient_sub_dir):
            os.makedirs(patient_sub_dir)

        fully_sampled_mat_files = glob.glob(patient + "/*.mat")

        for mat_file in fully_sampled_mat_files:
            try:
                fully_sampled_file = h5py.File(mat_file, "r")
            except Exception as err:
                logger.info(f"Couldn't read file {mat_file}. Exiting with Exception: {err}.")
                continue

            mat_file_name = pathlib.Path(mat_file).name
            mat_with_masks_path = patient_sub_dir / mat_file_name
            if mat_with_masks_path.exists():
                continue
            else:
                logger.info(f"Creating file {mat_with_masks_path}..")
            file_with_masks = h5py.File(mat_with_masks_path, "w")

            fully_sampled_file.copy("kspace_full", file_with_masks)
            fully_sampled_file.close()

            for acceleration in ACCELERATIONS:
                mask_path = (
                    data_path / f"AccFactor{acceleration}/" / patient_name / mat_file_name.replace(".mat", "_mask.mat")
                )
                mask_key = f"mask{acceleration}"
                try:
                    mask_file = h5py.File(mask_path, "r")

                    mask_file.copy(mask_key, file_with_masks)
                    mask_file.close()
                except Exception as err:
                    logger.info(f"Couldn't loaf mask for R={acceleration} with error: {err}.")
                    continue

            file_with_masks.close()
