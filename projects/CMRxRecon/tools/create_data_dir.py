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
import argparse
import logging
import pathlib
from argparse import RawTextHelpFormatter

from create_data_with_masks import ACCELERATIONS, create_data_with_masks
from create_symlinks import create_symlinks

# Define the available options for the 'data_type' argument
DATA_TYPES = ["Cine", "Mapping"]

ASSUMED_BASE_PATH_STRUCTURE = """
    base_path
    ├── MultiCoil
    │   ├── Cine_or_Mapping
    │   │   ├── TrainingSet
    │   │   │   ├── FullSample
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    │   │   ├── ValidationSet
    │   │   │   ├── FullSample
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    │   │   ├── TestSet
    │   │   │   ├── FullSample
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    """

SYMLINKS_PATH_STRUCTURE = """
    target_path
    ├── MultiCoil
    │   ├── training
    │   │   ├── P001_T1map.mat
    │   │   ├── with_masks_P001_T1map.mat
    │   │   ├── P001_cine_sax.mat
    │   │   ├── with_masks_P001_cine_sax.mat
    │   │   ├── ...
    │   ├── validation
    │   │   ├── P001_T1map.mat
    │   │   ├── P001_T2map.mat
    │   │   ├── P001_cine_lax.mat
    │   │   ├── P001_cine_sax.mat
    │   │   ├── ...
    │   │   ├── Cine or Mapping
    │   │   │   ├── AccFactor04
    │   │   │   |   ├── P001_<..>.mat
    │   │   │   ├── AccFactor08
    │   │   │   |   ├── P001_<..>.mat
    │   │   │   └── AccFactor10
    │   │   │   |   ├── P001_<..>.mat
    │   │   ├── test
    │   │   ├── P001_T1map.mat
    │   │   ├── P001_cine_sax.mat
    │   │   ├── ...
    │   │   ├── Cine or Mapping
    │   │   │   ├── AccFactor04
    │   │   │   |   ├── P001_<..>.mat
    │   │   │   ├── AccFactor08
    │   │   │   |   ├── P001_<..>.mat
    │   │   │   └── AccFactor10
    │   │   │   |   ├── P001_<..>.mat
    """


def main():
    logger = logging.getLogger("CreateCMRData")
    logger.setLevel(logging.DEBUG)

    # Create a file handler which logs even debug messages
    fh = logging.FileHandler("CreateCMRData.log")
    fh.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Process data and create symlinks",
        formatter_class=RawTextHelpFormatter,
    )

    # Add arguments for base path, target path, and data type
    parser.add_argument(
        "--base_path",
        type=pathlib.Path,
        required=True,
        help=f"Absolute path to the base directory where data is located. Assumed structure: {ASSUMED_BASE_PATH_STRUCTURE}",
    )

    parser.add_argument(
        "--target_path",
        type=pathlib.Path,
        required=True,
        help=f"Absolute path where symlinks will be created. Symlinks directory structure: {SYMLINKS_PATH_STRUCTURE}",
    )

    parser.add_argument(
        "--data_type",
        choices=DATA_TYPES,
        required=True,
        help="Choose 'Cine' or 'Mapping' to specify the type of data to process.",
    )

    parser.add_argument(
        "--create_training_data_with_masks",
        required=False,
        action="store_true",
        default=False,
        help="If provided then fully sampled training data with masks will be created.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the specified base path exists
    if not args.base_path.exists():
        logger.error(f"Base path '{args.base_path}' does not exist.")
        exit(1)

    # Check if the specified data type is valid
    if args.data_type not in DATA_TYPES:
        logger.error("Invalid data type. Use 'Cine' or 'Mapping'.")
        exit(1)

    # Construct the paths for data processing and symlink creation
    data_path = args.base_path / "MultiCoil" / args.data_type
    training_set_path = data_path / "TrainingSet"

    full_sample_path = training_set_path / "FullSample"
    full_sample_with_masks_path = training_set_path / "FullSampleWithMasks"

    training_symbolic_path = args.target_path / "MultiCoil" / "training"

    # Check if the required directories exist
    if not data_path.exists():
        logger.error(f"Data path '{data_path}' does not exist.")
        exit(1)

    if not training_set_path.exists():
        logger.error(f"Training set path '{training_set_path}' does not exist.")
        exit(1)

    if not full_sample_path.exists():
        logger.error(f"Training fully sampled data path '{full_sample_path}' does not exist.")
        exit(1)

    if args.create_training_data_with_masks:
        # Create fully sampled data with masks
        logger.info(f"Creating training fully sampled data with masks. Saving at {full_sample_path}.")
        create_data_with_masks(training_set_path, full_sample_with_masks_path)

    # Create symlinks for training. All data need to be in one directory.
    create_symlinks(full_sample_path, training_symbolic_path)
    create_symlinks(full_sample_with_masks_path, training_symbolic_path, "with_masks_")

    # Create symlinks for validation and testing
    validation_set_path = data_path / "ValidationSet"
    validation_symbolic_path = args.target_path / "MultiCoil" / "validation"
    validation_full_sample_path = validation_set_path / "FullSample"

    test_set_path = data_path / "TestSet"
    test_symbolic_path = args.target_path / "MultiCoil" / "test"
    test_full_sample_path = test_set_path / "FullSample"

    # Check if the required directories exist
    if not validation_set_path.exists():
        logger.error(f"Validation set path '{validation_set_path}' does not exist.")
        exit(1)

    if not validation_full_sample_path.exists():
        logger.warning(f"Validation full sample path '{validation_full_sample_path}' does not exist. Skipping...")
    else:
        logger.info(f"")
        # Create symlinks for fully sampled validation data. All data need to be in one directory.
        create_symlinks(validation_full_sample_path, validation_symbolic_path)

    # Check if the required test directory exist
    if not test_set_path.exists():
        logger.error(f"Test set path '{test_set_path}' does not exist.")
        exit(1)

    if not test_full_sample_path.exists():
        logger.warning(f"Test full sample path '{test_full_sample_path}' does not exist. Skipping...")
    else:
        # Create symlinks for fully sampled test data. All data need to be in one directory.
        create_symlinks(test_full_sample_path, test_symbolic_path)

    for acceleration in ACCELERATIONS:
        validation_acceleration_path = validation_set_path / f"AccFactor{acceleration}"
        if validation_acceleration_path.exists():
            logger.info(
                f"Creating symbolic paths for {validation_acceleration_path} "
                f"at {validation_symbolic_path / f'AccFactor{acceleration}'}..."
            )
            create_symlinks(validation_acceleration_path, validation_symbolic_path / f"AccFactor{acceleration}")
        else:
            logger.info(f"Path {validation_acceleration_path} does not exist. Skipping...")

        test_acceleration_path = test_set_path / f"AccFactor{acceleration}"
        if test_acceleration_path.exists():
            logger.info(
                f"Creating symbolic paths for {test_acceleration_path} "
                f"at {test_symbolic_path / f'AccFactor{acceleration}'}..."
            )
            create_symlinks(test_acceleration_path, test_symbolic_path / f"AccFactor{acceleration}")
        else:
            logger.info(f"Path {test_acceleration_path} does not exist. Skipping...")

    logger.info(f"Data processing and symlink creation for '{args.data_type}' data completed.")


if __name__ == "__main__":
    main()
