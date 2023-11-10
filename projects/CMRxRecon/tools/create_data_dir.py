import argparse
import logging
import pathlib
from argparse import RawTextHelpFormatter

from create_data_with_masks import ACCELERATIONS, create_data_with_masks
from create_symlinks import create_symlinks

logger = logging.getLogger("CreateTrainingData")
logger.setLevel(logging.INFO)

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
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    │   │   ├── TestSet
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    """

SYMLINKS_PATH_STRUCTURE = """
    target_path
    ├── MultiCoil
    │   ├── Cine_or_Mapping
    │   │   ├── training
    │   │   │   ├── P001_<..>.h5
    │   │   │   ├── with_masks_P001_<..>.h5
    │   │   ├── validation
    │   │   │   ├── AccFactor04
    │   │   │   |   ├── P001_<..>.h5
    │   │   │   ├── AccFactor08
    │   │   │   |   ├── P001_<..>.h5
    │   │   │   └── AccFactor10
    │   │   │   |   ├── P001_<..>.h5
    │   │   ├── test
    │   │   │   ├── AccFactor04
    │   │   │   |   ├── P001_<..>.h5
    │   │   │   ├── AccFactor08
    │   │   │   |   ├── P001_<..>.h5
    │   │   │   └── AccFactor10
    │   │   │   |   ├── P001_<..>.h5
    """

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

training_symbolic_path = args.target_path / "MultiCoil" / args.data_type / "training"

# Check if the required directories exist
if not data_path.exists():
    logger.error(f"Data path '{data_path}' does not exist.")
    exit(1)

if not training_set_path.exists():
    logger.error(f"Training set path '{training_set_path}' does not exist.")
    exit(1)

if not full_sample_path.exists():
    logger.error(f"Full sample path '{full_sample_path}' does not exist.")
    exit(1)

# Create fully sampled data with masks
logger.info(f"Creating training fully sampled data with masks. Saving at {full_sample_path}.")
create_data_with_masks(training_set_path, full_sample_with_masks_path)

# Create symlinks for training. All data need to be in one directory.
create_symlinks(full_sample_path, training_symbolic_path)
create_symlinks(full_sample_with_masks_path, training_symbolic_path, "with_masks_")

# Create symlinks for validation and testing
validation_set_path = data_path / "ValidationSet"
test_set_path = data_path / "TestSet"

validation_symbolic_path = args.target_path / "MultiCoil" / args.data_type / "validation"
test_symbolic_path = args.target_path / "MultiCoil" / args.data_type / "test"

for acceleration in ACCELERATIONS:
    validation_acceleration_path = validation_set_path / f"AccFactor{acceleration}"
    if validation_acceleration_path.exists():
        logger.info(
            f"Creating symbolic paths for {validation_acceleration_path} "
            f"at {validation_symbolic_path / f'AccFactor{acceleration}'}..."
        )
        create_symlinks(validation_acceleration_path, validation_symbolic_path / f"AccFactor{acceleration}")
    else:
        logger.info(f"Path {validation_acceleration_path} doesn't exist. Skipping...")

    test_acceleration_path = test_set_path / f"AccFactor{acceleration}"
    if test_acceleration_path.exists():
        logger.info(
            f"Creating symbolic paths for {test_acceleration_path} "
            f"at {test_symbolic_path / f'AccFactor{acceleration}'}..."
        )
        create_symlinks(test_acceleration_path, test_symbolic_path / f"AccFactor{acceleration}")
    else:
        logger.info(f"Path {test_acceleration_path} doesn't exist. Skipping...")

logger.info(f"Data processing and symlink creation for '{args.data_type}' data completed.")
