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
import pathlib

import h5py
import numpy as np
from tqdm import tqdm


def extract_mask(filename):
    """Extract the mask from masked k-space data, these are not explicitly given.

    Parameters
    ----------
    filename : pathlib.Path

    Returns
    -------
    np.ndarray
    """
    with h5py.File(filename, "r") as f:
        kspace = f["kspace"]
        size = kspace.shape[0]
        out = np.abs(kspace[0])
        for idx in range(1, size):
            out += np.abs(kspace[idx])

    sampling_mask = ~(np.abs(out).sum(axis=-1) == 0)

    return sampling_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testing_root", type=pathlib.Path, help="Path to the testing set.")
    parser.add_argument("output_directory", type=pathlib.Path, help="Path to the DoIterationOutput directory.")

    args = parser.parse_args()

    # Find all h5 files in the testing root
    testing_files = list(args.testing_root.glob("*.h5"))
    print(f"Found {len(testing_files)} files in {args.testing_root}.")
    print("Computing kspace masks...")

    for testing_file in tqdm(testing_files):
        mask = extract_mask(testing_file)
        np.save(args.output_directory / (testing_file.stem + ".npy"), mask)

    print("Computed masks.")
