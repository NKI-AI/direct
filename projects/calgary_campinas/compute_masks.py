# coding=utf-8
# Copyright (c) DIRECT Contributors
import h5py
import pathlib
import numpy as np
import argparse

from tqdm import tqdm


def extract_mask(filename):
    """
    Extract the mask from masked k-space data, these are not explicitly given.

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
    parser.add_argument(
        "testing_root", type=pathlib.Path, help="Path to the testing set."
    )
    parser.add_argument(
        "output_directory", type=pathlib.Path, help="Path to the output directory."
    )

    args = parser.parse_args()

    # Find all h5 files in the testing root
    testing_files = list(args.testing_root.glob("*.h5"))
    print(f"Found {len(testing_files)} files in {args.testing_root}.")
    print(f"Computing kspace masks...")

    for testing_file in tqdm(testing_files):
        mask = extract_mask(testing_file)
        np.save(args.output_directory / (testing_file.stem + ".npy"), mask)

    print(f"Computed masks.")
