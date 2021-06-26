# coding=utf-8
# Copyright (c) DIRECT Contributors

import argparse
import glob
import json
import os
import pathlib

import h5py

from direct.data.transforms import *
from direct.functionals.challenges import *


def _get_filenames_from_lists(path_to_lst):
    names = []
    for list_name in glob.glob(os.path.join(path_to_lst, "*.lst")):
        with open(os.path.join(os.getcwd(), list_name), "r") as f:
            names.extend(f.readlines())

    return [pathlib.Path(name.strip("\n")) for name in names]


def _get_file_from_h5(pred_filename, target_filename):
    pred_rec = h5py.File(pred_filename, "r")
    pred_rec = torch.tensor(pred_rec["reconstruction"])[:, np.newaxis, :, :]

    target = h5py.File(target_filename, "r")

    target_kspace = np.array(target["kspace"][50:-50])
    target_kspace = to_tensor(target_kspace[..., ::2] + 1j * target_kspace[..., 1::2])

    # TODO(gy): Needed?
    # sampling_rate_slice_encode = 0.85
    # num_z = target_kspace.shape[1]
    # target_kspace[:, int(np.ceil(num_z * sampling_rate_slice_encode)):, :] = 0.0 + 0.0 * 1j

    target_rec = _get_reconstruction(target_kspace.permute(0, 3, 1, 2, 4))

    return pred_rec, target_rec


def _get_reconstruction(kspace):

    rec = ifft2(kspace, dim=(2, 3), centered=False)
    rec = root_sum_of_squares(rec, 1)[:, None, :, :]

    return rec


def _get_metrics(pred_rec, target_rec):

    ssim = calgary_campinas_ssim(target_rec, pred_rec).item()
    psnr = calgary_campinas_psnr(target_rec, pred_rec).item()
    vif = calgary_campinas_vif(target_rec, pred_rec).item()

    return {
        "calgary_campinas_psnr_metric": psnr,
        "calgary_campinas_ssim_metric": ssim,
        "calgary_campinas_vif_metric": vif,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("target_data_root", type=pathlib.Path, help="Path to the target data.")

    parser.add_argument("predicted_data_root", type=pathlib.Path, help="Path to the predicted data.")

    parser.add_argument(
        "--filenames-filter",
        dest="filenames_filter",
        type=pathlib.Path,
        required=True,
        help="Path to list of filenames to parse. Lists must be of type .'lst'.",
    )

    parser.add_argument("--name", dest="name", type=str, required=True, help="Name to store metrics")
    args = parser.parse_args()

    filenames = _get_filenames_from_lists(args.filenames_filter)

    metrics = dict()

    for filename in filenames:

        pred_filename = pathlib.Path(pathlib.PurePath(args.predicted_data_root, filename))
        target_filename = pathlib.Path(pathlib.PurePath(args.target_data_root, filename))

        if pred_filename.exists() and target_filename.exists():

            pred_rec, target_rec = _get_file_from_h5(pred_filename, target_filename)

            metrics[filename.name] = _get_metrics(pred_rec, target_rec)

    if len(metrics) > 0:
        with open(args.name + ".json", "w") as f:
            f.write(json.dumps(metrics, indent=4, sort_keys=True))
