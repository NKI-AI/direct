# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib

from direct.utils.io import read_list


def get_filenames_for_datasets(cfg, files_root, data_root):
    """
    Given a list of filenames of data points, concatenate these into a large list of full filenames

    Parameters
    ----------
    cfg : cfg-object
        cfg object having property lists having the relative paths compared to files root.
    files_root : pathlib.Path
    data_root : pathlib.Path

    Returns
    -------

    """
    if not cfg.lists:
        return []
    filter_filenames = []
    for curr_list in cfg.lists:
        filter_filenames += [data_root / pathlib.Path(_) for _ in read_list(pathlib.Path(files_root) / curr_list)]

    return filter_filenames
