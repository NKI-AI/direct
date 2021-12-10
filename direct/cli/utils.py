# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import pathlib

from direct.utils.io import check_is_valid_url


def file_or_url(path):
    if check_is_valid_url(path):
        return path
    path = pathlib.Path(path)
    if path.is_file():
        return path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file or url.")
