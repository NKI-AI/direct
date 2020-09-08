# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import pathlib


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Explore data for FastMRI challenge 2020.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data_path", type=pathlib.Path, help="Path to metrics.json")

    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
