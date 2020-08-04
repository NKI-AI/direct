# coding=utf-8
# Copyright (c) DIRECT Contributors
import json
import argparse
import pathlib
import numpy as np


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Find the best checkpoint for a given metric",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("metrics_path", type=pathlib.Path, help="Path to metrics.json")
    parser.add_argument("key", type=str, help="Key to use to find the best checkpoint.")

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.metrics_path / "metrics.json", "r") as f:
        data = f.readlines()
        data = [json.loads(_) for _ in data]

    x = np.asarray([(int(_["iteration"]), -_[args.key]) for _ in data if args.key in _])
    out = x[np.where(x[:, 1] == x[:, 1].max())][0]

    print(f"{args.key} - {int(out[0])}: {out[1]}")


if __name__ == "__main__":
    main()
