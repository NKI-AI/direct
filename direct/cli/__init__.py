# coding=utf-8
# Copyright (c) DIRECT Contributors
"""DIRECT Command-line interface. This is the file which builds the main parser. Currently just a placeholder"""
import argparse
import sys


def main():
    """
    Console script for dlup.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="DIRECT utilities.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    args = root_parser.parse_args()
    args.subcommand(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
