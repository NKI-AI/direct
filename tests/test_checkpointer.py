# coding=utf-8
# Copyright (c) DIRECT Contributors

import datetime
import pathlib
import tempfile

import pytest
import torch
import torch.nn as nn

from direct.checkpointer import Checkpointer


def create_checkpointables(*keys):

    checkpointables = dict()
    checkpointables["model"] = nn.Linear(2, 2)

    if "optimizer" in keys:
        checkpointables["optimizer"] = torch.optim.Adam(checkpointables["model"].parameters())

    if "sensitivity_model" in keys:
        checkpointables["sensitivity_model"] = nn.Linear(2, 2)

    if "__author__" in keys:
        checkpointables["__author__"] == "Jane Doe"

    if "__datetime__" in keys:
        checkpointables["__datetime__"] == datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "__version__" in keys:
        checkpointables["__version__"] == "0.0.0"

    if "__mixed_precision__" in keys:
        checkpointables["__mixed_precision__"] = False

    return checkpointables


@pytest.mark.parametrize(
    "checkpoint_ids",
    [
        [10],
        [20, 40],
    ],
)
@pytest.mark.parametrize(
    "checkpointables_keys",
    [
        [],
        ["sensitivity_model", "optimizer"],
        ["sensitivity_model", "optimizer", "__author__", "__version__", "__datetime__", "__mixed_precision__"],
    ],
)
def test_checkpointer(checkpoint_ids, checkpointables_keys):

    with tempfile.TemporaryDirectory() as tempdir:
        for checkpoint_id in checkpoint_ids:
            checkpointables = create_checkpointables(checkpointables_keys)

            checkpointer = Checkpointer(save_directory=pathlib.Path(tempdir), save_to_disk=True, **checkpointables)
            # Test save function
            checkpointer.save(iteration=checkpoint_id)
            # Test load function
            checkpointer.load(iteration=checkpoint_id)

        # Test that '-1' option loads the same checkpoint as the 'latest' option
        last_checkpoint = checkpointer.load(iteration=-1)
        for key in last_checkpoint:
            assert key in checkpointer.load("latest")
            if isinstance(last_checkpoint[key], torch.Tensor):
                torch.allclose(checkpointer.load(iteration="latest")[key], last_checkpoint[key])
