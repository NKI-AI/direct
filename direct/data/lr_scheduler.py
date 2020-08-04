# coding=utf-8
# Copyright (c) DIRECT Contributors

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/989f52d67d05445ccd030d8f13d6cc53e297fb91/detectron2/solver/lr_scheduler.py
# Changes:
# - Docstring to match the rest of the library.
# - Calls to other subroutines which do not exist in DIRECT.
# - Stylistic changes.

import numpy as np
import torch
import logging

from bisect import bisect_right


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.logger = logging.getLogger(type(self).__name__)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the __optimizer or logger.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["__optimizer", "logger"]
        }


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iterations=500,
        warmup_method="linear",
        last_iteration=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones}."
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                f"Only `constant` or `linear` warmup_method accepted got {warmup_method}."
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iterations = warmup_iterations
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_iteration)

        self.logger.info(
            f"Initialized with gamma {gamma}, warmup_factor {warmup_factor},"
            f" warmup_iterations {warmup_iterations} and warmup_method {warmup_method}."
        )

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iterations:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iterations
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
