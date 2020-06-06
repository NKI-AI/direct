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
        return {key: value for key, value in self.__dict__.items() if key not in ['__optimizer', 'logger']}


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
        warmup_iters=500,
        warmup_method='linear',
        last_iteration=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(f'Milestones should be a list of increasing integers. Got {milestones}.')

        if warmup_method not in ('constant', 'linear'):
            raise ValueError(f'Only `constant` or `linear` warmup_method accepted got {warmup_method}.')

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_iteration)

        self.logger.info(f'Initialized with gamma {gamma}, warmup_factor {warmup_factor},'
                         f' warmup_iters {warmup_iters} and warmup_method {warmup_method}.')

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in ['__optimizer', 'logger']}


# From https://github.com/Harshvardhan1/cyclic-learning-schedulers-pytorch/blob/master/cyclicLR.py
class CyclicLinearLR(LRScheduler):
    """
    Implements reset on milestones inspired from Linear learning rate decay

    Set the learning rate of each parameter group using a linear decay
    schedule, where $\eta_{max}$ is set to the initial lr and
    $T_{cur}$ is the number of epochs since the last restart:

    $\eta_t = \eta_{min} + (\eta_{max} - \eta_{min})(1 -\frac{T_{cur}}{T_{max}})$
    When last_iteration > last set milestone, lr is automatically set to \eta_{min}

    Args:
        optimizer (Optimizer): Wrapped __optimizer.
        milestones (list of ints): List of iteration indices. Must be increasing.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last iteration. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones, eta_min=0, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(f'Milestones should be a list of increasing integers. Got {milestones}.')

        self.eta_min = eta_min
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx - 1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        return [self.eta_min + (base_lr - self.eta_min) *
                (1. - 1.0 * curr_pos / width)
                for base_lr in self.base_lrs]


class WarmRestartLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

        $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))$$

    When last_iteration=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implements the
    annealing and the restarts.

    Args:
        optimizer (Optimizer): Wrapped __optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last iteration. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mul=2, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mul = T_mul
        self.eta_min = eta_min
        super(WarmRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= 0:
            num_cycles = np.floor(np.log((self.last_epoch * (self.T_mul - 1) / self.T_max) + 1.0, self.T_mul)) - 1
            num_cycles = max(num_cycles, 0)
            last_restart = int(self.T_max * (np.pow(self.T_mul, num_cycles + 1) - 1) / (self.T_mul - 1))
            T_delta = max(self.last_epoch - last_restart, 0)
            T_cur = self.T_max * int(np.pow(self.T_mul, num_cycles))
        else:
            T_delta = 0
            T_cur = 1

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(0.5 * np.pi * T_delta / T_cur)) / 2
                for base_lr in self.base_lrs]


def build_optim(params, cfg):
    _optim = {'Adam': torch.optim.Adam, 'FusedAdam': FusedAdam,
              'RMSprop': torch.optim.RMSprop, 'SGD': torch.optim.SGD}[cfg.OPTIMIZER]
    optimizer = _optim(params, cfg.STARTER_LR, weight_decay=cfg.WEIGHT_DECAY)
    return optimizer

'''
#################################
# TEST FOR SCHEDULER
#################################
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
net = nn.Sequential(nn.Linear(2,2))
milestones = [(2**input_image)*300 for input_image in range(30)]
__optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.9,weight_decay=0.0005,nesterov=True)
scheduler = CyclicCosAnnealingLR(__optimizer,milestones=milestones,eta_min=1e-6)
lr_log = []
for i in range(20*300):
    __optimizer.step()
    scheduler.step()
    for param_group in __optimizer.param_groups:
        lr_log.append(param_group['lr'])
plt.plot(lr_log)
plt.show()
'''
