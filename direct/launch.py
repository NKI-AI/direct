# coding=utf-8
# Copyright (c) DIRECT Contributors

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/989f52d67d05445ccd030d8f13d6cc53e297fb91/detectron2/engine/launch.py
# Changes:
# - Docstring to match the rest of the library.
# - Calls to other subroutines which do not exist in DIRECT.
# - Stylistic changes.

import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable

from direct.utils import communication

import logging

logger = logging.getLogger(__name__)

__all__ = ["launch", "launch_distributed"]


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch_distributed(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
):
    """

    Parameters
    ----------
    main_func : Callable
        A function that will be called by `main_func(*args)`.
    num_gpus_per_machine : int
        The number of GPUs per machine.
    num_machines :
        The number of machines.
    machine_rank : int
        The rank of this machine (one per machine).
    dist_url : str
        url to connect to for distributed training, including protocol e.g. "tcp://127.0.0.1:8686".
        Can be set to auto to automatically select a free port on localhost
    args : tuple
        arguments passed to main_func.

    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            if num_machines != 1:
                raise ValueError("dist_url=auto cannot work with distributed training.")
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
):
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger = logging.getLogger(__name__)
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error(f"Process group URL: {dist_url}")
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    communication.synchronize()
    logger.info(f"Global rank {global_rank}.")
    logger.info("Synchronized GPUs.")

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert communication._LOCAL_PROCESS_GROUP is None  # noqa
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            communication._LOCAL_PROCESS_GROUP = pg

    main_func(*args)


def launch(
    func: Callable,
    num_machines: int,
    num_gpus: int,
    machine_rank: int,
    dist_url: str,
    *args,
):
    """
    Launch the training, in case there is only one GPU available the function can be called directly.

    Parameters
    ----------
    func : callable
        function to launch
    num_machines : int
    num_gpus : int
    machine_rank : int
    dist_url : str
    args : arguments to pass to func

    Returns
    -------
    None
    """
    # There is no need for the launch script within one node and at most one GPU.
    if num_machines == 1 and num_gpus <= 1:
        if torch.cuda.device_count() > 1:
            logger.warning(f"Device count is {torch.cuda.device_count()}, b")
        func(*args)
    elif torch.cuda.device_count() > 1 and num_gpus <= 1:
        print(
            f"Device count is {torch.cuda.device_count()}, yet number of GPUs is {num_gpus}. "
            f"Unexpected behavior will occur. Consider exposing less GPUs (e.g. through docker). Exiting."
        )
        sys.exit()

    else:
        launch_distributed(
            func,
            num_gpus,
            num_machines=num_machines,
            machine_rank=machine_rank,
            dist_url=dist_url,
            args=args,
        )
