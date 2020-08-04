# coding=utf-8
# Copyright (c) DIRECT Contributors
# `DistributedSampler` below taken from Detectron 2, licensed under Apache 2.0.
# Changes:
# - Docstring to match the rest of the library
# - Calls to other subroutines which do not exist in DIRECT.

import torch
import itertools

from typing import Optional
from torch.utils.data.sampler import Sampler

from direct.utils import communication


# https://stackoverflow.com/a/54802737
def chunks(list_to_chunk, number_of_chunks):
    """Yield number_of_chunks number of sequential chunks from list_to_chunk."""
    d, r = divmod(len(list_to_chunk), number_of_chunks)
    for i in range(number_of_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield list_to_chunk[si : si + (d + 1 if i < r else d)]


class DistributedSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Parameters
        ----------
        size : int
            Length of the underlying dataset.
        shuffle : bool
            If true, the indices will be shuffled.
        seed : int
            Initial seed of the shuffle, must be the same across all workers!
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = communication.shared_random_seed()
        self._seed = int(seed)

        self._rank = communication.get_rank()
        self._world_size = communication.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


class DistributedSequentialSampler(Sampler):
    """Sequential Sampler that restricts data loading to a subset of the dataset.

    It is useful during evaluation. It is especially useful when combined with
    `torch.nn.parallel.DistributedDataParallel`. Such that each process gets a subpart of the dataset.

    Notes
    =====
    Dataset is assumed to be of constant size.
    """

    def __init__(
        self, dataset, num_replicas=None, rank=None, limit_number_of_volumes=None
    ):
        if num_replicas is None:
            num_replicas = communication.get_world_size()
        if rank is None:
            rank = communication.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        filenames = list(self.dataset.volume_indices.keys())  # This is an OrderedDict
        if limit_number_of_volumes:
            filenames = filenames[:limit_number_of_volumes]

        chunked_filenames = list(chunks(filenames, self.num_replicas))
        filenames = chunked_filenames[self.rank]

        # Collect the indices belonging to these filenames.
        self.indices = []
        if self.rank < len(chunked_filenames):  # Otherwise there is nothing to fill.
            for filename in filenames:
                self.indices.extend(list(self.dataset.volume_indices[filename]))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
