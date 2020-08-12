# coding=utf-8
# Copyright (c) DIRECT Contributors
import random
import pytest

from direct.data.sampler import DistributedSequentialSampler, BatchVolumeSampler


class TestDS:
    def __init__(self):
        self.volume_indices = {}
        lower_number = 0
        for idx in range(11):
            upper_number = lower_number + random.randint(1, 25)
            self.volume_indices[f"label_{idx}"] = range(lower_number, upper_number)
            lower_number = upper_number

        self.reverse_dict = {}
        for k, v in self.volume_indices.items():
            for _ in list(v):
                self.reverse_dict[_] = k


@pytest.fixture
def dataset():
    return TestDS


def test_batch_volume_sampler(dataset):
    ds = dataset()
    sampler = DistributedSequentialSampler(ds, num_replicas=1, rank=0)
    batch_sampler = BatchVolumeSampler(sampler, 5)
    batches = [_ for _ in batch_sampler]
    output = []
    for batch in batches:
        names = []
        for idx in batch:
            names.append(ds.reverse_dict[idx])
        output.append((batch, set(names)))

    assert all([len(_[1]) == 1 for _ in output])
