# coding=utf-8
# # Copyright (c) DIRECT Contributors
# import random
# import pytest
#
# from direct.data.sampler import (
#     DistributedSequentialSampler,
#     BatchVolumeSampler,
#     DistributedSampler,
#     ConcatDatasetBatchSampler,
# )
# from torch.utils.data import ConcatDataset
#
#
# @pytest.fixture
# def dataset():
#     class TestDS:
#         def __init__(self, num_samples):
#             self.volume_indices = {}
#             lower_number = 0
#             for idx in range(num_samples):
#                 upper_number = lower_number + random.randint(1, 25)
#                 self.volume_indices[f"label_{idx}"] = range(lower_number, upper_number)
#                 lower_number = upper_number
#
#             self.reverse_dict = {}
#             for k, v in self.volume_indices.items():
#                 for _ in list(v):
#                     self.reverse_dict[_] = k
#
#         def __getitem__(self, idx):
#             return idx
#
#         def __len__(self):
#             return len(self.volume_indices)
#
#     return TestDS
#
#
# @pytest.mark.parametrize("num_samples", [10, 31, 68, 811])
# @pytest.mark.parametrize("num_replicas", [1, 3, 4, 6, 8])
# def test_distributed_sequential_sampler(dataset, num_samples, num_replicas):
#     """Tests if all samples are disjoint and unique."""
#     ds = dataset(num_samples)
#     indices_per_process = []
#     for rank in range(num_replicas):
#         sampler = DistributedSequentialSampler(ds, num_replicas=num_replicas, rank=rank)
#         indices = [_ for _ in sampler]
#         assert len(indices) == len(set(indices))
#         indices_per_process += indices
#     assert len(indices_per_process) == len(set(indices_per_process))
#
#
# @pytest.mark.parametrize("batch_size", [1, 3, 5, 8, 16, 32])
# @pytest.mark.parametrize("num_samples", [10, 31, 68, 811])
# @pytest.mark.parametrize("num_replicas", [1, 3, 4, 6, 8])
# def test_batch_volume_sampler(dataset, batch_size, num_samples, num_replicas):
#     ds = dataset(num_samples)
#
#     for rank in range(num_replicas):
#         sampler = DistributedSequentialSampler(ds, num_replicas=num_replicas, rank=rank)
#         batch_sampler = BatchVolumeSampler(sampler, batch_size)
#         batches = [_ for _ in batch_sampler]
#         output = []
#         for batch in batches:
#             names = []
#             for idx in batch:
#                 names.append(ds.reverse_dict[idx])
#             output.append((batch, set(names)))
#
#         assert all([len(_[1]) == 1 for _ in assert])
#
#
# @pytest.mark.parametrize("dataset_sizes", [[1], [1, 9], [19, 111, 7787, 2939]])
# @pytest.mark.parametrize("batch_size", [1, 3, 7, 8, 16])
# def test_concat_dataset_batch_sampler(dataset, dataset_sizes, batch_size):
#     # Create a list of datasets
#     datasets = [dataset(num_samples) for num_samples in dataset_sizes]
#     dataset = ConcatDataset(datasets)
#
#     dataset_indices = {}
#     curr_val = 0
#     for idx in range(len(dataset_sizes)):
#         indices_for_curr_dataset = list(range(curr_val, dataset.cumulative_sizes[idx]))
#         curr_val = dataset.cumulative_sizes[idx]
#         for _ in indices_for_curr_dataset:
#             dataset_indices[_] = idx
#
#     sampler = DistributedSampler(dataset, shuffle=True)
#     batch_sampler = ConcatDatasetBatchSampler(sampler, batch_size=batch_size)
#
#     idx = 0
#     batches = []
#     for batch in batch_sampler:
#         batches.append([int(_.numpy()) for _ in batch])
#         if idx > 1001:
#             break
#         idx += 1
#
#     # Make sure each batch comes from precisely one dataset
#     for batch in batches:
#         indices = list(set([dataset_indices[_] for _ in batch]))
#         assert len(indices) == 1
