
import math
import torch
import torch.distributed as dist

from copy import copy
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler
from .dataset import GroupBatchDataset

# __all__ = ["DistributedGroupSampler", ]

# class DistributedGroupSampler(Sampler):
#     r"""Sampler that restricts data loading to a subset of the dataset.

#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
#     process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
#     :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
#     original dataset that is exclusive to it.

#     .. note::
#         Dataset is assumed to be of constant size and that any instance of it always
#         returns the same elements in the same order.

#     Args:
#         dataset: Dataset used for sampling.
#         num_replicas (int, optional): Number of processes participating in
#             distributed training. By default, :attr:`world_size` is retrieved from the
#             current distributed group.
#         rank (int, optional): Rank of the current process within :attr:`num_replicas`.
#             By default, :attr:`rank` is retrieved from the current distributed
#             group.
#     """

#     def __init__(self, dataset: GroupBatchDataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None) -> None:
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
#         self.dataset = copy(dataset)
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
        
#         self.num_groups = len(self.dataset.groups) // self.num_replicas
#         self.num_samples = self.num_groups * self.dataset.group_size // self.dataset.batch_size
#         self.rank_chunks = self.dataset.sample_sub_groups(num_replicas=self.num_replicas)


#     def __iter__(self):
#         self.dataset.cur_groups = self.rank_chunks[self.rank]
#         return iter(self.dataset)



#     def __len__(self) -> int:
#         return self.num_samples


#     def set_epoch(self, epoch: int) -> None:
#         self.epoch = epoch



class DistributedGroupSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
    """

    def __init__(self, dataset: GroupBatchDataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas


    def __iter__(self):
        indices = list(range(len(self.dataset)))[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)



    def __len__(self) -> int:
        return self.num_samples


    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch