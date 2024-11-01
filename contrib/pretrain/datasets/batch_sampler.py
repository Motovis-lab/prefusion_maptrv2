# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
from torch.utils.data import BatchSampler, Sampler
from prefusion.registry import DATA_SAMPLERS


__all__ = ['SameSourceBatchSampler', 'SingleDatasetBatchSampler']

# TODO: maybe replace with a data_loader wrapper
@DATA_SAMPLERS.register_module()
class SameSourceBatchSampler(BatchSampler):
    """A batch sampler that ensures all samples in a batch come from the same dataset
    in a ConcatDataset.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        concat_dataset (ConcatDataset): The concatenated dataset.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = False,
                 ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.concat_dataset = sampler.dataset
        
        # 获取数据集的累积大小
        self.cumulative_sizes = self.concat_dataset.cumulative_sizes
        
        # 创建每个数据集对应的bucket
        self._dataset_buckets = [[] for _ in range(len(self.concat_dataset.datasets))]

    def _get_dataset_index(self, idx: int) -> int:
        """确定样本属于哪个数据集"""
        for dataset_idx, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                return dataset_idx
            if dataset_idx > 0:
                if idx < cumulative_size and idx >= self.cumulative_sizes[dataset_idx - 1]:
                    return dataset_idx
        return len(self.cumulative_sizes) - 1

    def __iter__(self):
        # 将索引分配到对应数据集的bucket中
        for idx in self.sampler:
            dataset_idx = self._get_dataset_index(idx)
            bucket = self._dataset_buckets[dataset_idx]
            bucket.append(idx)
            # 当bucket达到batch_size时，yield这个batch
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # 处理剩余的数据
        for bucket in self._dataset_buckets:
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket.clear()
                else:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]
        
        # 重置所有bucket
        self._dataset_buckets = [[] for _ in range(len(self.concat_dataset.datasets))]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


import torch
from torch.utils.data import Sampler, ConcatDataset
import numpy as np
from typing import Iterator, Optional

@DATA_SAMPLERS.register_module()
class SingleDatasetBatchSampler(BatchSampler):
    """
    A BatchSampler that ensures all samples in a batch come from the same dataset
    in a ConcatDataset.
    
    Args:
        concat_dataset (ConcatDataset): The concatenated dataset
        batch_size (int): Size of mini-batch
        drop_last (bool): If True, drop the last incomplete batch
        shuffle (bool): If True, shuffle the samples within each dataset
    """
    
    def __init__(self, 
                 sampler: Sampler, 
                 batch_size: int, 
                 drop_last: bool = False):
    
        self.concat_dataset = sampler.dataset.datasets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 获取每个数据集的累积大小
        self.cumulative_sizes = self.concat_dataset.cumulative_sizes
        
        # 计算每个数据集的大小
        self.dataset_sizes = []
        self.dataset_sizes.append(self.cumulative_sizes[0])
        for i in range(1, len(self.cumulative_sizes)):
            self.dataset_sizes.append(
                self.cumulative_sizes[i] - self.cumulative_sizes[i-1]
            )
            
    def __iter__(self) -> Iterator[list]:
        # 为每个数据集创建索引列表
        indices_per_dataset = []
        start_idx = 0
        
        for size in self.dataset_sizes:
            if self.shuffle:
                indices = torch.randperm(size).tolist()
            else:
                indices = list(range(size))
            
            # 添加偏移量使索引对应到连接数据集中的正确位置
            indices = [idx + start_idx for idx in indices]
            indices_per_dataset.append(indices)
            start_idx += size
            
        # 创建每个数据集的batch
        batches = []
        for dataset_indices in indices_per_dataset:
            num_samples = len(dataset_indices)
            if self.drop_last:
                num_batches = num_samples // self.batch_size
            else:
                num_batches = (num_samples + self.batch_size - 1) // self.batch_size
                
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                batch = dataset_indices[start_idx:end_idx]
                
                if not self.drop_last or len(batch) == self.batch_size:
                    batches.append(batch)
                    
        # 随机打乱不同数据集的batch顺序
        if self.shuffle:
            np.random.shuffle(batches)
            
        return iter(batches)
    
    def __len__(self) -> int:
        if self.drop_last:
            return sum(size // self.batch_size for size in self.dataset_sizes)
        else:
            return sum((size + self.batch_size - 1) // self.batch_size 
                      for size in self.dataset_sizes)
