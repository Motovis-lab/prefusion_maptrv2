# Written by rlan, AI-RagTag. All rights reserved.
import abc
import warnings
import time
import multiprocessing
import functools
from typing import Any, Callable

import numpy as np
import torch.distributed as dist
from mmengine.logging import print_log
from torch.distributed import is_initialized as is_torch_dist_initialized
from prefusion.registry import DATASET_TOOLS

from .utils import divide


_print_log = functools.partial(print_log, logger='current')


class EndOfAllSubEpochs(Exception):
    pass

def _method_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f'Please implemented method [{type(self).__name__}] in the subclasses.')


class SharedKVStore(metaclass=abc.ABCMeta):
    set: Callable[..., Any] = _method_unimplemented
    get: Callable[..., Any] = _method_unimplemented
    has_key: Callable[..., Any] = _method_unimplemented
    clear: Callable[..., Any] = _method_unimplemented
    todict: Callable[..., Any] = _method_unimplemented
    __len__: Callable[..., Any] = _method_unimplemented


class MultiprocessingStore(SharedKVStore):
    def __init__(self):
        self.dict = multiprocessing.Manager().dict()

    def __len__(self) -> int:
        return len(self.dict)

    def set(self, key, value):
        self.dict[key] = value

    def get(self, key):
        return self.dict.get(key)

    def has_key(self, key) -> bool:
        return key in self.dict

    def clear(self):
        self.dict.clear()

    def todict(self) -> dict:
        return dict(self.dict)


class RedisStore(SharedKVStore):
    def __init__(self, host="localhost", port=6379, db=8):
        import redis
        self.client = redis.Redis(host=host, port=port, db=db)
        self.client.flushdb()

    def __len__(self) -> int:
        return self.client.dbsize()

    def set(self, key, value):
        self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)
    
    def has_key(self, key: str) -> bool:
        assert isinstance(key, str), "RedisStore doesn't support non string keys."
        return self.client.get(key) is not None

    def clear(self):
        is_master = dist.get_rank() == 0
        if is_master:
            self.client.flushdb()
        else:
            time.sleep(2)

    def todict(self) -> dict:
        if self.client.dbsize() == 0:
            return {}
        
        d = {}
        for key in self.client.scan_iter():
            d[key.decode("utf-8")] = self.client.get(key).decode("utf-8")
        return d


class IndicesVisittingLogger:
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=8):
        if is_torch_dist_initialized():
            self.kv_store: SharedKVStore = RedisStore(host=redis_host, port=redis_port, db=redis_db)
        else:
            self.kv_store: SharedKVStore = MultiprocessingStore()

    def __len__(self) -> int:
        return len(self.kv_store)

    def visit(self, idx: int):
        self.kv_store.set(idx, 1)

    def clear(self):
        self.kv_store.clear()

    def todict(self, ensure_int_key=True) -> dict:
        d = self.kv_store.todict()
        if ensure_int_key:
            d = {int(k): v for k, v in d.items()}
        return d


@DATASET_TOOLS.register_module()
class SubEpochManager:
    def __init__(
        self,
        batch_size: int,
        num_group_batches_per_subepoch: int,
        drop_last_group_batch: bool = False,
        drop_last_subepoch: bool = False,
        verbose: bool = True,
        debug_mode: bool = False,
    ):
        """SubEpochManager stores and manages the state of sub epochs. This feature is mainly used in the case
        that one wants to train an original epoch as a series of sub epochs.

        Parameters
        ----------
        batch_size : int
            batch_size
        num_group_batches_per_subepoch : int
            number of groups in one sub epoch
        drop_last_group_batch : bool, optional
            whether to drop the last group batch when splitting subepochs, by default False
        drop_last_subepoch : bool, optional
            Whether to drop the last subepoch if the subepoch's length doesn't meet `num_group_batches_per_subepoch`.
            If False, the last subepoch will be padded with index from the previous subepochs to make its length equal to `num_group_batches_per_subepoch`.
            In other words, no matter it is True or False, the last subepoch will always have the same length as other subepochs.
            So, drop_last_subepoch's value only affects self.num_subepochs.
            By default False.
        debug_mode: bool, optional
            if debug_mode=True, will store the translated idx to visited. By default False.
        """
        self.batch_size = batch_size
        self.num_group_batches_per_subepoch = num_group_batches_per_subepoch
        self.drop_last_group_batch = drop_last_group_batch
        self.drop_last_subepoch = drop_last_subepoch
        self.verbose = verbose
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.visited = IndicesVisittingLogger()

    def get_actual_num_group_batches_in_cur_subepoch(self) -> int:
        """It's different from self.num_group_batches_per_subepoch when considering the last subepoch.
        For the last subepoch, it calculates the actual number of group batches in the last subepoch (i.e. no padding).
        """
        if self.cur_subepoch_idx < self.num_subepochs - 1:
            return self.num_group_batches_per_subepoch
        dropped = self.num_total_group_batches % self.num_group_batches_per_subepoch if self.drop_last_subepoch else 0
        num_used_group_batches = self.num_group_batches_per_subepoch * (self.num_subepochs - 1)
        num_group_batches_in_the_last_subepoch = self.num_total_group_batches - num_used_group_batches - dropped
        return num_group_batches_in_the_last_subepoch

    def translate_index(self, idx: int) -> int:
        """Translate the index of a group batch in the current subepoch to the index of the group batch in the whole dataset.

        Parameters
        ----------
        idx : int
            the index of a group batch in the current subepoch

        Returns
        -------
        int
            the index of the group batch in the whole dataset

        Raises
        ------
        IndexError
        """
        if idx >= self.num_group_batches_per_subepoch:
            raise IndexError
        
        if idx < self.get_actual_num_group_batches_in_cur_subepoch():
            translated_index = idx + self.cur_subepoch_idx * self.num_group_batches_per_subepoch
        else:
            prev_subepoch_idx = idx % self.cur_subepoch_idx # prev_subepoch_idx is entailed to be less than self.cur_subepoch_idx
            translated_index = idx + prev_subepoch_idx * self.num_group_batches_per_subepoch

        # from mmengine.dist import get_dist_info
        # from loguru import logger
        # rank, _ = get_dist_info()
        # time.sleep(rank * 2 + 0.1)
        if self.debug_mode:
            self.visited.visit(translated_index)
        # logger.debug(f"[rank{rank}] visited={self.visited.todict().keys()}")

        return translated_index

    def to_next_sub_epoch(self) -> None:
        if self.verbose:
            _print_log(f"cur_subepoch_idx is {self.cur_subepoch_idx}, moving to the next subepoch.")
        self.cur_subepoch_idx += 1
        if self.cur_subepoch_idx == self.num_subepochs: # finished all the subepochs
            if self.verbose:
                _print_log(f"Reached the end of all subepochs.")
            raise EndOfAllSubEpochs

    def reset(self, num_total_groups: int) -> None:
        """Resets the state of the SubEpochManager

        Parameters
        ----------
        num_total_groups : int
            total number of groups that generated by GroupSampler.
        """
        if self.debug_mode:
            self._check_visited_indices()
            self.visited.clear()

        self.num_total_groups = num_total_groups
        self.num_total_group_batches = divide(self.num_total_groups, self.batch_size, drop_last=self.drop_last_group_batch)
        assert self.num_group_batches_per_subepoch <= self.num_total_group_batches, "num_group_batches_per_subepoch should be no larger than num_total_group_batches"
        self.num_subepochs = divide(self.num_total_group_batches, self.num_group_batches_per_subepoch, drop_last=self.drop_last_subepoch)
        self.cur_subepoch_idx = 0
        if self.verbose:
            _print_log(f"Reset SubEpochManager state to {self}")

    init: Callable = reset

    def _check_visited_indices(self):
        if len(self.visited) == 0:
            return

        indices_not_visited = set(range(self._get_num_group_batches_available_to_visit())) - set(self.visited.todict().keys())
        if indices_not_visited:
            warnings.warn(f"Some group batches are not visited! (group_batch_index: {indices_not_visited})", UserWarning)

    def _get_num_group_batches_available_to_visit(self) -> int:
        return int(np.ceil(self.num_total_groups / self.batch_size))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_total_group_batches={self.num_total_group_batches} "
            f"num_subepochs={self.num_subepochs} "
            f"cur_subepoch_idx={self.cur_subepoch_idx}"
            f")"
        )
