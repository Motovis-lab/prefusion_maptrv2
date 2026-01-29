import copy
import datetime
import re
import math
from collections import OrderedDict
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch
from tabulate import tabulate

from prefusion.registry import LOG_PROCESSORS
from mmengine.runner.log_processor import LogProcessor
from mmengine.device import is_cuda_available, is_musa_available


@LOG_PROCESSORS.register_module()
class GroupAwareLogProcessor(LogProcessor):
    def __init__(
        self, 
        window_size=10, 
        by_epoch=True, 
        custom_cfg: List[dict] | None = None, 
        num_digits: int = 4, 
        tabulate: bool = True, 
        tabulate_ncols: int = 3, 
        tabulate_fmt: str = "rounded_outline", 
        log_with_hierarchy: bool = False, 
        mean_pattern=r'.*(loss|time|data_time|grad_norm).*'
    ):
        super().__init__(window_size, by_epoch, custom_cfg, num_digits, log_with_hierarchy, mean_pattern)
        self.tabulate = tabulate
        self.tabulate_ncols = tabulate_ncols
        self.tabulate_fmt = tabulate_fmt

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        dataloader = self._get_cur_loop(runner=runner, mode=mode).dataloader
        return len(dataloader) * dataloader.dataset.group_size
    
    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              self.custom_cfg)
        # log_tag is used to write log information to terminal
        log_tag = self._collect_scalars(parsed_cfg, runner, mode)

        # If `self.log_with_hierarchy` is False, the tag is the same as
        # log_tag. Otherwise, each key in tag starts with prefix `train`,
        # `test` or `val`
        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, mode, True)

        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith('lr'):
                key = self._remove_prefix(key, f'{mode}/')
                log_tag.pop(key)
                lr_str_list.append(f'{key}: '
                                   f'{value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if mode in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, mode)
                if not (isinstance(runner._train_loop, dict)
                        or runner._train_loop is None):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f'[{cur_epoch}]'.rjust(
                        len(str(max_epochs)) + 3, ' ')
                else:
                    cur_epoch_str = f'[{cur_epoch}]'
                tag['epoch'] = cur_epoch
                log_str = (f'Epoch({mode}){cur_epoch_str}'
                           f'[{cur_iter_str}/{dataloader_len}]  ')
            else:
                log_str = (f'Epoch({mode}) '
                           f'[{cur_iter_str}/{dataloader_len}]  ')
        else:
            if mode == 'train':
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = (f'Iter({mode}) '
                           f'[{cur_iter_str}/{runner.max_iters}]  ')
            else:
                dataloader_len = self._get_dataloader_size(runner, mode)
                cur_iter_str = str(batch_idx + 1).rjust(
                    len(str(dataloader_len)))
                log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag['iter'] = 0
        else:
            tag['iter'] = runner.iter + 1
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in log_tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
                        f'data_time: '
                        f'{log_tag["data_time"]:.{self.num_digits}f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda/musa is available,
        # the max memory occupied should be calculated.
        if is_cuda_available() or is_musa_available():
            max_memory = self._get_max_memory(runner)
            log_str += f'memory: {max_memory}  '
            tag['memory'] = max_memory

        # Deal with total loss
        if "loss" in log_tag:
            log_str += f'loss: {log_tag["loss"]:.{self.num_digits}f}  '
            log_tag.pop('loss')

        # Deal with gradnorm (if applicable)
        if "grad_norm" in log_tag:
            log_str += f'grad_norm: {log_tag["grad_norm"]:.{self.num_digits}f}  '
            log_tag.pop('grad_norm')

        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            if self.tabulate:
                log_str += self._generate_tabulate_loss_report(log_tag, mode)
            else:
                log_str += self._generate_naive_loss_report(log_tag, mode)
        return tag, log_str

    def _generate_tabulate_loss_report(self, log_tag: OrderedDict, mode: str):
        losses = np.array(list(f"{k}_#_#_{v:.4f}" for k, v in log_tag.items() if mode != 'val' or k.startswith('val/loss')))
        if len(losses) % self.tabulate_ncols != 0:
            losses = np.pad(losses, ((0, self.tabulate_ncols - len(losses) % self.tabulate_ncols)), constant_values='plh')
        losses = losses.reshape(-1, self.tabulate_ncols, order="F").tolist()
        losses = [[itm for cmb in row if cmb != 'plh' for itm in cmb.split('_#_#_')] for row in losses]
        return "\n" + tabulate(losses, headers=['loss name', 'value'] * self.tabulate_ncols, tablefmt=self.tabulate_fmt)

    def _generate_naive_loss_report(self, log_tag: OrderedDict, mode: str):
        log_items = []
        loss_str = ""
        for name, val in log_tag.items():
            if mode == 'val' and not name.startswith('val/loss'):
                continue
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            log_items.append(f'{name}: {val}\n')
        loss_str += '  '.join(log_items)

        return loss_str
