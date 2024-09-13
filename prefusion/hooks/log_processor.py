import copy
import datetime
import re
from collections import OrderedDict
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch

from mmengine.registry import LOG_PROCESSORS
from mmengine.runner.log_processor import LogProcessor


@LOG_PROCESSORS.register_module()
class MV_LogProcessor(LogProcessor):
    def __init__(self, window_size=10, by_epoch=True, custom_cfg: List[dict] | None = None, num_digits: int = 4, log_with_hierarchy: bool = False, mean_pattern=r'.*(loss|time|data_time|grad_norm).*'):
        super().__init__(window_size, by_epoch, custom_cfg, num_digits, log_with_hierarchy, mean_pattern)

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
        