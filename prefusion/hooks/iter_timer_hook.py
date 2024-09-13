import time
from typing import Optional, Sequence, Union

from mmengine.registry import HOOKS
from mmengine.hooks.hook import Hook
from mmengine.hooks import IterTimerHook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class GroupIterTimerHook(IterTimerHook):
    def __init__(self):
        super().__init__()

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict, Sequence]] = None,
                    mode: str = 'train') -> None:
        """Calculating time for an iteration and updating "time"
        ``HistoryBuffer`` of ``runner.message_hub``.

        Args:
            runner (Runner): The runner of the training validation and
                testing process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict or sequence, optional): Outputs from model.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # Update iteration time in `runner.message_hub`.
        message_hub = runner.message_hub
        message_hub.update_scalar(f'{mode}/time', time.time() - self.t)
        self.t = time.time()
        iter_time = message_hub.get_scalar(f'{mode}/time')
        if mode == 'train':
            self.time_sec_tot += iter_time.current()
            # Calculate average iterative time.
            time_sec_avg = self.time_sec_tot / (
                runner.iter - self.start_iter + 1)
            # Calculate eta.
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            runner.message_hub.update_info('eta', eta_sec)
        else:
            if mode == 'val':
                cur_dataloader = runner.val_dataloader
            else:
                cur_dataloader = runner.test_dataloader

            self.time_sec_test_val += iter_time.current()
            time_sec_avg = self.time_sec_test_val / (batch_idx + 1)
            eta_sec = time_sec_avg * (len(cur_dataloader) * cur_dataloader.dataset.group_size - batch_idx - 1)
            runner.message_hub.update_info('eta', eta_sec)