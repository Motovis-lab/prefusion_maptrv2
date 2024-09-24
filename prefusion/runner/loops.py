
from prefusion.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop, ValLoop, TestLoop
import torch
from prefusion.dataset.dataset import GroupBatchDataset
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.runner.amp import autocast

__all__ = ['GroupBatchTrainLoop', 'GroupValLoop', 'GroupTestLoop', 'GroupInferLoop']

@LOOPS.register_module()
class GroupBatchTrainLoop(EpochBasedTrainLoop):
    """Loop for epoch-based training. Modified for GroupBatchDataset.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(self, runner, dataloader, max_epochs, 
                 val_begin=1, val_interval=1, dynamic_intervals=None):
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)
        assert isinstance(self.dataloader.dataset, GroupBatchDataset), (
            'dataset must be a GroupBatchDataset instance, '
            f'but got {type(self.dataloader.dataset)}.'
        )
        self._max_iters = self._max_epochs * len(self.dataloader) * self.dataloader.dataset.group_size

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if self.val_interval == -1:
                continue
            elif (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model


    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for group_idx, group_batch in enumerate(self.dataloader):
            for frame_idx, frame_batch in enumerate(group_batch):
                idx = group_idx * self.dataloader.dataset.group_size + frame_idx
                self.run_iter(idx, frame_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class GroupValLoop(ValLoop):
    """Loop for validation. For GroupBatchDataset.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self, runner, dataloader, evaluator, fp16=False):
        super().__init__(runner, dataloader, evaluator, fp16=fp16)
        assert isinstance(self.dataloader.dataset, GroupBatchDataset)

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for group_idx, group_batch in enumerate(self.dataloader):
            for frame_idx, frame_batch in enumerate(group_batch):
                idx = group_idx * self.dataloader.dataset.group_size + frame_idx
                self.run_iter(idx, frame_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset) * self.dataloader.dataset.group_size * self.dataloader.dataset.batch_size)
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics


@LOOPS.register_module()
class GroupTestLoop(TestLoop):
    """Loop for test. For GroupBatchDataset.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 test. Defaults to
            False.
    """

    def __init__(self, runner, dataloader, evaluator, fp16=False):
        super().__init__(runner, dataloader, evaluator, fp16=fp16)
        assert isinstance(self.dataloader.dataset, GroupBatchDataset)

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for group_idx, group_batch in enumerate(self.dataloader):
            for frame_idx, frame_batch in enumerate(group_batch):
                idx = group_idx * self.dataloader.dataset.group_size + frame_idx
                self.run_iter(idx, frame_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset) * self.dataloader.dataset.group_size * self.dataloader.dataset.batch_size)
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
        

@LOOPS.register_module()
class GroupInferLoop(ValLoop):

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for group_idx, group_batch in enumerate(self.dataloader):
            for frame_idx, frame_batch in enumerate(group_batch):
                idx = group_idx * self.dataloader.dataset.group_size + frame_idx
                self.run_iter(idx, [frame_batch])
        
        return None
    
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

