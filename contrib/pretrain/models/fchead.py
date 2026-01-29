from mmseg.models import FCNHead
from prefusion.registry import MODELS
from .accuracy import accuracy
from mmseg.models.utils import resize
from torch import nn, Tensor
from mmseg.utils import ConfigType, SampleList
import torch

__all__ = ['PrefusionFCNHead']

@MODELS.register_module()
class PrefusionFCNHead(FCNHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kwargs['loss_decode'], dict):
            self.loss_decode = MODELS.build(kwargs['loss_decode'])
        elif isinstance(kwargs['loss_decode'], (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in kwargs['loss_decode']:
                self.loss_decode.append(MODELS.build(loss))

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        return super().forward(inputs)
    
    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # loss['acc_seg'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss