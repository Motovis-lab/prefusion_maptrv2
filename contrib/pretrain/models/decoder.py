from mmengine.model import BaseModel, BaseModule
from torch.nn.modules import Module
from prefusion.registry import MODELS
import torch.nn as nn
from contrib.fastray_planar.modules import EltwiseAdd, ConvBN, OSABlock
import torch
from mmseg.models.utils import resize
import torch.nn.functional as F
import random
from prefusion.registry import MODELS
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss

__all__ = ['SegDecoder']

@MODELS.register_module()
class SegDecoder(BaseModule):
    def __init__(self, num_classes=26,loss_decode=None, init_cfg: dict | None = None):
        super().__init__(init_cfg)
        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        self.seg_linear = ConvBN(128, 128, kernel_size=1, padding=0)
        self.seg_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.seg_osa = OSABlock(128, 128, 128, stride=1, repeat=2, has_bn=False)
        self.seg = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=4, padding=0)
        # self.seg_linear.conv.weight.register_hook(self.print_grad)
    
    def forward(self, x):
        x = self.seg_osa(self.seg_up(self.seg_linear(x)))
        seg = self.seg(x)
        # import pdb
        # pdb.set_trace()
        return seg
    
    def loss(self, seg_logits, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # seg_logits = resize(
        #     input=seg_logits,
        #     size=seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=False)
       
        seg_label = seg_label.squeeze(1)
        # cls_score = seg_logits.sigmoid()

        loss_seg = self.pretrain_seg_iou(seg_logits, seg_label)
        loss_ce = self.dual_focal_loss(seg_logits, seg_label)

        loss = dict(loss=5*(1-loss_seg) + 20*loss_ce, loss_ce=20*loss_ce, loss_seg=5*(1-loss_seg))

        return loss

    def _stack_batch_gt(self, batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def print_grad(self, grad):
        print(f'Gradient of linear layer weights: {grad.min(), grad.max(), grad.mean()}')

    def dual_focal_loss(self, pred, label, reduction='mean', pos_weight=None, ignore_index=None):
        assert reduction in ['none', 'mean', 'sum']
        label = label.type_as(pred)
        valid_mask = label[:, -1, ...].unsqueeze(1).to(torch.float32)
        label = label[:, :-1, ...] # .permute(0, 3, 1, 2)
        pred_sigmoid = pred.sigmoid()
        pred_sigmoid = pred_sigmoid * (1-valid_mask)
        label = label * (1-valid_mask)
        
        l1 = torch.abs(label - pred_sigmoid) # * valid_mask # L1 Loss Eq.: |y - sigmoid(logits)|
        
        bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight) * (1-valid_mask) # bce_w_logits Eq.: log(1 + exp(-pred)) + (1 - label) * pred
        loss = l1 + bce
        
        loss = weight_reduce_loss(loss, None, reduction, avg_factor=(1-valid_mask).sum())
        
        return loss
    
    def pretrain_seg_iou(self, pred, label, dim=None, class_weight=None, ignore_index=None):
        """
        pred must be sigmoided, pred and label share same shape
        shape in (N, C, H, W) or (N, H, W) or (H, W)
        """
        if label.max() == 0:
            return self.pretrain_seg_iou(1 - pred, 1 - label)
        valid_mask = label[:, -1, ...].unsqueeze(1).to(torch.float32)
        label = label[:, :-1, ...]# .permute(0, 3, 1, 2)
        pred = pred.sigmoid()
        pred = pred * (1-valid_mask)
        label = label * (1-valid_mask)
        inter = (pred * label).sum(dim=dim) + 1
        union = (pred + label - pred * label).sum(dim=dim) + 1
        return inter / union
    
    def init_weights(self):
        return super().init_weights()