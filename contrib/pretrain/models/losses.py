import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from prefusion.registry import MODELS
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss
from mmseg.models.losses.cross_entropy_loss import _expand_onehot_labels
import matplotlib.pyplot as plt


__all__ = ['PretrainSegLoss', 'PretrainFocalLoss']


def pretrain_seg_iou(pred, label, dim=None, class_weight=None, ignore_index=None):
    """
    pred must be sigmoided, pred and label share same shape
    shape in (N, C, H, W) or (N, H, W) or (H, W)
    """
    if label.max() == 0:
        return pretrain_seg_iou(1 - pred, 1 - label)
    valid_mask = label[:, -1, ...].unsqueeze(1).to(torch.float32)
    label = label[:, :-1, ...]# .permute(0, 3, 1, 2)
    pred = pred * (1-valid_mask)
    label = label * (1-valid_mask)
    inter = (pred * label).sum(dim=dim) + 1
    union = (pred + label - pred * label).sum(dim=dim) + 1
    return inter / union



@MODELS.register_module()
class PretrainSegLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 loss_weight=1.0,
                 method=None,
                 loss_name='loss_seg',
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
    
        if self.use_sigmoid:
            self.cls_criterion = pretrain_seg_iou
        else:
            raise NotImplementedError
        self._loss_name = loss_name
        self.method = method

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.use_sigmoid:
            cls_score = cls_score.sigmoid()
        # Note: for BCE loss, label < 0 is invalid.
        iou = self.cls_criterion(cls_score, label, ignore_index=ignore_index)
        if self.method == 'log':
            loss = -iou.log()
        else:
            loss = 1 - iou 
        
        return loss * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    

def dual_focal_loss(pred, label, reduction='mean', pos_weight=None, ignore_index=None):
    assert reduction in ['none', 'mean', 'sum']
    label = label.type_as(pred)
    valid_mask = label[:, -1, ...].unsqueeze(1).to(torch.float32)
    label = label[:, :-1, ...] # .permute(0, 3, 1, 2)
    pred = pred * (1-valid_mask)
    label = label * (1-valid_mask)
    
    l1 = torch.abs(label - pred) # * valid_mask # L1 Loss Eq.: |y - sigmoid(logits)|
    bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight) # * valid_mask # bce_w_logits Eq.: log(1 + exp(-pred)) + (1 - label) * pred
    loss = l1 + bce
    
    loss = weight_reduce_loss(loss, None, reduction, avg_factor=None)
    
    return loss

@MODELS.register_module()
class PretrainFocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 method=None,
                 loss_name='loss_focal',
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = dual_focal_loss
        else:
            raise NotImplementedError
        self._loss_name = loss_name
        self.method = method

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.use_sigmoid:
            cls_score = cls_score.sigmoid()
        # Note: for BCE loss, label < 0 is invalid.
        loss = self.cls_criterion(cls_score, label, self.reduction, ignore_index=ignore_index)
        
        return loss * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
