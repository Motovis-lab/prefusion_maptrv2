import torch
import torch.nn as nn
import torch.nn.functional as F

from prefusion.registry import MODELS

__all__ = ['seg_iou', 'SegIouLoss', 'dual_focal_loss', 'DualFocalLoss']

def seg_iou(pred, label, dim=None):
    """
    pred must be sigmoided, pred and label share same shape
    shape in (N, C, H, W) or (N, H, W) or (H, W)
    """
    if label.max() == 0:
        return seg_iou(1 - pred, 1 - label, dim=dim)
    inter = (pred * label).sum(dim=dim) + 1
    union = (pred + label - pred * label).sum(dim=dim) + 1
    return inter / union


@MODELS.register_module()
class SegIouLoss(nn.Module):
    def __init__(self, method='log', pred_logits=True, reduction_dim=None):
        super().__init__()
        assert method in ['log', 'linear']
        self.method = method
        self.pred_logits = pred_logits
        self.reduction_dim = reduction_dim
   
    def forward(self, pred, label, mask=None):
        assert pred.shape == label.shape

        if self.pred_logits:
            pred = pred.sigmoid()
        if mask is None:
            mask = torch.ones_like(label)

        iou = seg_iou(pred * mask, label * mask, dim=self.reduction_dim)

        if self.method == 'log':
            loss = -iou.log()
        else:
            loss = 1 - iou        

        return loss


def dual_focal_loss(pred, label, reduction='mean', pos_weight=None):
    assert reduction in ['none', 'mean', 'sum']
    pred_sigmoid = pred.sigmoid()
    label = label.type_as(pred)
    l1 = torch.abs(label - pred_sigmoid) # L1 Loss Eq.: |y - sigmoid(logits)|
    bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight) # bce_w_logits Eq.: log(1 + exp(-pred)) + (1 - label) * pred
    loss = l1 + bce
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


@MODELS.register_module()
class DualFocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, label, pos_weight=None):
        return dual_focal_loss(pred, label, self.reduction, pos_weight)