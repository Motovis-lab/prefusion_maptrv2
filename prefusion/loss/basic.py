import torch
import torch.nn as nn
import torch.nn.functional as F


def seg_iou(pred, label, dim=None):
    """
    pred must be sigmoided, pred and label share same shape
    shape in (N, C, H, W) or (N, H, W) or (H, W)
    """
    if label.max() == 0:
        return seg_iou(1 - pred, 1 - label)
    inter = (pred * label).sum(dim=dim) + 1
    union = (pred + label - pred * label).sum(dim=dim) + 1
    return inter / union


class SegIouLoss(nn.Module):
    def __init__(self, method='log', pred_logits=True, channel_weights=None):
        super().__init__()
        assert method in ['log', 'linear']
        self.method = method
        self.pred_logits = pred_logits
        self.channel_weights = channel_weights

    @staticmethod
    def _validate_inputs(pred, label, channel_weights):
        assert pred.shape == label.shape
        assert 2 <= pred.ndim <= 4
        if channel_weights is not None:
            assert len(channel_weights) == pred.shape[1]
   
    @staticmethod
    def _ensure_shape_nchw(tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        return tensor

    def forward(self, pred, label, mask=None):
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)
        channel_weights = torch.ones(pred.shape[1]) if self.channel_weights is None else torch.tensor(self.channel_weights)
        self._validate_inputs(pred, label, channel_weights)

        if self.pred_logits:
            pred = pred.sigmoid()
        if mask is None:
            mask = torch.ones_like(label)

        iou = seg_iou(pred * mask, label * mask, dim=(0, 2, 3))

        if self.method == 'log':
            loss = -iou.log()
        else:
            loss = 1 - iou
        
        if channel_weights is not None:
            loss = (loss * channel_weights).sum() / channel_weights.sum()

        return loss


def dual_focal_loss(pred, label, reduction='mean', pos_weight=None):
    assert reduction in ['none', 'mean', 'sum']
    pred_sigmoid = pred.sigmoid()
    label = label.type_as(pred)
    l1 = torch.abs(label - pred_sigmoid)
    loss = l1 + F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight)
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()

def l1_ohem_dual_focal_loss(pred, label, mask=None, reduction='weighted', pos_weight=None):
    assert reduction in ['none', 'mean', 'sum', 'weighted']
    pred_sigmoid = pred.sigmoid()
    label = label.type_as(pred)
    if mask is None:
        mask = torch.ones_like(label)
    l1 = torch.abs(label - pred_sigmoid) * mask
    loss = l1 * (l1 + F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight) * mask)
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'weighted':
        return loss.sum() / (l1.sum() + 1e-5)
    else:
        return loss.sum() / (mask.sum() + 1e-5)

def _random_mask(tensor, percent):
    assert percent > 0
    return (torch.rand_like(tensor) < percent).float()

def auto_balanced_dual_focal_loss(pred, label, neg_pos_ratio=4, least_neg_pct=0.05, reduction='mean', mask_minv=0, pos_weight=None):
    assert reduction in ['none', 'mean', 'sum']
    label = label.type_as(pred)
    mask_pos = (label > 0).float()
    rand_pct = mask_pos.sum() / mask_pos.nelement()
    neg_pct = (rand_pct * (neg_pos_ratio + 1)).clamp(least_neg_pct)
    mask = mask_minv + ((_random_mask(label, neg_pct) + mask_pos) > 0).float()
    mask /= mask.mean()
    loss = dual_focal_loss(pred, label, reduction='none', pos_weight=pos_weight) * mask

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class DualFocalLoss(nn.Module):

    def __init__(self, reduction='mean', balanced=False, l1_weighted=False, mask_minv=0):
        super().__init__()
        self.reduction = reduction
        self.balanced = balanced
        self.mask_minv = mask_minv
        self.l1_weighted = l1_weighted
    
    def forward(self, pred, label, mask=None, pos_weight=None):
        if self.l1_weighted:
            return l1_ohem_dual_focal_loss(pred, label, mask, reduction='weighted', pos_weight=pos_weight)
        if self.balanced:
            return auto_balanced_dual_focal_loss(pred, label, reduction=self.reduction, pos_weight=pos_weight, mask_minv=self.mask_minv)
        else:
            return dual_focal_loss(pred, label, self.reduction, pos_weight)
