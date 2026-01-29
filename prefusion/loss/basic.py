import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from prefusion.registry import MODELS
import os
__all__ = ['SegIouLoss', 'dual_focal_loss', 'DualFocalLoss',
'WeightedFocalLossWithSoftmax', 'calculate_l2_loss', 'WeightedFocalLossWithSigmoid']


def print_channel_minmax(mask: torch.Tensor):
    """
    mask: Tensor [B, C, H, W]
    打印每个样本的每个通道的最小值和最大值
    """
    B, C, H, W = mask.shape
    for b in range(B):
        # 展平成 [C, H*W]，方便对每个通道做 min/max
        mins = mask[b].view(C, -1).min(dim=1).values
        maxs = mask[b].view(C, -1).max(dim=1).values
        print(f"Sample {b}:")
        for c in range(C):
            print(f"  Channel {c+1}: min={mins[c].item()}, max={maxs[c].item()}")
        print()

from typing import Optional, Tuple, Union

def seg_iou_per_channel(
    pred: torch.Tensor,          # [B, C, H, W]，已是 sigmoid 概率
    label: torch.Tensor,         # [B, C, H, W]，0/1
    mask: Optional[torch.Tensor] = None,  # 可为 [B,1,H,W] 或 [B,C,H,W] 或可广播
    spatial_dims: Union[Tuple[int, int], Tuple[int, int, int]] = (0, 2, 3),
    eps: float = 1e-6,
    index_info_str: str = None
):
    """
    返回:
      iou:  [B, C]  每个样本、每个通道的 IoU（前景或背景，依 has_pos 决定）
      has_pos: [B, C]  该样本×通道在 mask 区域是否有正样本
    """
    assert pred.shape == label.shape and pred.dim() == 4, "pred/label 必须是 [B,C,H,W]"
    B, C, H, W = pred.shape

    if mask is None:
        mask = torch.ones_like(label, dtype=pred.dtype, device=pred.device)
    else:
        # 规范到与 pred 相同的 device/dtype，并二值化
        mask = (mask > 0).to(device=pred.device, dtype=pred.dtype)
        # 让 mask 可广播到 [B,C,H,W]；若不能直接 expand，则显式广播
        if mask.shape != label.shape:
            try:
                mask = mask.expand_as(label)
            except RuntimeError:
                mask = mask * torch.ones_like(label, dtype=pred.dtype, device=pred.device)

    # 只保留 mask==1 的区域
    pred_m  = pred  * mask
    label_m = label * mask

    # 每个样本×通道是否在 mask 区域有正样本（只在空间维聚合，不跨通道/批次）
    has_pos = (label_m.sum(dim=spatial_dims) > 0)   # [B, C] 的布尔张量

    # 前景 IoU
    inter_fg = (pred_m * label_m).sum(dim=spatial_dims)                      # [B, C]
    union_fg = (pred_m + label_m - pred_m * label_m).sum(dim=spatial_dims)  # [B, C]

    # 背景 IoU（仅在 mask 区域）
    pred_b  = (1 - pred)  * mask
    label_b = (1 - label) * mask
    inter_bg = (pred_b * label_b).sum(dim=spatial_dims)                      # [B, C]
    union_bg = (pred_b + label_b - pred_b * label_b).sum(dim=spatial_dims)  # [B, C]

    # 逐样本×通道选择前景或背景公式
    inter = torch.where(has_pos, inter_fg, inter_bg)
    union = torch.where(has_pos, union_fg, union_bg)

    # if pred.shape[1] == 13:
    #     # has_pos 可能是 bool Tensor，可以转成 Python bool
    #     print(
    #         has_pos[3].detach().cpu().numpy(),   # 如果 has_pos 是向量/矩阵，用 numpy 展开
    #         inter[3].detach().cpu().item(),   # 单个数值
    #         union[3].detach().cpu().item(),
    #         label_m.sum(dim=spatial_dims)[3].detach().cpu().item(),
    #         index_info_str
    #     )
    #     s = str(index_info_str[0])  
    #     clean = s.split("(")[0].strip()   # 去掉括号后的部分
    #     save_path = "./seg_label/" + clean + ".png"
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     print(save_path)
    #     cv2.imwrite( save_path,label_m[0,3,:,:].detach().cpu().numpy())

    iou = (inter + eps) / (union + eps)   # [B, C]
    return iou, has_pos


class SegIouLoss(nn.Module):
    def __init__(self, method='log', pred_logits=True, reduction_dim=None, eps=1e-6):
        super().__init__()
        assert method in ['log', 'linear']
        self.method = method
        self.pred_logits = pred_logits
        self.reduction_dim = reduction_dim
        self.eps = eps

    def forward(self, pred, label, mask=None,index_info_str=None):
        assert pred.shape == label.shape

        # print(pred.shape,label.shape)
        if self.pred_logits:
            pred = pred.sigmoid()

        # # 若未设置 reduction_dim，默认沿 (N,H,W) 维聚合，返回 [C]
        # spatial_dims = self.reduction_dim if self.reduction_dim is not None else (0, 2, 3)
        iou,has_pos = seg_iou_per_channel(pred, label, mask=mask, spatial_dims=self.reduction_dim, eps=self.eps,index_info_str=index_info_str)

        if self.method == 'log': 
            loss = -torch.log(iou + self.eps)
        else:

            loss = 1 - iou

        # if pred.shape[1]==13:
        #     print(loss)
        return loss



# def dual_focal_loss(pred, label, reduction='mean', pos_weight=None):
#     assert reduction in ['none', 'mean', 'sum']
#     pred_sigmoid = pred.sigmoid()
#     label = label.type_as(pred)
#     l1 = torch.abs(label - pred_sigmoid) # L1 Loss Eq.: |y - sigmoid(logits)|
#     bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight) # bce_w_logits Eq.: log(1 + exp(-pred)) + (1 - label) * pred
#     loss = l1 + bce
#     if reduction == 'none':
#         return loss
#     elif reduction == 'sum':
#         return loss.sum()
#     else:
#         return loss.mean()

def dual_focal_loss(pred, label, mask=None, reduction='mean', pos_weight=None):
    """
    Args:
        pred: [B, C, H, W] or [B, ...] raw logits
        label: same shape, 0/1
        mask: same shape or [B, 1, H, W] (可广播)
        reduction: 'none' | 'mean' | 'sum'
        pos_weight: optional tensor for BCE
    """
    # print(reduction)
    assert reduction in ['none', 'mean', 'sum']

    pred_sigmoid = pred.sigmoid()
    label = label.type_as(pred)
    l1 = torch.abs(label - pred_sigmoid)
    bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight)
    loss = l1 + bce

    if mask is not None:
        # 广播到同 shape
        if mask.shape != loss.shape:
            mask = mask.expand_as(loss)
        loss = loss * mask

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:  # mean
        if mask is not None:
            # 只对有效区域取平均
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()


def dual_focal_loss_mask(pred, label, mask=None, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None, reduction_dim=None, eps=1e-6):
    """
    pred: logits, 任意 shape
    label: 同 shape，0/1 或 [0,1] 浮点
    mask: 0/1，形状可广播到 label（如 N,1,H,W 或与 label 同形）
    reduction: 'none' | 'sum' | 'mean'；'mean' 仅对 mask==1 的位置求平均
    pos_weight: 同 PyTorch BCEWithLogits 的 pos_weight，形状可广播到 label
    """
    assert reduction in ['none', 'mean', 'sum']
    # pred = pred.to(torch.float32)
    label = label.to(device=pred.device, dtype=pred.dtype)

    # 处理 mask（只计入 mask==1 的位置）
    if mask is None:
        mask = torch.ones_like(label, dtype=pred.dtype, device=pred.device)
    else:
        # 强制 0/1，并放到与 pred 相同的 device/dtype
        mask = (mask > 0).to(device=pred.device, dtype=pred.dtype)

    # 确保 pos_weight 与 pred 同 device/dtype，避免 BCE 报错
    if pos_weight is not None and isinstance(pos_weight, torch.Tensor):
        pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype)

    pred_sigmoid = torch.sigmoid(pred)
    l1  = torch.abs(label - pred_sigmoid)  # |y - σ(logits)|

    # p_t
    pt = label * pred_sigmoid + (1 - label) * (1 - pred_sigmoid)
    # focal term
    focal_weight = (alpha * label + (1 - alpha) * (1 - label)) * ((1 - pt) ** gamma)


    # 元素级 BCE（不在这里做内部 mean/sum，方便我们用自定义 mask 归一化）
    bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none', pos_weight=pos_weight)

    # 组合并应用 mask
    loss_elem = (l1 + focal_weight*bce) * mask

    if reduction == 'none':
        return loss_elem
    elif reduction == 'sum':
        return loss_elem.sum()
    else:
        if reduction_dim is not None:
            numer = loss_elem.sum(dim=reduction_dim)
            denom = mask.sum(dim=reduction_dim).clamp_min(eps)
            return numer / denom
        else:
            denom = mask.sum()
            if denom.item() == 0:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            return loss_elem.sum() / (denom + eps)




@MODELS.register_module()
class DualFocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, label, pos_weight=None):
        # print(self.reduction)
        return dual_focal_loss(pred, label, None, self.reduction, pos_weight)



@MODELS.register_module()
class WeightedFocalLossWithSoftmax(nn.Module):
    """Focal loss using a sigmoid activation for binary classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        p_t = probs[:, 0, :, :].reshape(-1) * targets.reshape(-1)
        p_t += (1 - probs[:, 0, :, :].reshape(-1)) * (1 - targets.reshape(-1)) # p_t dtype: torch.float16
        focal_loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-4) # 1e-4 # 因为这里采用了autocast
        return focal_loss.mean()

class WeightedFocalLossWithSigmoid(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, index_total_class=0):
        super(WeightedFocalLossWithSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.index_total_class = index_total_class

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)  # 每个类别的独立概率
        num_classes = probs.size(1)
        # losses = [0.0] * num_classes  # 初始化每个类别的损失列表
        losses = [torch.tensor(0.0).cuda() for _ in range(num_classes)]  # 初始化每个类别的损失列表

        # # 计算总类别的损失（保留所有 targets > 0 的位置）
        # mask_total_class = (targets > 0).float()
        # if mask_total_class.sum() > 0:
        #     p_total = probs[:, self.index_total_class, :, :].reshape(-1) * mask_total_class.reshape(-1) + \
        #               (1 - probs[:, self.index_total_class, :, :].reshape(-1)) * (1 - mask_total_class.reshape(-1))
        #     losses[self.index_total_class] = (-self.alpha * (1 - p_total) ** self.gamma * torch.log(p_total)).mean()

        # 计算每个子类别的损失
        for i in range(0, num_classes):  # 遍历子类别
            mask_subclass = (targets == i+1 ).float()  # 针对每个子类别的掩码
            # print("mask_subclass shape:", mask_subclass.shape)
            # print("probs shape:", probs[:, i, :, :].shape)
            p_subclass = probs[:, i, :, :].reshape(-1) * mask_subclass.reshape(-1) + \
                            (1 - probs[:, i, :, :].reshape(-1)) * (1 - mask_subclass.reshape(-1))
            # eps = 1e-7 if p_subclass.dtype == torch.float32 else 1e-4
            eps = 1e-4  # dtype为torch.float16
            p_subclass = torch.clamp(p_subclass, min=eps, max=1.0 - eps)
            losses[i] =  (-self.alpha * (1 - p_subclass) ** self.gamma * torch.log(p_subclass)).mean()

        return losses  # 返回每个类别的独立损失列表
    
    
def calculate_l2_loss(loc_gt, loc_map, cls_gt):
    """
    loc_*: (B, 6, H, W) with channels:
      [xlow, xhigh, ylow, yhigh, sin2θ, cos2θ]
    只在 cls_gt>0 的 cell 上计算：
      - 坐标4通道：sigmoid 到 [0,1] 后逐元素 MSE
      - 角度2通道：单位化后做余弦距离 1 - dot
    """
    B, C, H, W = loc_gt.shape
    assert C == 6 and loc_map.shape[1] == 6, "expect 6-channel loc maps"

    # Reshape for easier indexing (保持你的逻辑)
    loc_gt  = loc_gt.permute(0, 2, 3, 1).reshape(-1, 6)
    loc_map = loc_map.permute(0, 2, 3, 1).reshape(-1, 6)
    cls_gt  = cls_gt.permute(0, 2, 3, 1).reshape(-1)

    # 只在有线段的格子计算
    total_mask = (cls_gt > 0)
    if total_mask.sum() == 0:
        return (loc_map.sum() * 0.0)

    valid_gt   = loc_gt[total_mask]
    valid_pred = loc_map[total_mask]

    # ---- 坐标 4 通道：sigmoid 后逐元素 MSE ----
    gt_xy   = valid_gt[:, :4]
    pred_xy = torch.sigmoid(valid_pred[:, :4])   # 或 F.sigmoid(valid_pred[:, :4])
    loss_xy = F.mse_loss(pred_xy, gt_xy)

    # ---- 角度 2 通道：单位化 → 1 - dot ----
    gt_ang   = valid_gt[:, 4:6]
    pred_ang = valid_pred[:, 4:6]
    gt_ang   = F.normalize(gt_ang,   dim=1, eps=1e-4)
    pred_ang = F.normalize(pred_ang, dim=1, eps=1e-4)
    cos_sim  = (gt_ang * pred_ang).sum(dim=1).clamp(-1.0, 1.0)
    loss_ang = (1.0 - cos_sim).mean()  # [0, 2]
    w_xy = 1.4
    w_ang = 0.6
    loss = {
        "loss_xy": w_xy * loss_xy,
        "loss_ang": w_ang * loss_ang,
        "loss"  : w_xy *loss_xy + w_ang * loss_ang
    }
    # # 总损失（如需权重可自行加系数）
    # loss = loss_xy + w_ang * loss_ang

    return loss