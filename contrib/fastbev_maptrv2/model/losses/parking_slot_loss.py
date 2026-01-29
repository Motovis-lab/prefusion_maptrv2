import torch
import torch.nn as nn

class ParkingSlotLoss(nn.Module):
    """
    最小可用 loss：
      - 点坐标 L1（按 gt_valid_mask 只对有效点计算）
      - 实例分类 CE
    """
    def __init__(self, w_pts=1.0, w_cls=1.0, eps=1e-6):
        super().__init__()
        self.w_pts = w_pts
        self.w_cls = w_cls
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, gt_points_xy01, gt_valid_mask, gt_cls_labels):
        """
        pred["points_xy"]: (B,N,P,2) 归一化预测
        pred["cls_logits"]:(B,N,Ck)
        pred["valid_mask"]: (B,Q) —— 训练时以 gt_valid_mask 为准

        gt_points_xy01 : (B,N,P,2) 归一化 GT
        gt_valid_mask  : (B,N,P)   True=真实点
        gt_cls_labels  : (B,N)     实例类别
        """
        B, N, P, _ = gt_points_xy01.shape

        # 点回归
        l1 = torch.abs(pred["points_xy"] - gt_points_xy01)   # (B,N,P,2)
        mask = gt_valid_mask.unsqueeze(-1)                   # (B,N,P,1)
        l1 = (l1 * mask).sum() / (mask.sum() * 2.0 + self.eps)

        # 实例分类
        cls_logit = pred["cls_logits"].reshape(B*N, -1)
        cls_tgt   = gt_cls_labels.reshape(B*N)
        ce = self.ce(cls_logit, cls_tgt)

        loss = self.w_pts * l1 + self.w_cls * ce
        return {"loss_pts": l1, "loss_cls": ce, "loss": loss}
