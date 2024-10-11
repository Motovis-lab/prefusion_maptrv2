import functools

import torch
import torch.nn as nn

from prefusion.loss.basic import dual_focal_loss, SegIouLoss


class PlanarBbox3DLoss(nn.Module):
    def __init__(self, *args, class_weights=None, loss_weights=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_weights = {"seg": 1.0, "cen": 1.0, "reg": 1.0}
        self.loss_weights.update(loss_weights or {})
        self.class_weights = class_weights
        self.reduction_dim = (0, 2, 3)
        self.seg_iou_loss = SegIouLoss(pred_logits=True, reduction_dim=self.reduction_dim)
    
    def forward(self, pred, label):
        assert set(pred.keys()) == set(label.keys()) == {"seg", "cen", "reg"}
        loss = dict(
            **self._seg_loss(pred, label)
        )
        loss["total_planar_bbox_3d_loss"] = 0.0
        loss["seg"] = (
            
            + self.seg_iou_loss(pred["seg"], label["seg"])
        )
        loss["cen"] = nn.L1Loss(pred["cen"], label["cen"], reduction="mean")
        loss["reg"] = nn.L1Loss(pred["reg"], label["reg"], reduction="mean")
        return loss

    def _seg_loss(self, pred, label, dual_focal_loss_weight=1.0, iou_loss_weight=1.0):
        dual_loss = dual_focal_loss(pred["seg"], label["seg"], reduction="none").mean(dim=self.reduction_dim)
        iou_loss = self.seg_iou_loss(pred["seg"], label["seg"])
        _class_weights = torch.ones(pred.shape[1]) if self.class_weights is None else torch.tensor(self.class_weights)
        loss_dict = {}
        seg_iou_loss_by_channel = {f"seg_iou_loss_{i}": l * _class_weights[i] for i, l in enumerate(iou_loss)}
        seg_dual_focal_loss_by_channel = {f"seg_dual_focal_loss_{i}": l * _class_weights[i] for i, l in enumerate(dual_loss)}
        loss_dict.update(seg_iou_loss_by_channel)
        loss_dict.update(seg_dual_focal_loss_by_channel)
        loss_dict["seg_iou_loss"]  = sum(seg_iou_loss_by_channel.values()) / _class_weights.sum() # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict["seg_dual_focal_loss"] = sum(seg_dual_focal_loss_by_channel.values()) / _class_weights.sum()
        loss_dict["seg_loss"] = loss_dict["seg_dual_focal_loss"] + loss_dict["seg_iou_loss"]
        return loss_dict


    def _cen_loss(self, pred, label):
        pass

    def _reg_loss(self, pred, label):
        pass
