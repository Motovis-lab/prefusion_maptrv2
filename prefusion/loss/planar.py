import functools

import torch
import torch.nn as nn

from prefusion.loss.basic import dual_focal_loss, SegIouLoss


class PlanarBbox3DLoss(nn.Module):
    def __init__(self, *args, loss_name_prefix="plnrbox3d", class_weights=None, loss_weights=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = self.complete_loss_name()
        self.loss_weights = {"seg": 1.0, "cen": 1.0, "reg": 1.0}
        self.loss_weights.update(loss_weights or {})
        self.class_weights = class_weights
        self.reduction_dim = (0, 2, 3)
        self.iou_loss = SegIouLoss(pred_logits=True, reduction_dim=self.reduction_dim)

    def complete_loss_name(self, sub_item_name: str = ""):
        if sub_item_name:
            return f"{self.loss_name_prefix}_{sub_item_name}_loss"
        return f"{self.loss_name_prefix}_loss"

    @staticmethod
    def _ensure_shape_nchw(tensor_dict):
        for k, v in tensor_dict.items():    
            if v.dim() == 2:
                tensor_dict[k] = v.unsqueeze(0).unsqueeze(0)
            elif v.dim() == 3:
                tensor_dict[k] = v.unsqueeze(1)
        return tensor_dict

    def forward(self, pred, label):
        assert set(pred.keys()) == set(label.keys()) == {"seg", "cen", "reg"}
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        loss = dict(
            **self._seg_loss(pred["seg"], label["seg"]),
            **self._cen_loss(pred["cen"], label["cen"], mask=label["seg"][:, 0:1]),
            **self._reg_loss(pred["reg"], label["reg"], mask=label["seg"][:, 0:1]),
        )
        _L = self.complete_loss_name
        loss[self.total_loss_name] = loss[_L("seg")] + loss[_L("cen")] + loss[_L("reg")]
        return loss

    def _seg_loss(self, pred, label, iou_loss_weight=1.0, dual_focal_loss_weight=1.0):
        dual_loss = dual_focal_loss(pred, label, reduction="none").mean(dim=self.reduction_dim)
        iou_loss = self.iou_loss(pred, label)
        _class_weights = torch.ones(pred.shape[1]) if self.class_weights is None else torch.tensor(self.class_weights)
        loss_dict = {}
        _L = self.complete_loss_name
        seg_iou_loss_by_channel = {_L(f"seg_iou_{i}"): l * _class_weights[i] for i, l in enumerate(iou_loss)}
        seg_dual_focal_loss_by_channel = {_L(f"seg_dual_focal_{i}"): l * _class_weights[i] for i, l in enumerate(dual_loss)}
        loss_dict.update(seg_iou_loss_by_channel)
        loss_dict.update(seg_dual_focal_loss_by_channel)
        loss_dict[_L("seg_iou")]  = iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / _class_weights.sum() # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[_L("seg_dual_focal")] = dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / _class_weights.sum()
        loss_dict[_L("seg")] = (loss_dict[_L("seg_dual_focal")] + loss_dict[_L("seg_iou")]) * self.loss_weights["seg"]
        return loss_dict

    def _cen_loss(self, pred, label, fg_mask, fg_weight=1.0, bg_weight=1.0):
        dual_loss = dual_focal_loss(pred, label, reduction="none").mean()
        fg_dual_loss = (dual_focal_loss(pred, label, reduction="none") * fg_mask).sum() / fg_mask.sum()
        loss_dict = {}
        _L = self.complete_loss_name
        loss_dict[_L("cen_dual_focal")] = bg_weight * dual_loss
        loss_dict[_L("cen_fg_dual_focal")] = fg_weight * fg_dual_loss
        loss_dict[_L("cen")] = (loss_dict[_L("cen_dual_focal")] + loss_dict[_L("cen_fg_dual_focal")]) * self.loss_weights["cen"]
        return loss_dict
    
    def _reg_loss(self, pred, label, mask):
        # only apply to foreground (leverage label['seg'] as mask)
        loss_dict = {}
        return {"reg_loss": nn.L1Loss(pred["reg"], label["reg"], reduction="mean") * self.loss_weights["reg"]}
