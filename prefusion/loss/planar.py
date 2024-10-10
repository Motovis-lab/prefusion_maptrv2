import torch
import torch.nn as nn

from prefusion.loss.basic import dual_focal_loss, seg_iou


class PlanarBbox3DLoss(nn.Module):
    def __init__(self, *args, class_weights=None, loss_weights=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_weights = {"seg": 1.0, "cen": 1.0, "reg": 1.0}
        self.loss_weights.update(loss_weights or {})
        self.class_weights = class_weights
    
    def forward(self, pred, label):
        assert set(pred.keys()) == set(label.keys()) == {"seg", "cen", "reg"}
        loss = {}
        loss["seg"] = (
            dual_focal_loss(pred["seg"], label["seg"], reduction="mean")
            + seg_iou(pred["seg"], label["seg"])
        ) # per class
        loss["cen"] = nn.L1Loss(pred["cen"], label["cen"], reduction="mean")
        loss["reg"] = nn.L1Loss(pred["reg"], label["reg"], reduction="mean")
        return loss

    def _seg_loss():
        pass