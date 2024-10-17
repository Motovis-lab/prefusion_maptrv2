from functools import partial

import torch
import torch.nn as nn

from prefusion.loss.basic import dual_focal_loss, SegIouLoss


def complete_loss_name(loss_name_prefix, sub_item_name: str = ""):
    if sub_item_name:
        return f"{loss_name_prefix}_{sub_item_name}_loss"
    return f"{loss_name_prefix}_loss"


class PlanarBbox3DLoss(nn.Module):
    def __init__(self, *args, loss_name_prefix="plnrbox3d", class_weights=None, loss_weights=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.loss_weights = {"seg": 1.0, "cen": 1.0, "reg": 1.0}
        self.loss_weights.update(loss_weights or {})
        self.class_weights = class_weights
        self.reduction_dim = (0, 2, 3)
        self.iou_loss = SegIouLoss(pred_logits=True, reduction_dim=self.reduction_dim)

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
            **self._cen_loss(pred["cen"], label["cen"], fg_mask=label["seg"][:, 0:1]),
            **self._reg_loss(pred["reg"], label["reg"], fg_mask=label["seg"][:, 0:1]),
        )
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss[self.total_loss_name] = loss[L("seg")] + loss[L("cen")] + loss[L("reg")]
        return loss

    def _seg_loss(self, pred, label, iou_loss_weight=1.0, dual_focal_loss_weight=1.0):
        dual_loss = dual_focal_loss(pred, label, reduction="none").mean(dim=self.reduction_dim)
        iou_loss = self.iou_loss(pred, label)
        _class_weights = torch.ones(pred.shape[1]) if self.class_weights is None else torch.tensor(self.class_weights)
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        seg_iou_loss_by_channel = {L(f"seg_iou_{i}"): l * _class_weights[i] for i, l in enumerate(iou_loss)}
        seg_dual_focal_loss_by_channel = {L(f"seg_dual_focal_{i}"): l * _class_weights[i] for i, l in enumerate(dual_loss)}
        loss_dict.update(seg_iou_loss_by_channel)
        loss_dict.update(seg_dual_focal_loss_by_channel)
        loss_dict[L("seg_iou")]  = iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / _class_weights.sum() # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[L("seg_dual_focal")] = dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / _class_weights.sum()
        loss_dict[L("seg")] = (loss_dict[L("seg_dual_focal")] + loss_dict[L("seg_iou")]) * self.loss_weights["seg"]
        return loss_dict

    def _cen_loss(self, pred, label, fg_mask, fg_weight=1.0, bg_weight=1.0):
        dual_loss = dual_focal_loss(pred, label, reduction="none").mean()
        fg_dual_loss = (dual_focal_loss(pred, label, reduction="none") * fg_mask).sum() / fg_mask.sum()
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("cen_dual_focal")] = bg_weight * dual_loss
        loss_dict[L("cen_fg_dual_focal")] = fg_weight * fg_dual_loss
        loss_dict[L("cen")] = (loss_dict[L("cen_dual_focal")] + loss_dict[L("cen_fg_dual_focal")]) * self.loss_weights["cen"]
        return loss_dict

    def _reg_loss(
        self,
        pred,
        label,
        fg_mask,
        center_xy_weight=1.0,
        center_z_weight=1.0,
        size_weight=1.0,
        unit_xvec_weight=1.0,
        abs_xvec_weight=1.0,
        xvec_product_weight=1.0,
        abs_roll_angle_weight=1.0,
        roll_angle_product_weight=1.0,
        velo_weight=1.0,
    ):
        assert pred.shape[1] == 20
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        mask_sum = fg_mask.sum()
        def _calc_sub_loss(_weight, channel_slice):
            l1 = nn.L1Loss(reduction="none")(pred[:, channel_slice, ...], label[:, channel_slice, ...])
            n_channels = l1.shape[1]
            return _weight * (fg_mask * l1).sum() / mask_sum / n_channels

        loss_dict = {
            L("reg_center_xy"): _calc_sub_loss(center_xy_weight, slice(0, 2)),
            L("reg_center_z"): _calc_sub_loss(center_z_weight, slice(2, 3)),
            L("reg_size"): _calc_sub_loss(size_weight, slice(3, 6)),
            L("reg_unit_xvec"): _calc_sub_loss(unit_xvec_weight, slice(6, 9)),
            L("reg_abs_xvec"): _calc_sub_loss(abs_xvec_weight, slice(9, 12)),
            L("reg_xvec_product"): _calc_sub_loss(xvec_product_weight, slice(12, 14)),
            L("reg_abs_roll_angle"): _calc_sub_loss(abs_roll_angle_weight, slice(14, 16)),
            L("reg_roll_angle_product"): _calc_sub_loss(roll_angle_product_weight, slice(16, 17)),
            L("reg_velo"): _calc_sub_loss(velo_weight, slice(17, 20)),
        }
        loss_dict[L("reg")] = sum(loss_dict.values()) * self.loss_weights["reg"]
        return loss_dict


class PlanarPolyline3DLoss(nn.Module):
    def __init__(self, *args, loss_name_prefix="plnrplyln3d", class_weights=None, loss_weights=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.loss_weights = {"seg": 1.0, "reg": 1.0}
        self.loss_weights.update(loss_weights or {})
        self.class_weights = class_weights
        self.reduction_dim = (0, 2, 3)
        self.iou_loss = SegIouLoss(pred_logits=True, reduction_dim=self.reduction_dim)

    @staticmethod
    def _ensure_shape_nchw(tensor_dict):
        for k, v in tensor_dict.items():
            if v.dim() == 2:
                tensor_dict[k] = v.unsqueeze(0).unsqueeze(0)
            elif v.dim() == 3:
                tensor_dict[k] = v.unsqueeze(1)
        return tensor_dict

    def forward(self, pred, label):
        assert set(pred.keys()) == set(label.keys()) == {"seg", "reg"}
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        loss = dict(
            **self._seg_loss(pred["seg"], label["seg"]),
            **self._reg_loss(pred["reg"], label["reg"], fg_mask=label["seg"][:, 0:1]),
        )
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss[self.total_loss_name] = loss[L("seg")] + loss[L("reg")]
        return loss

    def _seg_loss(self, pred, label, iou_loss_weight=1.0, dual_focal_loss_weight=1.0):
        dual_loss = dual_focal_loss(pred, label, reduction="none").mean(dim=self.reduction_dim)
        iou_loss = self.iou_loss(pred, label)
        _class_weights = torch.ones(pred.shape[1]) if self.class_weights is None else torch.tensor(self.class_weights)
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        seg_iou_loss_by_channel = {L(f"seg_iou_{i}"): l * _class_weights[i] for i, l in enumerate(iou_loss)}
        seg_dual_focal_loss_by_channel = {L(f"seg_dual_focal_{i}"): l * _class_weights[i] for i, l in enumerate(dual_loss)}
        loss_dict.update(seg_iou_loss_by_channel)
        loss_dict.update(seg_dual_focal_loss_by_channel)
        loss_dict[L("seg_iou")]  = iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / _class_weights.sum() # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[L("seg_dual_focal")] = dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / _class_weights.sum()
        loss_dict[L("seg")] = (loss_dict[L("seg_dual_focal")] + loss_dict[L("seg_iou")]) * self.loss_weights["seg"]
        return loss_dict

    def _reg_loss(
        self,
        pred,
        label,
        fg_mask,
        dist_weight=1.0,
        vert_vec_weight=1.0,
        abs_dir_weight=1.0,
        dir_product_weight=1.0,
        height_weight=1.0,
    ):
        assert pred.shape[1] == 7
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        mask_sum = fg_mask.sum()
        def _calc_sub_loss(_weight, channel_slice):
            l1 = nn.L1Loss(reduction="none")(pred[:, channel_slice, ...], label[:, channel_slice, ...])
            n_channels = l1.shape[1]
            return _weight * (fg_mask * l1).sum() / mask_sum / n_channels

        loss_dict = {
            L("reg_dist"): _calc_sub_loss(dist_weight, slice(0, 1)),
            L("reg_vert_vec"): _calc_sub_loss(vert_vec_weight, slice(1, 3)),
            L("reg_abs_dir"): _calc_sub_loss(abs_dir_weight, slice(3, 5)),
            L("reg_dir_product"): _calc_sub_loss(dir_product_weight, slice(5, 6)),
            L("reg_height"): _calc_sub_loss(height_weight, slice(6, 7)),
        }
        loss_dict[L("reg")] = sum(loss_dict.values()) * self.loss_weights["reg"]
        return loss_dict