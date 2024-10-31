from typing import Tuple, Union, List, Dict
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from prefusion.registry import MODELS
from prefusion.loss.basic import dual_focal_loss, SegIouLoss


__all__ = ["PlanarLoss"]


def complete_loss_name(loss_name_prefix, sub_item_name: str = ""):
    if sub_item_name:
        return f"{loss_name_prefix}_{sub_item_name}_loss"
    return f"{loss_name_prefix}_loss"


@MODELS.register_module()
class PlanarLoss(nn.Module):

    reduction_dim: Union[Tuple, List] = (0, 2, 3)

    def __init__(
        self, *args, loss_name_prefix: str = None, weight_scheme: Dict[str, Dict] = None, auto_loss_init_value: float = 1e-5, **kwargs
    ) -> None:
        """Specially designed loss for planar data.

        Parameters
        ----------
        loss_name_prefix : str
            the name prefix of the loss (used to distinguish between different losses)

        weight_scheme : dict, optional
            the weight scheme for different losses, e.g. seg, cen, reg

            Example
            ---
                ```{
                    "seg": {
                        "loss_weight": 1.0,  # most superior loss weight for seg
                        "channel_weights": {
                            "all": {"weight": 0.5}, # user must add this special channel manually
                            "passenger_car": {"weight": 0.7},
                            "pedestrian": {"weight": 0.3},
                        },
                        "iou_loss_weight": "auto",
                        "dual_focal_loss_weight": "auto",
                    },
                    "cen": {
                        "loss_weight": 1.0, # most superior loss weight for cen
                        "fg_weight": "auto",
                        "bg_weight": "auto",
                    },
                    "reg": {
                        "loss_weight": 1.0, # most superior loss weight for reg
                        "partition_weights": {
                            "center_xy": {"weight": 1.0, "slice": (0, 2)},
                            "center_z": {"weight": 1.0, "slice": 2},
                            "size": {"weight": 1.0, "slice": (3, 6)},
                            "unit_xvec": {"weight": 1.0, "slice": (6, 9)},
                            "abs_xvec": {"weight": 1.0, "slice": (9, 12)},
                            "xvec_product": {"weight": 1.0, "slice": (12, 14)},
                            "abs_roll_angle": {"weight": 1.0, "slice": (14, 16)},
                            "roll_angle_product": {"weight": 1.0, "slice": 16},
                            "velo": {"weight": 1.0, "slice": (17, 20)},
                        }
                    }
                }

        auto_loss_init_value : float, optional
            the initial value for auto weight (torch parameter). Defaults to 1e-5.

        """
        super().__init__(*args, **kwargs)
        assert loss_name_prefix is not None, "loss_name_prefix must be provided"
        assert weight_scheme is not None and len(weight_scheme) > 0, "weight_scheme must be provided with minimum info. E.g. {'taskA': {'loss_weight': 1.0}}"
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.weight_scheme = edict(weight_scheme)
        self.auto_loss_init_value = float(auto_loss_init_value)
        self.iou_loss = SegIouLoss(pred_logits=True, reduction_dim=self.reduction_dim)
        self.init_auto_loss_weight_(self.weight_scheme)
        self.unify_reg_partition_weights_()

    def init_auto_loss_weight_(self, weight_scheme: Dict):
        for k, v in weight_scheme.items():
            if isinstance(v, dict):
                weight_scheme[k] = self.init_auto_loss_weight_(v)
                continue

            if k.endswith("weight") and isinstance(v, str) and v.lower() == "auto":
                weight_scheme[k] = nn.Parameter(torch.tensor(self.auto_loss_init_value))

        return weight_scheme

    def unify_reg_partition_weights_(self):
        """Unify single value index to standard 2-value slice"""
        if "reg" not in self.weight_scheme or "partition_weights" not in self.weight_scheme["reg"]:
            return
        
        for partition_name, partition_info in self.weight_scheme.reg.partition_weights.items():
            try:
                _slice = slice(*partition_info.slice)
            except TypeError:  # partition_info.slice is a single number
                _slice = slice(partition_info.slice, partition_info.slice + 1)
            self.weight_scheme.reg.partition_weights[partition_name].slice = _slice

    @staticmethod
    def _ensure_shape_nchw(tensor_dict: Dict[str, torch.Tensor]):
        for k, v in tensor_dict.items():
            if v.dim() == 2:
                tensor_dict[k] = v.unsqueeze(0).unsqueeze(0)
            elif v.dim() == 3:
                tensor_dict[k] = v.unsqueeze(1)
        return tensor_dict

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        assert set(pred.keys()) == set(label.keys()) == set(self.weight_scheme.keys())
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        loss = {}
        for task in self.weight_scheme:
            _loss_func = getattr(self, f"_{task}_loss")
            fg_mask = label["seg"][:, 0:1] if task in ["cen", "reg"] else None
            task_related_loss = _loss_func(pred[task], label[task], fg_mask=fg_mask, **self.weight_scheme[task])
            loss.update(**task_related_loss)

        L = partial(complete_loss_name, self.loss_name_prefix)
        loss[self.total_loss_name] = sum([loss[L(task)] for task in self.weight_scheme])
        return loss

    def _seg_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        channel_weights: dict = None,
        iou_loss_weight: float = 1.0,
        dual_focal_loss_weight: float = 1.0,
        **kwargs,
    ):
        num_cls = pred.shape[1]

        _channel_weights = torch.ones(num_cls)
        if channel_weights is not None:
            _channel_weights = [c["weight"] for _, c in channel_weights.items()]

        assert num_cls == len(_channel_weights), f"number of channel weights does not match with number of classes"

        _class_names = [f"{i}" for i in range(num_cls)] if channel_weights is None else list(channel_weights.keys())

        dual_loss = dual_focal_loss(pred, label, reduction="none").mean(dim=self.reduction_dim)
        iou_loss = self.iou_loss(pred, label)

        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)

        seg_iou_loss_by_channel = {
            L(f"seg_iou_{c}"): l * w for c, w, l in zip(_class_names, _channel_weights, iou_loss)
        }
        seg_dual_focal_loss_by_channel = {
            L(f"seg_dual_focal_{c}"): l * w for c, w, l in zip(_class_names, _channel_weights, dual_loss)
        }
        loss_dict.update(seg_iou_loss_by_channel)
        loss_dict.update(seg_dual_focal_loss_by_channel)
        loss_dict[L("seg_iou")] = (
            iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / sum(_channel_weights)
        )  # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[L("seg_dual_focal")] = (
            dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / sum(_channel_weights)
        )
        loss_dict[L("seg")] = (loss_dict[L("seg_dual_focal")] + loss_dict[L("seg_iou")]) * loss_weight
        return loss_dict

    def _cen_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        fg_weight: float = 1.0,
        bg_weight: float = 1.0,
        **kwargs,
    ):
        dual_loss = dual_focal_loss(pred, label, reduction="none").mean()
        fg_dual_loss = (dual_focal_loss(pred, label, reduction="none") * fg_mask).sum() / fg_mask.sum()
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("cen_dual_focal")] = bg_weight * dual_loss
        loss_dict[L("cen_fg_dual_focal")] = fg_weight * fg_dual_loss
        loss_dict[L("cen")] = (loss_dict[L("cen_dual_focal")] + loss_dict[L("cen_fg_dual_focal")]) * loss_weight
        return loss_dict

    def _reg_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        partition_weights: Dict[str, dict] = None,
        **kwargs,
    ):
        if partition_weights is None:
            partition_weights = edict({f"{c}": {"weight": 1.0, "slice": slice(c, c + 1)} for c in range(label.shape[1])})

        assert list(range(pred.shape[1])) == self.enumerate_slices(
            [p.slice for p in partition_weights.values()]
        ), "partition weight slices doesn't meet MECE principle."
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        mask_sum = fg_mask.sum()

        def _calc_sub_loss(_weight, channel_slice):
            l1 = nn.L1Loss(reduction="none")(pred[:, channel_slice, ...], label[:, channel_slice, ...])
            n_channels = l1.shape[1]
            return _weight * (fg_mask * l1).sum() / mask_sum / n_channels

        loss_dict = {}
        for partition_name, partition_info in partition_weights.items():
            loss_dict[L(f"reg_{partition_name}")] = _calc_sub_loss(partition_info.weight, partition_info.slice)

        loss_dict[L("reg")] = sum(loss_dict.values()) * loss_weight
        return loss_dict

    @staticmethod
    def enumerate_slices(slices: List):
        # Same as: sorted(reduce(lambda acc, x: acc + list(range(x.start, x.stop)), slices, []))
        return sorted(chain.from_iterable(range(s.start, s.stop) for s in slices))
