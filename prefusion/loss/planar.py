from typing import Tuple, Union, List, Dict
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
from easydict import EasyDict as edict
import torch.nn.functional as F
from prefusion.registry import MODELS
from prefusion.loss.basic import dual_focal_loss, SegIouLoss,dual_focal_loss_mask, calculate_l2_loss, WeightedFocalLossWithSoftmax, WeightedFocalLossWithSigmoid
import os
from typing import Sequence, Union
from PIL import Image
import math
__all__ = ["PlanarLoss"]


def complete_loss_name(loss_name_prefix, sub_item_name: str = ""):
    if sub_item_name:
        return f"{loss_name_prefix}_{sub_item_name}_loss"
    return f"{loss_name_prefix}_loss"


def save_mask_grid(
    mask_batch: torch.Tensor,
    index_info_str_batch: Union[Sequence[str], list, tuple],
    save_dir: str = ".",
    ncols: int = None,
    line_width: int = 2,
    red=(255, 0, 0),
):
    """
    将形如 [B, C, H, W] 的 mask 批次保存为网格图。
    - 支持任意通道数 C（每个通道输出为一张灰度小图，拼成网格）
    - 通道之间及四周使用红色分隔线（line_width像素）
    - 自动按近似正方形布局，也可以通过 ncols 指定列数

    参数:
        mask_batch: torch.Tensor, [B, C, H, W]
        index_info_str_batch: 长度为 B 的字符串列表/序列
        save_dir: 输出目录
        ncols: 指定列数；None 则自动 sqrt(C) 取整
        line_width: 分隔线宽度（像素）
        red: 分隔线颜色 (R,G,B)
    """
    assert mask_batch.ndim == 4, "mask_batch 必须是 [B, C, H, W]"
    B, C, H, W = mask_batch.shape
    assert len(index_info_str_batch) == B, "index_info_str_batch 长度应等于 batch 大小"

    # 自动列数：尽量接近正方形
    if ncols is None:
        ncols = max(1, int(math.ceil(math.sqrt(C))))
    nrows = int(math.ceil(C / ncols))

    os.makedirs(save_dir, exist_ok=True)

    # 预处理到 uint8（0或255）
    m = mask_batch.detach().cpu()
    if m.dtype.is_floating_point:
        m = m.clamp(0, 1) * 255
        m = m.to(torch.uint8)
    else:
        m = (m > 0).to(torch.uint8) * 255
    # m: [B, C, H, W]，每个通道灰度(0~255)

    # 画布大小（含外边框）：红色背景 + 灰度贴片
    canvas_h = nrows * H + (nrows + 1) * line_width
    canvas_w = ncols * W + (ncols + 1) * line_width

    for b in range(B):
        # 建立红色背景画布 (H, W, 3)
        canvas = torch.zeros((canvas_h, canvas_w, 3), dtype=torch.uint8)
        canvas[:, :, 0] = red[0]
        canvas[:, :, 1] = red[1]
        canvas[:, :, 2] = red[2]

        # 将每个通道贴到对应网格位置
        for c in range(C):
            r = c // ncols  # 行索引
            col = c % ncols  # 列索引

            y0 = r * H + (r + 1) * line_width
            x0 = col * W + (col + 1) * line_width
            y1 = y0 + H
            x1 = x0 + W

            tile = m[b, c]  # [H, W] uint8
            # 转为 RGB 灰度（3通道相同值）
            # 为了减少拷贝，直接写入三个通道
            canvas[y0:y1, x0:x1, 0] = tile
            canvas[y0:y1, x0:x1, 1] = tile
            canvas[y0:y1, x0:x1, 2] = tile

        # 保存
        index_info_str = str(index_info_str_batch[b])
        filename = index_info_str.replace("/", "_") + ".png"
        img = Image.fromarray(canvas.numpy(), mode="RGB")
        img.save(os.path.join(save_dir, filename))
        print(f"保存完成: {os.path.join(save_dir, filename)}")

@MODELS.register_module()
class PlanarLoss(nn.Module):

    reduction_dim: Union[Tuple, List] = (0, 2, 3)

    def __init__(
        self,
        *args,
        loss_name_prefix: str = None,
        seg_iou_method: str = "log",
        weight_scheme: Dict[str, Dict] = None,
        auto_loss_init_value: float = 1e-5,
        **kwargs,
    ) -> None:
        """Specially designed loss for planar data.

        Parameters
        ----------
        loss_name_prefix : str
            the name prefix of the loss (used to distinguish between different losses)

        seg_iou_method : str
            the method to use in the calculation of SegIOU Loss. Valid choices are ['log', 'linear']

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
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.weight_scheme = edict(weight_scheme)
        self.auto_loss_init_value = float(auto_loss_init_value)
        self.iou_loss = SegIouLoss(method=seg_iou_method, pred_logits=True, reduction_dim=self.reduction_dim)
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

    def forward(self, pred: torch.Tensor, label: torch.Tensor, index_info_str: str="",mask: torch.Tensor=None):
        assert set(pred.keys()) == set(label.keys())
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        # 自动获取 batch
        batch = mask.size(0)
        # 判断每个样本 mask 是否全为 1，全为1在该样本返回1，存在mask=0的地方则返回1   保持可广播形状
        # 逻辑：目前只有IPM图片存在mask, IPM图片没有车位外的其它标注，因此需要完全mask掉车位外的其它分支loss
        all_one_flag = (mask.view(batch, -1).min(dim=1).values == 1).view(batch, 1, 1, 1)

        loss = {}
        for task in label:
            _loss_func = getattr(self, f"_{task}_loss")
            fg_mask = label["seg"][:, 0:1] if task in ["cen", "reg"] else None
            self.weight_scheme.setdefault(task, {"loss_weight": 1.0})  # if not provided, use 1.0 as default
            task_related_loss = _loss_func(pred[task], label[task], fg_mask=fg_mask,mask=mask, all_one_flag=all_one_flag,index_info_str=index_info_str, **self.weight_scheme[task])
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
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        index_info_str="",
        **kwargs,
    ):
        # 如果存在mask flag是1，则mask全为1，不参与这个head训练；
        # 逻辑：目前只有IPM图片存在mask, IPM图片没有车位外的其它标注，因此需要完全mask掉车位外的其它分支loss
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))
        # 需要将mask扩展到与pred相同的形状
        mask = mask.expand(-1, pred.shape[1], -1, -1)
        # mask是在车位分支分辨率生成的，需要进行下采样，适配其它head分辨率
        mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')

        # move mask into loss function
        # pred  = pred * mask  
        # label = label * mask

        num_cls = pred.shape[1]

        _channel_weights = torch.ones(num_cls)
        if channel_weights is not None:
            _channel_weights = [c["weight"] for _, c in channel_weights.items()]
        channel_weights_sum = max(sum(_channel_weights), 1e-4)

        assert num_cls == len(_channel_weights), f"number of channel weights does not match with number of classes"

        _class_names = [f"{i}" for i in range(num_cls)] if channel_weights is None else list(channel_weights.keys())

        dual_loss = dual_focal_loss_mask(pred, label,  mask=mask, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None,reduction_dim=self.reduction_dim, eps=1e-6)
        # dual_loss = dual_focal_loss(pred, label, mask, reduction="none").mean(dim=self.reduction_dim)
        iou_loss = self.iou_loss(pred, label, mask,index_info_str)

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
            iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / channel_weights_sum
        )  # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[L("seg_dual_focal")] = (
            dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / channel_weights_sum
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
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        index_info_str="",        
        **kwargs,
    ):
        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.expand(-1, pred.shape[1], -1, -1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        
        # pred  = pred * mask
        # label = label * mask

        dual_loss = dual_focal_loss_mask(pred, label,  mask=mask, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None,eps=1e-6)
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("cen_dual_focal")] = bg_weight * dual_loss
        loss_dict[L("cen")] = loss_dict[L("cen_dual_focal")] * loss_weight
        return loss_dict

        # dual_loss = dual_focal_loss(pred, label, mask, reduction="none").mean()
        # mask_sum = max(fg_mask.sum(), 1e-4)  # fg_mask is assumed to be from label (gt), so no need to worry about backward
        # fg_dual_loss = (dual_focal_loss(pred, label, mask, reduction="none") * fg_mask).sum() / mask_sum
        # loss_dict = {}
        # L = partial(complete_loss_name, self.loss_name_prefix)
        # loss_dict[L("cen_dual_focal")] = bg_weight * dual_loss
        # loss_dict[L("cen_fg_dual_focal")] = fg_weight * fg_dual_loss
        # loss_dict[L("cen")] = (loss_dict[L("cen_dual_focal")] + loss_dict[L("cen_fg_dual_focal")]) * loss_weight
        # return loss_dict



    def _reg_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        partition_weights: Dict[str, dict] = None,
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        index_info_str="",          
        **kwargs,
    ):
        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.expand(-1, pred.shape[1], -1, -1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        pred  = pred * mask
        label = label * mask

        if partition_weights is None:
            partition_weights = edict(
                {f"{c}": {"weight": 1.0, "slice": slice(c, c + 1)} for c in range(label.shape[1])}
            )

        assert list(range(pred.shape[1])) == self.enumerate_slices(
            [p.slice for p in partition_weights.values()]
        ), f"partition weight slices doesn't meet MECE principle. ({label.shape[1]=}, {pred.shape[1]=})"
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        mask_sum = max(fg_mask.sum(), 1e-4)  # fg_mask is assumed to be from label (gt), so no need to worry about backward

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

@MODELS.register_module()
class PolylineLoss(nn.Module):

    reduction_dim: Union[Tuple, List] = (0, 2, 3)

    def __init__(
        self,
        *args,
        loss_name_prefix: str = None,
        seg_iou_method: str = "log",
        weight_scheme: Dict[str, Dict] = None,
        auto_loss_init_value: float = 1e-5,
        **kwargs,
    ) -> None:
        """Specially designed loss for planar data.

        Parameters
        ----------
        loss_name_prefix : str
            the name prefix of the loss (used to distinguish between different losses)

        seg_iou_method : str
            the method to use in the calculation of SegIOU Loss. Valid choices are ['log', 'linear']

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
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.weight_scheme = edict(weight_scheme)
        self.auto_loss_init_value = float(auto_loss_init_value)
        self.iou_loss = SegIouLoss(method=seg_iou_method, pred_logits=True, reduction_dim=self.reduction_dim)
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

    def forward(self, pred: torch.Tensor, label: torch.Tensor):  #, index_info_str: str="", mask: torch.Tensor=None):
        assert set(pred.keys()) == set(label.keys())
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        loss = {}
        for task in label:
            _loss_func = getattr(self, f"_{task}_loss")
            # fg_mask = label["seg"][:, 0:1] if task in ["cen", "reg"] else None
            if task in ["cen", "reg"]:
                fg_mask = label["seg"][:, 0:1]
            elif task == "laneloc":
                fg_mask = (
                    label.get("lanecls")
                )
            else:
                fg_mask = None
            self.weight_scheme.setdefault(task, {"loss_weight": 1.0})  # if not provided, use 1.0 as default
            task_related_loss = _loss_func(pred[task], label[task], fg_mask=fg_mask, **self.weight_scheme[task])
            loss.update(**task_related_loss)

        L = partial(complete_loss_name, self.loss_name_prefix)
        loss[self.total_loss_name] = sum([loss[L(task)] for task in self.weight_scheme])
        return loss
    
    def _lanecls_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        **kwargs,
    ):
        L = partial(complete_loss_name, self.loss_name_prefix)
        # if label.sum() > 0:
        cls_loss_fn = WeightedFocalLossWithSoftmax().to(pred.device)
        cls_loss = cls_loss_fn(pred, label)
        # print(f"cls_loss: {cls_loss * loss_weight}")
        # else:
        #     cls_loss = torch.tensor(0.0, device=pred.device)
        return {L("lanecls"): cls_loss * loss_weight}

    def _laneloc_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        **kwargs,
    ):
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict = {}

        if fg_mask.sum() > 0:
            loc_loss = calculate_l2_loss(label, pred, fg_mask)
            loss_dict[L("loss_xy")] = loc_loss["loss_xy"] * loss_weight
            loss_dict[L("loss_ang")] = loc_loss["loss_ang"] * loss_weight
            loss_dict[L("laneloc")] = loc_loss ["loss"]* loss_weight
        else:
            # loc_loss = (pred.reshape(-1).sum() * 0.0).abs()  # 为了置零
            zero_loss = (pred.reshape(-1).sum() * 0.0).abs()
            loss_dict[L("loss_xy")] = zero_loss
            loss_dict[L("loss_ang")] = zero_loss
            loss_dict[L("laneloc")] = zero_loss
        # print(f"loc_loss: {loc_loss}")
        return loss_dict
        # if fg_mask.sum() > 0:
        #     loc_loss = calculate_l2_loss(label, pred, fg_mask)
        # else:
        #     loc_loss = (pred.reshape(-1).sum() * 0.0).abs()  # 为了置零
        # # print(f"loc_loss: {loc_loss}")

        # return {L("laneloc"): loc_loss * loss_weight}
    
    
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
        channel_weights_sum = max(sum(_channel_weights), 1e-4)

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
            iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / channel_weights_sum
        )  # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[L("seg_dual_focal")] = (
            dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / channel_weights_sum
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
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("cen_dual_focal")] = bg_weight * dual_loss
        loss_dict[L("cen")] = loss_dict[L("cen_dual_focal")] * loss_weight
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
            partition_weights = edict(
                {f"{c}": {"weight": 1.0, "slice": slice(c, c + 1)} for c in range(label.shape[1])}
            )

        assert list(range(pred.shape[1])) == self.enumerate_slices(
            [p.slice for p in partition_weights.values()]
        ), f"partition weight slices doesn't meet MECE principle. ({label.shape[1]=}, {pred.shape[1]=})"
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        mask_sum = max(fg_mask.sum(), 1e-4)  # fg_mask is assumed to be from label (gt), so no need to worry about backward

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



@MODELS.register_module()
class PolylineMulLoss(nn.Module):

    reduction_dim: Union[Tuple, List] = (0, 2, 3)

    def __init__(
        self,
        *args,
        loss_name_prefix: str = None,
        seg_iou_method: str = "log",
        weight_scheme: Dict[str, Dict] = None,
        auto_loss_init_value: float = 1e-5,
        **kwargs,
    ) -> None:
        """Specially designed loss for planar data.

        Parameters
        ----------
        loss_name_prefix : str
            the name prefix of the loss (used to distinguish between different losses)

        seg_iou_method : str
            the method to use in the calculation of SegIOU Loss. Valid choices are ['log', 'linear']

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
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.weight_scheme = edict(weight_scheme)
        self.auto_loss_init_value = float(auto_loss_init_value)
        self.iou_loss = SegIouLoss(method=seg_iou_method, pred_logits=True, reduction_dim=self.reduction_dim)
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
        assert set(pred.keys()) == set(label.keys())
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        loss = {}
        for task in label:
            _loss_func = getattr(self, f"_{task}_loss")
            # fg_mask = label["seg"][:, 0:1] if task in ["cen", "reg"] else None
            if task in ["cen", "reg"]:
                fg_mask = label["seg"][:, 0:1]
            elif task == "laneloc":
                fg_mask = (
                    label.get("lanecls")
                )
            else:
                fg_mask = None
            self.weight_scheme.setdefault(task, {"loss_weight": 1.0})  # if not provided, use 1.0 as default
            task_related_loss = _loss_func(pred[task], label[task], fg_mask=fg_mask, **self.weight_scheme[task])
            loss.update(**task_related_loss)

        L = partial(complete_loss_name, self.loss_name_prefix)
        loss[self.total_loss_name] = sum([loss[L(task)] for task in self.weight_scheme])
        return loss

        
        
    
    def _lanecls_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        **kwargs,
    ):
        L = partial(complete_loss_name, self.loss_name_prefix)
        cls_loss_fn = WeightedFocalLossWithSigmoid().to(pred.device)
        # print("pred shape:", pred.shape)
        # print("label shape:", label.shape)
        cls_losses = cls_loss_fn(pred, label)  # 返回每个类别的损失列表
        
        loss_dict = {}
        # 为每个类别创建单独的损失项
        for i, cls_loss in enumerate(cls_losses):
            # print(f"lanecls_{i}: {cls_loss}")
            loss_dict[L(f"lanecls_{i}")] = cls_loss * loss_weight
        
        # 计算总的lanecls损失作为所有类别损失的和
        total_cls_loss = sum(cls_losses)
        loss_dict[L("lanecls")] = total_cls_loss * loss_weight

        return loss_dict

    def _laneloc_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        **kwargs,
    ):
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict = {}


        # return loss_dict
        if fg_mask.sum() > 0:
            loc_loss = calculate_l2_loss(label, pred, fg_mask)
            loss_dict[L("loss_xy")] = loc_loss["loss_xy"] * loss_weight
            loss_dict[L("loss_ang")] = loc_loss["loss_ang"] * loss_weight
            loss_dict[L("laneloc")] = loc_loss ["loss"]* loss_weight
        else:
            # loc_loss = (pred.reshape(-1).sum() * 0.0).abs()  # 为了置零
            zero_loss = (pred.reshape(-1).sum() * 0.0).abs()
            loss_dict[L("loss_xy")] = zero_loss
            loss_dict[L("loss_ang")] = zero_loss
            loss_dict[L("laneloc")] = zero_loss
        # print(f"loc_loss: {loc_loss}")
        return loss_dict
        # return {L("laneloc"): loc_loss * loss_weight}



@MODELS.register_module()
class ParkingLoss_IPM(nn.Module):

    reduction_dim: Union[Tuple, List] = (0, 2, 3)

    def __init__(
        self,
        *args,
        loss_name_prefix: str = None,
        seg_iou_method: str = "log",
        weight_scheme: Dict[str, Dict] = None,
        auto_loss_init_value: float = 1e-5,
        **kwargs,
    ) -> None:
        """Specially designed loss for planar data.

        Parameters
        ----------
        loss_name_prefix : str
            the name prefix of the loss (used to distinguish between different losses)

        seg_iou_method : str
            the method to use in the calculation of SegIOU Loss. Valid choices are ['log', 'linear']

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
        self.loss_name_prefix = loss_name_prefix
        self.total_loss_name = complete_loss_name(self.loss_name_prefix)
        self.weight_scheme = edict(weight_scheme)
        self.auto_loss_init_value = float(auto_loss_init_value)
        self.iou_loss = SegIouLoss(method=seg_iou_method, pred_logits=True, reduction_dim=self.reduction_dim)
        self.init_auto_loss_weight_(self.weight_scheme)
        self.unify_reg_partition_weights_()
        # self.criterion = torch.nn.MSELoss(size_average=True).cuda()
        # self.seg_bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).cuda())
        self.criterion = torch.nn.MSELoss(reduction='none').cuda()
        self.seg_bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor([3.0]).cuda()).cuda()

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

    def forward(self, pred: torch.Tensor, label: torch.Tensor, index_info_str: str="",mask: torch.Tensor=None):
        assert set(pred.keys()) == set(label.keys())
        pred = self._ensure_shape_nchw(pred)
        label = self._ensure_shape_nchw(label)

        # 自动获取 batch
        batch = mask.size(0)
        # 判断每个样本 mask 是否全为 1，保持可广播形状
        all_one_flag = (mask.view(batch, -1).min(dim=1).values == 1).view(batch, 1, 1, 1)

        loss = {}
        for task in label:
            _loss_func = getattr(self, f"_{task}_loss")
            fg_mask = label["seg"][:, 0:1] if task in ["cen", "reg"] else None
            self.weight_scheme.setdefault(task, {"loss_weight": 1.0})  # if not provided, use 1.0 as default
            task_related_loss = _loss_func(pred[task], label[task], fg_mask=fg_mask,mask=mask, all_one_flag=all_one_flag,index_info_str=index_info_str, **self.weight_scheme[task])
            loss.update(**task_related_loss)

        L = partial(complete_loss_name, self.loss_name_prefix)
        loss[self.total_loss_name] = sum([loss[L(task)] for task in self.weight_scheme])
        return loss

    # with sigmoid and all gt=1
    def _pts_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        loss_weight_seg: float = 30.0,
        loss_weight_reg: float = 150.0,
        index_info_str="",
        **kwargs,
    ):

        out_point = F.sigmoid(pred) 
        loss_seg=self.seg_bce_loss(pred, label)
        loss_reg=self.criterion(out_point, label)

        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), mask)
        mask = mask.expand(-1, label.shape[1], -1, -1)

        # save_dir = "./pts_mask_output"
        # os.makedirs(save_dir, exist_ok=True)
        # save_mask_grid(mask, index_info_str, save_dir)   
        
        loss_seg = (loss_seg * mask).sum() / (mask.sum() + 1e-6)
        loss_reg = (loss_reg * mask).sum() / (mask.sum() + 1e-6)

        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("pts_seg")] = loss_weight_seg * loss_seg
        loss_dict[L("pts_reg")] = loss_weight_reg * loss_reg
        loss_dict[L("pts")] = (loss_dict[L("pts_seg")] + loss_dict[L("pts_reg")]) * loss_weight
        return loss_dict

    def _pin_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        fg_mask: torch.Tensor = None,
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        loss_weight_seg: float = 30.0,
        loss_weight_reg: float = 150.0,
        index_info_str="",
        **kwargs,
    ):
        out_point = F.sigmoid(pred) 
        loss_seg=self.seg_bce_loss(pred, label)
        loss_reg=self.criterion(out_point, label)

        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.expand(-1, label.shape[1], -1, -1)
        loss_seg = (loss_seg * mask).sum() / (mask.sum() + 1e-6)
        loss_reg = (loss_reg * mask).sum() / (mask.sum() + 1e-6)

        # save_dir = "./pin_mask_output"
        # os.makedirs(save_dir, exist_ok=True)
        # save_mask_grid(mask, index_info_str, save_dir)   

        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("pin_seg")] = loss_weight_seg * loss_seg
        loss_dict[L("pin_reg")] = loss_weight_reg * loss_reg
        loss_dict[L("pin")] = (loss_dict[L("pin_seg")] + loss_dict[L("pin_reg")]) * loss_weight
        return loss_dict

    def _seg_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        channel_weights: dict = None,
        iou_loss_weight: float = 1.0,
        dual_focal_loss_weight: float = 1.0,
        index_info_str="",
        **kwargs,
    ):
        num_cls = pred.shape[1]

        _channel_weights = torch.ones(num_cls)
        if channel_weights is not None:
            _channel_weights = [c["weight"] for _, c in channel_weights.items()]
        channel_weights_sum = max(sum(_channel_weights), 1e-4)

        assert num_cls == len(_channel_weights), f"number of channel weights does not match with number of classes"

        _class_names = [f"{i}" for i in range(num_cls)] if channel_weights is None else list(channel_weights.keys())


        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), mask)
        mask = mask.expand(-1, label.shape[1], -1, -1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        pred  = pred * mask
        label = label * mask

        # save_dir = "./seg_mask_output"
        # os.makedirs(save_dir, exist_ok=True)
        # save_mask_grid(mask, index_info_str, save_dir)   

        dual_loss = dual_focal_loss_mask(pred, label,  mask=mask, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None, reduction_dim=self.reduction_dim,eps=1e-6)
        iou_loss = self.iou_loss(pred, label, mask,index_info_str)
        # dual_loss = dual_focal_loss(pred, label, reduction="none").mean(dim=self.reduction_dim)
        # iou_loss = self.iou_loss(pred, label)

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
            iou_loss_weight * sum(seg_iou_loss_by_channel.values()) / channel_weights_sum
        )  # functools.reduce(lambda x, y: x + y, seg_iou_loss_by_channel.values())
        loss_dict[L("seg_dual_focal")] = (
            dual_focal_loss_weight * sum(seg_dual_focal_loss_by_channel.values()) / channel_weights_sum
        )
        loss_dict[L("seg")] = (loss_dict[L("seg_dual_focal")] + loss_dict[L("seg_iou")]) * loss_weight
        return loss_dict

    def _cen_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,
        fg_mask: torch.Tensor = None,
        fg_weight: float = 1.0,
        bg_weight: float = 1.0,
        index_info_str="",
        **kwargs,
    ):
        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.expand(-1, label.shape[1], -1, -1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        # pred  = pred * mask
        # label = label * mask

        # save_dir = "./cen_mask_output"
        # os.makedirs(save_dir, exist_ok=True)
        # save_mask_grid(mask, index_info_str, save_dir)   
        dual_loss = dual_focal_loss_mask(pred, label,  mask=mask, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None, eps=1e-6)
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        loss_dict[L("cen_dual_focal")] = bg_weight * dual_loss
        loss_dict[L("cen")] = (loss_dict[L("cen_dual_focal")]) * loss_weight
        return loss_dict

    def _reg_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        loss_weight: float = 1.0,
        mask: torch.Tensor = None, 
        all_one_flag: torch.Tensor = None,        
        fg_mask: torch.Tensor = None,
        partition_weights: Dict[str, dict] = None,
        index_info_str="",
        **kwargs,
    ):
        
        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.expand(-1, label.shape[1], -1, -1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        pred  = pred * mask
        label = label * mask

        # save_dir = "./reg_mask_output"
        # os.makedirs(save_dir, exist_ok=True)
        # save_mask_grid(mask, index_info_str, save_dir)   

        if partition_weights is None:
            partition_weights = edict(
                {f"{c}": {"weight": 1.0, "slice": slice(c, c + 1)} for c in range(label.shape[1])}
            )

        assert list(range(pred.shape[1])) == self.enumerate_slices(
            [p.slice for p in partition_weights.values()]
        ), f"partition weight slices doesn't meet MECE principle. ({label.shape[1]=}, {pred.shape[1]=})"
        loss_dict = {}
        L = partial(complete_loss_name, self.loss_name_prefix)
        mask_sum = max(fg_mask.sum(), 1e-4)  # fg_mask is assumed to be from label (gt), so no need to worry about backward

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