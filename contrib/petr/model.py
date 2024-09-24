from typing import Union, List, Dict, Any, Optional
from collections import OrderedDict

import torch.nn as nn
import torch
import numpy as np
from mmengine.model import BaseModel
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from prefusion.registry import MODELS
from contrib.petr.misc import locations

__all__ = ["StreamPETR"]


@MODELS.register_module()
class FrameBatchMerger(BaseDataPreprocessor):
    def __init__(self, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self._device = device

    def forward(self, data: List[Dict[str, Any]], training: bool = False) -> Dict[str, List[Any]]:
        merged = {}
        for key in data[0].keys():
            merged[key] = [self._cast_data(i[key]) for i in data]
        return merged

    def _cast_data(self, data: Any):
        if isinstance(data, torch.Tensor):
            _dtype = torch.float32 if "float" in str(data.dtype) else data.dtype
            return data.to(dtype=_dtype, device=self._device)
        if isinstance(data, dict):
            return {k: self._cast_data(data[k]) for k in data}
        return data
        # return self.cast_data(merged)  # type: ignore


@MODELS.register_module()
class StreamPETR(BaseModel):
    def __init__(
        self,
        *,
        data_preprocessor=None,
        img_backbone=None,
        img_neck=None,
        roi_head=None,
        box_head=None,
        stride=16,
        position_level=0,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        data_preprocessor : _type_, optional
            _description_, by default None
        img_backbone : _type_, optional
            _description_, by default None
        img_neck : _type_, optional
            _description_, by default None
        box_head : _type_, optional
            _description_, by default None
        stride : int, optional
            _description_, by default 16
        position_level : int, optional
            用于选择 FPN 结果中的哪一个粒度的特征来做后续的 bbox 预测, by default 0
        """
        super().__init__(data_preprocessor=data_preprocessor)
        assert not any(m is None for m in [img_backbone, img_neck])
        self.img_backbone = MODELS.build(img_backbone)
        self.box_head = MODELS.build(box_head)
        self.roi_head = MODELS.build(roi_head)
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.stride = stride
        self.position_level = position_level

    def forward(self, *args, mode="loss", **kwargs):
        if mode == "loss":
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, *, index_info=None, camera_images=None, bbox_3d=None, ego_poses=None, meta_info=None):
        B, (N, C, H, W) = len(camera_images), camera_images[0].shape
        camera_images = torch.vstack([i.unsqueeze(0) for i in camera_images]).reshape(B * N, C, H, W)
        im_size = camera_images.shape[-2:][::-1]
        img_feats = self.extract_img_feat(camera_images)
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)
        img_feats = img_feats[self.position_level]
        img_feats = img_feats.reshape(B, N, *img_feats.shape[1:])
        location = self.prepare_location(img_feats, im_size)
        outs_roi = self.roi_head(location, img_feats)
        topk_indexes = outs_roi['topk_indexes']
        _device = img_feats.device
        
        data = {
            "timestamp": torch.tensor([int(ii.frame_id) for ii in index_info], device=_device, dtype=torch.float64),
            "prev_exists": torch.tensor([ii.prev is None for ii in index_info], device=_device, dtype=torch.float32),
            "ego_pose": torch.tensor(np.array([p.transformables['0'].trans_mat for p in ego_poses]), device=_device, dtype=torch.float32),
            "ego_pose_inv": torch.tensor(np.array([np.linalg.inv(p.transformables['0'].trans_mat) for p in ego_poses]), device=_device, dtype=torch.float32),
            "intrinsics": torch.tensor(np.array([m["camera_images"]["intrinsic"] for m in meta_info]), device=_device, dtype=torch.float32),
            "lidar2img": torch.tensor(np.array([m["camera_images"]["extrinsic_inv"] for m in meta_info]), device=_device, dtype=torch.float32)
        }

        img_metas = []
        for m in meta_info:
            img_metas.append({
                "pad_shape": [(im_size[1], im_size[0], 3)] * N,
            })
        gt_labels = [m['bbox_3d']['classes'] for m in meta_info]
        outs = self.box_head(img_feats, location, img_metas, bbox_3d, gt_labels, topk_indexes=topk_indexes, **data)

        loss_inputs = [bbox_3d, gt_labels, outs]
        losses = self.box_head.loss(*loss_inputs)

        # if self.with_img_roi_head:
        #     loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
        #     losses2d = self.img_roi_head.loss(*loss2d_inputs)
        #     losses.update(losses2d)

        return losses

    def forward_test(self, *args, **kwargs):
        return

    def extract_img_feat(self, img: torch.Tensor):
        """
        Parameters
        ----------
        img : torch.Tensor
            of shape (N, C, H, W), where N is the batch size, C is usually 3, H and W are image height and width.
        """
        return self.img_backbone(img)

    def prepare_location(self, img_feats, im_size):
        pad_w, pad_h = im_size
        batch_size, num_cameras = img_feats.shape[:2]
        location = locations(img_feats.flatten(0, 1), self.stride, pad_h, pad_w)[None].repeat(batch_size * num_cameras, 1, 1, 1)
        return location
