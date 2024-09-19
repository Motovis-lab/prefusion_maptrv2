from typing import Union, List, Dict, Any, Optional
from collections import OrderedDict

import torch.nn as nn
import torch
from mmengine.model import BaseModel
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmdet.models.dense_heads import AnchorFreeHead

from prefusion.registry import MODELS
from contrib.petr.misc import locations

__all__ = ["StreamPETR", "StreamPETRHead"]


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
            return data.to(dtype=torch.float32, device=self._device)
        return data
        # return self.cast_data(merged)  # type: ignore


@MODELS.register_module()
class StreamPETR(BaseModel):
    def __init__(self, *, data_preprocessor=None, img_backbone=None, img_neck=None, box_head=None, stride=16, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor)
        assert not any(m is None for m in [img_backbone, img_neck])
        self.img_backbone = MODELS.build(img_backbone)
        # self.box_head = MODELS.build(box_head)
        self.img_neck = MODELS.build(img_neck) if img_neck else None

    def forward(self, *args, mode="loss", **kwargs):
        if mode == "loss":
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, *, index_info=None, camera_images=None, bbox_3d=None, meta_info=None):
        B, (N, C, H, W) = len(camera_images), camera_images[0].shape
        camera_images = torch.vstack([i.unsqueeze(0) for i in camera_images]).reshape(B * N, C, H, W)
        img_feats = self.extract_img_feat(camera_images)
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)
        
        location = self.prepare_location(img_feats)

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
    
    def prepare_location(self, img_feats, img_size):
        pad_w, pad_h = img_size
        bs, n = img_feats.shape[:2]
        x = img_feats.flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location


@MODELS.register_module()
class StreamPETRHead(AnchorFreeHead):
    def __init__(self):
        pass
