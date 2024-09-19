from typing import Union, List, Dict, Any, Optional
from collections import OrderedDict

import torch.nn as nn
import torch
from torch.utils.data.dataloader import default_collate
from mmengine.model import BaseModel
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from prefusion.registry import MODELS

__all__ = ["ToyModel", "StreamPETR"]

@MODELS.register_module()
class ToyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(dict(layer0=nn.Linear(1, 1), layer1=nn.Linear(1, 1)))
        self.head = nn.Sequential(OrderedDict(linear=nn.Linear(1, 1), bn=nn.BatchNorm1d(1)))

    def forward(self, x, mode='loss'):
        return self.head(self.backbone.layer0(x))


@MODELS.register_module()
class FrameBatchMerger(BaseDataPreprocessor):
    def __init__(self, device='cuda', **kwargs):
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
    def __init__(self, *, data_preprocessor=None, img_backbone=None, img_neck=None, box_head=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor)
        assert not any(m is None for m in [img_backbone, img_neck])
        self.img_backbone = MODELS.build(img_backbone)
        # self.box_head = MODELS.build(box_head)
        self.img_neck = MODELS.build(img_neck) if img_neck else None
    
    def forward(self, *args, mode='loss', **kwargs):
        if mode=='loss':
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)
    
    def forward_train(self, *, index_info=None, camera_images=None, bbox_3d=None, meta_info=None):
        B, (N, C, H, W) = len(camera_images), camera_images[0].shape
        camera_images = torch.vstack([i.unsqueeze(0) for i in camera_images]).reshape(B * N, C, H, W)
        img_feat = self.extract_img_feat(camera_images)
        if self.img_neck is not None:
            img_feat = self.img_neck(img_feat)
        a = 10

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
