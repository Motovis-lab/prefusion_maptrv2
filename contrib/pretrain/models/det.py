import torch
from torch import nn
from mmengine.runner import autocast
from mmengine.model import BaseModel
from prefusion.registry import MODELS
import torch.nn.functional as F
import numpy as np

__all__ = ['ADAS_Det']

@MODELS.register_module()
class ADAS_Det(BaseModel):
    def __init__(self,
                 data_preprocessor,
                 backbone,
                 bbox_head,
                 fpn=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if fpn is not None:
            self.fpn = MODELS.build(fpn)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def forward(self, inputs, data_samples, mode): # type: ignore
        out = self.bbox_head(self.fpn(self.backbone(inputs)))
        if mode == "loss":
            losses = dict()
            losses.update(self.bbox_head.loss(out, data_samples))
            return losses
        elif mode == 'tensor':
            return out
        elif mode == 'predict':
            pass

    def init_weights(self):
        return super().init_weights()