import torch
from torch import nn
from mmengine.runner import autocast
from mmengine.model import BaseModel
from prefusion.registry import MODELS
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['SegEncoderDecoder']

@MODELS.register_module()
class SegEncoderDecoder(BaseModel):
    def __init__(self,
                 data_preprocessor,
                 backbone,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor,init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.decoder_head = MODELS.build(decode_head)
        self.init_weights()
    
    def forward(self, inputs, data_samples, mode):
        out = self.decoder_head(self.backbone(inputs))
        # import pdb
        # pdb.set_trace()
        if mode == "loss":
            losses = dict()
            losses.update(self.decoder_head.loss(out, data_samples))
            return losses
        elif mode == 'tensor':
            return out
        elif mode == 'predict':
            pass

    def init_weights(self):
        return super().init_weights()