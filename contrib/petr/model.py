from collections import OrderedDict
import torch.nn as nn

from prefusion.registry import MODELS
from mmengine.model import BaseModel

__all__ = ["ToyModel"]

@MODELS.register_module()
class ToyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(dict(layer0=nn.Linear(1, 1), layer1=nn.Linear(1, 1)))
        self.head = nn.Sequential(OrderedDict(linear=nn.Linear(1, 1), bn=nn.BatchNorm1d(1)))

    def forward(self, x, mode='loss'):
        return self.head(self.backbone.layer0(x))





@MODELS.register_module()
class StreamPETR(BaseModel):
    def __init__(self, *, backbone=None, neck=None, 
                #  box_head=None, 
        **kwargs):
        super().__init__()
        assert not any(m is None for m in [backbone, neck])
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        # self.box_head = MODELS.build(box_head)
    
    def forward(self, model_food, mode='loss'):
        if mode=='loss':
            return self.forward_train(model_food)
        else:
            return self.forward_test(model_food)
    
    def forward_train(self, model_food):
        pass