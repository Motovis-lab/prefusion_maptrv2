from collections import OrderedDict
import torch.nn as nn

from mmengine.registry import MODELS
from mmengine.model import BaseModel

__all__ = ["ToyModel"]

@MODELS.register_module()
class ToyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(dict(layer0=nn.Linear(1, 1), layer1=nn.Linear(1, 1)))
        self.head = nn.Sequential(OrderedDict(linear=nn.Linear(1, 1), bn=nn.BatchNorm1d(1)))

    def forward(self, x):
        return self.head(self.backbone.layer0(x))
