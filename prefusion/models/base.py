# Copyright (c) MOTOVIS. All rights reserved.

import torch
import torch.nn as nn

from mmengine.model import BaseModel



class MvBaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def get_module_name(self, module):
        for n, m in self.named_modules():
            if module is m:
                return n
        raise ModuleNotFoundError

    def find_module_by_name(self, name):
        """Note: This method can return the attribute of an module, for e.g., net.backbone.conv.weight. """
        parts = name.split('.')
        m = self
        for p in parts:
            m = getattr(m, p)
        return m



class MvBaseModel(BaseModel):
    pass