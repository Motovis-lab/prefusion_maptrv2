import torch
from torch import nn
from mmengine.runner import autocast
from mmengine.model import BaseModule
from mmengine.registry import MODELS

@MODELS.register_module()
class DepthReducer(BaseModule):

    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, feat, depth):
        vert_weight = self.vertical_weighter(feat).softmax(2)  # [N,1,H,W]
        depth = (depth * vert_weight).sum(2)
        return depth

@MODELS.register_module()
class Mono_DepthReducer(BaseModule):

    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, depth):
        return self.vertical_weighter(depth)