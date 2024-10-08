import torch
import torch.nn.functional as F
from prefusion.registry import MODELS
from torch import nn
from mmengine.runner import autocast
from ..backbones.backbone import BasicBlock
from .utils import Mlp, ASPP, SELayer

@MODELS.register_module()
class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                d_bound = None,
                #  d_bound_fish=[0.1, 5.1, 0.2],  # Categorical Depth bounds and division (m)
                #  d_bound_pv=[0.1, 12.1, 0.2],
                #  d_bound_front=[0.1, 36.1, 0.2],
                 ):
        super(DepthNet, self).__init__()
        self.d_bound = d_bound
        self.depth_channels = int(
            (d_bound[1] - d_bound[0]) / d_bound[2])
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        # self.bn = nn.BatchNorm1d(20)

        self.context_mlp = Mlp(20, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_mlp = Mlp(20, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_super_conv = nn.Conv2d(mid_channels,
                                         self.depth_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            # build_conv_layer(cfg=dict(
            #     type='DCN',
            #     in_channels=mid_channels,
            #     out_channels=mid_channels,
            #     kernel_size=3,
            #     padding=1,
            #     groups=4,
            #     im2col_step=128,
            # )),
        )

    def forward(self, x, intrinsic, extrinsic):
        mlp_input = torch.cat(
            [intrinsic, extrinsic],
            -1,
        )
        # mlp_input = self.bn(mlp_input)
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        supervised_depth = self.depth_super_conv(depth).softmax(dim=1)
        return torch.cat([depth, context], dim=1), supervised_depth
            
    def dummy_forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input)
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return depth.softmax(1), context
