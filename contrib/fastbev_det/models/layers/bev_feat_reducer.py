from prefusion.registry import MODELS
from mmengine.model import BaseModule
from torch import nn as nn
from ..backbones.vovnet import OSA

@MODELS.register_module()
class BEV_Feat_Reducer(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to
            [dict(type='Kaiming', layer='ConvTranspose2d'),
             dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)].
    """

    def __init__(self,
                 in_channels=80,
                 out_channels=80,
                 init_cfg=[
                     dict(type='Kaiming', layer='ConvTranspose2d'),
                     dict(
                         type='Constant',
                         layer='NaiveSyncBatchNorm2d',
                         val=1.0)
                 ]):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(BEV_Feat_Reducer, self).__init__(init_cfg=init_cfg)

        self.reducer = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, bev_feature):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        out = self.reducer(bev_feature)
        return out