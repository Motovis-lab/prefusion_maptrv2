import sys
sys.path.append("../../")

from mmengine.config import Config  
from prefusion.registry import MODELS  
from contrib.fastbev_det.models import *
from pathlib import Path as P
import torch

def test_vovnet():
    img_backbone_conf=dict(
        type='VoVNet',
        # model_type="vovnet57",
        # out_indices=[4, 8],
        model_type="vovnet39",
        out_indices=[2, 3, 4, 5],
        base_channels=32,
        # init_cfg=dict(type='Pretrained', checkpoint="./work_dirs/backbone_checkpoint/vovnet57_match.pth")
    )
    img_neck_conf=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 128, 256],
        upsample_strides=[1, 1, 1, 2],
        out_channels=[64, 64, 64, 64],
    )
    front_img_backbone_neck_conf=dict(type='ImgBackboneNeck',
                                         img_backbone_conf=img_backbone_conf,
                                         img_neck_conf=img_neck_conf)

    model = MODELS.build(front_img_backbone_neck_conf).cuda()

    out = model(torch.rand(2, 3, 768, 384).cuda())
    print(out)


def test_resnet():
    img_backbone_conf=dict(
        type='mmdet.ResNet',
        depth=50,
        frozen_stages=0,
        base_channels=16,
        strides=(1, 1, 1, 2),
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained',
                        checkpoint='torchvision://resnet50'),
    )
    img_neck_conf=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256, 512],
        upsample_strides=[1, 1, 1, 2],
        out_channels=[64, 64, 64, 64],
    )
    front_img_backbone_neck_conf=dict(type='ImgBackboneNeck',
                                         img_backbone_conf=img_backbone_conf,
                                         img_neck_conf=img_neck_conf)
    model = MODELS.build(front_img_backbone_neck_conf).cuda()
    out = model(torch.rand(1,3,768, 384).cuda())
    print(out)
