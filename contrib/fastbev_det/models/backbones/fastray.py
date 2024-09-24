import torch
from torch import nn
from mmengine.runner import autocast
from mmengine.model import BaseModule
from prefusion.registry import MODELS
import torch.nn.functional as F
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import mmcv
import pdb

@MODELS.register_module()
class FastRay(BaseModule):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        downsample_factor,
        output_channels,
        img_backbone_conf=None,
        img_neck_conf=None,
        depth_net_conf=None,
        voxel_shape=None, 
        bev_feature_reducer_conf=None,
        each_camera_nums=dict(fish=4, pv=5, front=1),
        mono_depth_channels=256,
        # depth_reducer_conf=None,
        # horiconv_conf=None,
        init_cfg=None
        ):
        super().__init__(init_cfg=init_cfg)
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound  
        self.downsample_factor = downsample_factor
        self.outptu_channels = output_channels
        self.img_backbone = MODELS.build(img_backbone_conf)
        self.img_neck = MODELS.build(img_neck_conf)
        self.depth_net = MODELS.build(depth_net_conf)
        self.bev_feat_reducer = MODELS.build(bev_feature_reducer_conf)
        self.each_camera_nums = each_camera_nums
        self.mono_depth_channels = mono_depth_channels
        self.fastray_vt = MODELS.build(
                                dict(type='fastray_vt', 
                                     voxel_shape=voxel_shape,
                                     each_camera_nums=each_camera_nums
                                     ))
        # self.depth_reducer = MODELS.build(depth_reducer_conf)
        # self.horiconv = MODELS.build(horiconv_conf)

    def forward(self, sweep_infos, is_return_depth=False, is_return_bev=True):
        img_feats = dict()
        img_depth_feats = dict()
        depth_feats = dict()
        mono_depth_feats = dict()
        for key in sweep_infos:
            img_feats[f"{key.split('_')[0]}_feats"] = self.get_cam_feats(sweep_infos[key]['imgs'])
            if 'fish' in key:
                mats_dict = dict(
                    intrinsic=sweep_infos[key]['intrinsic'],
                    extrinsic=sweep_infos[key]['extrinsic'][:, :3, :].view(-1, 12)
                )
            else:
                intrinsic = torch.zeros(sweep_infos[key]['extrinsic'].shape[0], 8).to(sweep_infos[key]['extrinsic'].device)
                intrinsic[:, :4] = sweep_infos[key]['intrinsic']
                mats_dict = dict(
                    intrinsic=intrinsic,
                    extrinsic=sweep_infos[key]['extrinsic'][:, :3, :].view(-1, 12)
                )
            img_depth_feats_, supervised_depth_feat_ = self.depth_net(img_feats[f"{key.split('_')[0]}_feats"],
                                                                       mats_dict, key.split('_')[0])
            img_depth_feats[f"{key.split('_')[0]}_feats"] = img_depth_feats_
            depth_feats[f"{key.split('_')[0]}_feats"] = supervised_depth_feat_
            mono_depth_feats[f"{key.split('_')[0]}_mono_depth_feats"] = img_depth_feats_[:, :self.mono_depth_channels, ...]

        if is_return_bev:
            img_bev_feats = self.fastray_vt(sweep_infos, img_depth_feats)
            img_bev_feats = self.bev_feat_reducer(img_bev_feats)
        else:
            img_bev_feats = None
        if is_return_depth:            
            return dict(bev_img_feats=img_bev_feats, depth_feats=depth_feats, mono_depth_feats=mono_depth_feats)
        else:
            return dict(bev_img_feats=img_bev_feats)

    def get_cam_feats(self, sweep_imgs):
        img_feats = self.img_neck(self.img_backbone(sweep_imgs))[0]
        
        return img_feats