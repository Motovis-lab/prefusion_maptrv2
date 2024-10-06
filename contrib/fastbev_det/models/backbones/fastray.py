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
        depth_net_fish_conf=None,
        depth_net_pv_conf=None,
        depth_net_front_conf=None,
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
        self.img_backbone_fish = MODELS.build(img_backbone_conf)
        self.img_backbone_pv = MODELS.build(img_backbone_conf)
        self.img_backbone_front = MODELS.build(img_backbone_conf)
        
        self.img_neck_fish = MODELS.build(img_neck_conf)
        self.img_neck_pv = MODELS.build(img_neck_conf)
        self.img_neck_front = MODELS.build(img_neck_conf)

        self.depth_net_fish = MODELS.build(depth_net_fish_conf)
        self.depth_net_pv = MODELS.build(depth_net_pv_conf)
        self.depth_net_front = MODELS.build(depth_net_front_conf)
        self.bev_feat_reducer = MODELS.build(bev_feature_reducer_conf)
        self.each_camera_nums = each_camera_nums
        self.mono_depth_channels = mono_depth_channels
        self.fastray_vt_fish = MODELS.build(
                                dict(type='fastray_vt', 
                                     voxel_shape=voxel_shape,
                                     ))
        self.fastray_vt_pv = MODELS.build(
                                dict(type='fastray_vt', 
                                     voxel_shape=voxel_shape,
                                     ))
        self.fastray_vt_front = MODELS.build(
                                dict(type='fastray_vt', 
                                     voxel_shape=voxel_shape,
                                     ))
        # self.depth_reducer = MODELS.build(depth_reducer_conf)
        # self.horiconv = MODELS.build(horiconv_conf)

    def forward(self, sweep_infos, is_return_depth=False, is_return_bev=True):
        img_feats = dict()
        img_depth_feats = dict()
        depth_feats = dict()
        mono_depth_feats = dict()
        B = sweep_infos["fish_data"]['imgs'].shape[0] // 4

        img_feats_fish = self.get_cam_feats_fish(sweep_infos["fish_data"]['imgs'])
        img_feats_pv = self.get_cam_feats_pv(sweep_infos["pv_data"]['imgs'])
        img_feats_front = self.get_cam_feats_front(sweep_infos["front_data"]['imgs'])

        intrinsic=sweep_infos["fish_data"]['intrinsic'].view(-1, 8)
        extrinsic=sweep_infos["fish_data"]['extrinsic'][:, :3, :].view(-1, 12)
        img_depth_feats_fish, supervised_depth_feat_fish = self.depth_net_fish(img_feats_fish, intrinsic, extrinsic)

        intrinsic = torch.zeros(sweep_infos["pv_data"]['extrinsic'].shape[0], 8).to(sweep_infos["pv_data"]['extrinsic'].device)
        intrinsic[:, :4] = sweep_infos["pv_data"]['intrinsic']
        extrinsic=sweep_infos["pv_data"]['extrinsic'][:, :3, :].view(-1, 12)
        img_depth_feats_pv, supervised_depth_feat_pv = self.depth_net_pv(img_feats_pv, intrinsic, extrinsic)

        intrinsic = torch.zeros(sweep_infos["front_data"]['extrinsic'].shape[0], 8).to(sweep_infos["front_data"]['extrinsic'].device)
        intrinsic[:, :4] = sweep_infos["front_data"]['intrinsic']
        extrinsic=sweep_infos["front_data"]['extrinsic'][:, :3, :].view(-1, 12)
        img_depth_feats_front, supervised_depth_feat_front = self.depth_net_front(img_feats_front, intrinsic, extrinsic)

        depth_feats = dict(
            fish_feats = supervised_depth_feat_fish,
            pv_feats   = supervised_depth_feat_pv,
            front_feats = supervised_depth_feat_front
        )
        if is_return_bev:
            C, H, W = img_depth_feats_fish.shape[-3:]
            img_bev_feats_fish = self.fastray_vt_fish(sweep_infos['fish_data']['uu'], sweep_infos['fish_data']['vv'], 
                                                 sweep_infos['fish_data']['valid_map_sampled'], img_depth_feats_fish.reshape(B, -1, C, H, W),
                                                 # img=sweep_infos['fish_data']['imgs']
                                                 )
            C, H, W = img_depth_feats_pv.shape[-3:]
            img_bev_feats_pv = self.fastray_vt_pv(sweep_infos['pv_data']['uu'], sweep_infos['pv_data']['vv'], 
                                                 sweep_infos['pv_data']['valid_map_sampled'], img_depth_feats_pv.reshape(B, -1, C, H, W),
                                                 # img=sweep_infos['pv_data']['imgs']
                                                 )
            C, H, W = img_depth_feats_front.shape[-3:]
            img_bev_feats_front = self.fastray_vt_front(sweep_infos['front_data']['uu'], sweep_infos['front_data']['vv'], 
                                                 sweep_infos['front_data']['valid_map_sampled'], img_depth_feats_front.reshape(B, -1, C, H, W),
                                                 # img=sweep_infos['front_data']['imgs']
                                                 )
            
            img_bev_feats = img_bev_feats_fish + img_bev_feats_pv + img_bev_feats_front
            img_bev_feats = self.bev_feat_reducer(img_bev_feats)
        else:
            img_bev_feats = None
        if is_return_depth:            
            return dict(bev_img_feats=img_bev_feats, depth_feats=depth_feats, mono_depth_feats=mono_depth_feats)
        else:
            return dict(bev_img_feats=img_bev_feats)

    def get_cam_feats_fish(self, sweep_imgs):
        img_feats = self.img_neck_fish(self.img_backbone_fish(sweep_imgs))[0]
        
        return img_feats

    def get_cam_feats_pv(self, sweep_imgs):
        img_feats = self.img_neck_pv(self.img_backbone_pv(sweep_imgs))[0]
        
        return img_feats
    
    def get_cam_feats_front(self, sweep_imgs):
        img_feats = self.img_neck_front(self.img_backbone_front(sweep_imgs))[0]
        
        return img_feats