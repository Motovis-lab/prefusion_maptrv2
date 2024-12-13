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
from ..necks import VoxelProjection_fish, VoxelProjection_pv, VoxelProjection_front, VoxelProjection, VoxelProjection_V2

__all__ = ['FastRay_DP_v2', 'FastRay_DP', 'FastRay_DP_v3','FastRay']


@MODELS.register_module()
class FastRay(BaseModule):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        downsample_factor,
        output_channels,
        fish_img_backbone_neck_conf=None,
        pv_img_backbone_neck_conf=None,
        front_img_backbone_neck_conf=None,
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

        self.img_backbone_neck_fish = MODELS.build(fish_img_backbone_neck_conf)
        self.img_backbone_neck_pv = MODELS.build(pv_img_backbone_neck_conf)
        self.img_backbone_neck_front = MODELS.build(front_img_backbone_neck_conf)

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
                                                 sweep_infos['fish_data']['valid_map'], sweep_infos['fish_data']['norm_density_map'].float(), img_depth_feats_fish.reshape(B, -1, C, H, W),
                                                #  img=sweep_infos['fish_data']['imgs']
                                                 )
            C, H, W = img_depth_feats_pv.shape[-3:]
            img_bev_feats_pv = self.fastray_vt_pv(sweep_infos['pv_data']['uu'], sweep_infos['pv_data']['vv'], 
                                                 sweep_infos['pv_data']['valid_map'], sweep_infos['pv_data']['norm_density_map'].float(), img_depth_feats_pv.reshape(B, -1, C, H, W),
                                                #  img=sweep_infos['pv_data']['imgs']
                                                 )
            C, H, W = img_depth_feats_front.shape[-3:]
            img_bev_feats_front = self.fastray_vt_front(sweep_infos['front_data']['uu'], sweep_infos['front_data']['vv'], 
                                                 sweep_infos['front_data']['valid_map'], sweep_infos['front_data']['norm_density_map'].float(), img_depth_feats_front.reshape(B, -1, C, H, W),
                                                #  img=sweep_infos['front_data']['imgs']
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
        img_feats = self.img_backbone_neck_fish(sweep_imgs)[0]
        
        return img_feats

    def get_cam_feats_pv(self, sweep_imgs):
        img_feats = self.img_backbone_neck_pv(sweep_imgs)[0]
        
        return img_feats
    
    def get_cam_feats_front(self, sweep_imgs):
        img_feats = self.img_backbone_neck_front(sweep_imgs)[0]
        
        return img_feats


@MODELS.register_module()
class FastRay_DP(FastRay):
    def __init__(self, **kwargs):
        super(FastRay_DP, self).__init__(**kwargs)
        self.plugin_fish = VoxelProjection_fish()
        self.plugin_pv = VoxelProjection_pv()
        self.plugin_front = VoxelProjection_front()
        self.dummy_conv = nn.Conv2d(3, 3, 1, 1)
        with torch.no_grad():
            self.dummy_conv.weight = nn.Parameter(torch.eye(3).view(3, 3, 1, 1))
            self.dummy_conv.bias = nn.Parameter(torch.zeros(3))

    def forward(self, fish_data_imgs, pv_data_imgs, front_data_imgs,
                fish_intrinsic, fish_extrinsic,
                pv_intrinsic, pv_extrinsic,
                front_intrinsic, front_extrinsic
        ):
        img_feats_fish = self.get_cam_feats_fish(fish_data_imgs)
        img_feats_pv = self.get_cam_feats_pv(pv_data_imgs)
        img_feats_front = self.get_cam_feats_front(front_data_imgs)

        img_depth_feats_fish, supervised_depth_feat_fish = self.depth_net_fish(img_feats_fish, fish_intrinsic, fish_extrinsic)

        img_depth_feats_pv, supervised_depth_feat_pv = self.depth_net_pv(img_feats_pv, pv_intrinsic, pv_extrinsic)

        img_depth_feats_front, supervised_depth_feat_front = self.depth_net_front(img_feats_front, front_intrinsic, front_extrinsic)

        # C, H, W = img_depth_feats_fish.shape[-3:]
        img_bev_feats_fish = self.plugin_fish(img_depth_feats_fish
                                                )
        # C, H, W = img_depth_feats_pv.shape[-3:]
        img_bev_feats_pv = self.plugin_pv(img_depth_feats_pv
                                                )
        # C, H, W = img_depth_feats_front.shape[-3:]
        img_bev_feats_front = self.plugin_front(img_depth_feats_front
                                                )
        
        img_bev_feats = img_bev_feats_fish + img_bev_feats_pv + img_bev_feats_front
        img_bev_feats = self.bev_feat_reducer(img_bev_feats)
        
        return img_bev_feats
    
    def pure_forward(self, img_feats_fish, img_feats_pv, img_feats_front
        ):
        # img_feats_fish =  img_feats_fish.permute(0,2,3,1)
        img_bev_feats_fish = self.plugin_fish(self.dummy_conv(img_feats_fish)
                                                )
        # img_feats_pv = img_feats_pv.permute(0,2,3,1)
        img_bev_feats_pv = self.plugin_pv(self.dummy_conv(img_feats_pv)
                                                )
        # img_feats_front = img_feats_front.permute(0,2,3,1)
        img_bev_feats_front = self.plugin_front(self.dummy_conv(img_feats_front)
                                                )
        
        img_bev_feats = img_bev_feats_fish + img_bev_feats_pv + img_bev_feats_front
        
        # img_bev_feats = img_feats_fish + img_feats_pv + img_feats_front
        
        # img_bev_feats = self.bev_feat_reducer(img_bev_feats)
        # return img_bev_feats.permute(0,3,1,2)
        return img_bev_feats
    

@MODELS.register_module()
class FastRay_DP_v2(FastRay):
    def __init__(self, **kwargs):
        super(FastRay_DP_v2, self).__init__(**kwargs)
        self.plugin = VoxelProjection()
        
        self.dummy_conv = nn.Conv2d(3, 64, 1, 1)
        # with torch.no_grad():
        #     self.dummy_conv.weight = nn.Parameter(torch.eye(3).view(3, 3, 1, 1))
        #     self.dummy_conv.bias = nn.Parameter(torch.zeros(3))

    def forward(self, fish_data_imgs, pv_data_imgs, front_data_imgs,
                fish_intrinsic, fish_extrinsic,
                pv_intrinsic, pv_extrinsic,
                front_intrinsic, front_extrinsic
        ):
        img_feats_fish = self.get_cam_feats_fish(fish_data_imgs)
        img_feats_pv = self.get_cam_feats_pv(pv_data_imgs)
        img_feats_front = self.get_cam_feats_front(front_data_imgs)

        img_depth_feats_fish, supervised_depth_feat_fish = self.depth_net_fish(img_feats_fish, fish_intrinsic, fish_extrinsic)

        img_depth_feats_pv, supervised_depth_feat_pv = self.depth_net_pv(img_feats_pv, pv_intrinsic, pv_extrinsic)

        img_depth_feats_front, supervised_depth_feat_front = self.depth_net_front(img_feats_front, front_intrinsic, front_extrinsic)

        # C, H, W = img_depth_feats_fish.shape[-3:]
        img_bev_feats = self.plugin(img_depth_feats_fish, img_depth_feats_pv, img_depth_feats_front
                                                )
    
        img_bev_feats = self.bev_feat_reducer(img_bev_feats)
        
        return img_bev_feats
    
    def pure_forward(self, img_feats_fish, img_feats_pv, img_feats_front
        ):
        img_bev_feats = self.plugin(self.dummy_conv(img_feats_fish).permute(0,2,3,1), self.dummy_conv(img_feats_pv).permute(0,2,3,1), self.dummy_conv(img_feats_front).permute(0,2,3,1)
                                                )
        
        return img_bev_feats
    

@MODELS.register_module()
class FastRay_DP_v3(FastRay):
    def __init__(self, **kwargs):
        super(FastRay_DP_v3, self).__init__(**kwargs)
        self.plugin = VoxelProjection_V2()
        
        self.dummy_conv = nn.Conv2d(3, 64, 1, 1)
        # with torch.no_grad():
        #     self.dummy_conv.weight = nn.Parameter(torch.eye(3).view(3, 3, 1, 1))
        #     self.dummy_conv.bias = nn.Parameter(torch.zeros(3))

    def forward(self, fish_data_imgs, pv_data_imgs, front_data_imgs,
                fish_intrinsic, fish_extrinsic,
                pv_intrinsic, pv_extrinsic,
                front_intrinsic, front_extrinsic
        ):
        img_feats_fish = self.get_cam_feats_fish(fish_data_imgs)
        img_feats_pv = self.get_cam_feats_pv(pv_data_imgs)
        img_feats_front = self.get_cam_feats_front(front_data_imgs)

        img_depth_feats_fish, supervised_depth_feat_fish = self.depth_net_fish(img_feats_fish, fish_intrinsic, fish_extrinsic)

        img_depth_feats_pv, supervised_depth_feat_pv = self.depth_net_pv(img_feats_pv, pv_intrinsic, pv_extrinsic)

        img_depth_feats_front, supervised_depth_feat_front = self.depth_net_front(img_feats_front, front_intrinsic, front_extrinsic)

        # C, H, W = img_depth_feats_fish.shape[-3:]
        img_bev_feats = self.plugin(img_depth_feats_fish, img_depth_feats_pv, img_depth_feats_front
                                                )
    
        img_bev_feats = self.bev_feat_reducer(img_bev_feats)
        
        return img_bev_feats
    
    def pure_forward(self, img_feats_fish, img_feats_pv, img_feats_front, depth_fish, depth_pv, depth_front
        ):
        img_bev_feats = self.plugin(self.dummy_conv(img_feats_fish).permute(0,2,3,1), self.dummy_conv(img_feats_pv).permute(0,2,3,1), self.dummy_conv(img_feats_front).permute(0,2,3,1), \
                                    depth_fish.permute(0,2,3,1), depth_pv.permute(0,2,3,1), depth_front.permute(0,2,3,1)
                                    )
        
        return img_bev_feats