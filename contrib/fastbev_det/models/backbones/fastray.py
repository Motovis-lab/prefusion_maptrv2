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
        self.register_buffer('voxel_feature', torch.zeros(voxel_shape))
        self.bev_z, self.bev_h, self.bev_w = voxel_shape[:-1]
        self.each_camera_nums = each_camera_nums
        self.mono_depth_channels = mono_depth_channels
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
            img_bev_feats = self.get_fastbev_feats(sweep_infos, img_depth_feats)
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
    
    @autocast(enabled=False)
    def get_fastbev_feats(self, sweep_infos, img_feats):
        # TODO: add multi-scale feature fusion
        BN, C, H, W = img_feats['fish_feats'].shape
        B = BN//4
        voxel_feature = self.voxel_feature.repeat(B, 1, 1, 1, C).view(B, C, -1)
        
        for key in img_feats:
            # tmp_img = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['imgs'], self.each_camera_nums[key.split('_')[0]], dim=0)
            img_feats_ = torch.split(img_feats[key], self.each_camera_nums[key.split('_')[0]], dim=0)
            valid_ind_map = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['valid_map_sampled'], self.each_camera_nums[key.split('_')[0]], dim=0)
            uu = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['uu'], self.each_camera_nums[key.split('_')[0]], dim=0)
            vv = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['vv'], self.each_camera_nums[key.split('_')[0]], dim=0)
            for i in range(B):
                for k in range(self.each_camera_nums[key.split('_')[0]]):
                    uu_ = uu[i][k][valid_ind_map[i][k]]
                    vv_ = vv[i][k][valid_ind_map[i][k]]
                    # img_feats_ = F.interpolate(((img - img.min()) * 255).unsqueeze(0), scale_factor=1/8, mode='bilinear', align_corners=False)
                    # img_ = (img.cpu().numpy().transpose(1,2,0) - img.cpu().numpy().min()) * 255
                    # mmcv.imwrite(img_, f"./work_dirs/{i}_{key.split('_')[0]}_{k}.jpg")
                    voxel_feature[i][..., valid_ind_map[i][k]] = img_feats_[i][k][..., vv_, uu_]
        voxel_feature = voxel_feature.view(B, C * self.bev_z, self.bev_h, self.bev_w)
        # B * (C * Z) * H * W
        return voxel_feature