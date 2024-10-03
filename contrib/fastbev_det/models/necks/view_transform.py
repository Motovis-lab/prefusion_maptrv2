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
class fastray_vt(BaseModule):
    def __init__(self, voxel_shape, each_camera_nums, init_cfg: dict | torch.List[dict] | None = None):
        super().__init__(init_cfg)
        self.register_buffer('voxel_feature', torch.zeros(voxel_shape))
        self.each_camera_nums = each_camera_nums
        self.bev_z, self.bev_h, self.bev_w = voxel_shape[:-1]

    def forward(self, sweep_infos, img_feats):
        # TODO: add multi-scale feature fusion
        BN, C, H, W = img_feats['fish_feats'].shape
        B = BN//4
        # C = 3
        with autocast(enabled=False, device_type="cuda"):
            voxel_feature = self.voxel_feature.repeat(B, 1, 1, 1, C).view(B, C, -1)
            for key in img_feats:
                # img = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['imgs'], self.each_camera_nums[key.split('_')[0]], dim=0)
                img_feats_ = torch.split(img_feats[key].float(), self.each_camera_nums[key.split('_')[0]], dim=0)
                valid_ind_map = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['valid_map_sampled'], self.each_camera_nums[key.split('_')[0]], dim=0)
                uu = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['uu'], self.each_camera_nums[key.split('_')[0]], dim=0)
                vv = torch.split(sweep_infos[f"{key.split('_')[0]}_data"]['vv'], self.each_camera_nums[key.split('_')[0]], dim=0)
                for i in range(B):
                    for k in range(self.each_camera_nums[key.split('_')[0]]):
                        uu_ = uu[i][k][valid_ind_map[i][k]]
                        vv_ = vv[i][k][valid_ind_map[i][k]]
                        # img_feats_ = F.interpolate(((img[i][k] - img[i][k].min())).unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)
                        # voxel_feature[i][..., valid_ind_map[i][k]] = img_feats_[..., vv_, uu_]
                        # import matplotlib.pyplot as plt
                        # plt.imshow(img_feats_[0].cpu().numpy().transpose(1,2,0))
                        # plt.savefig(f"./work_dirs/vt_debug/img_{key}_{i}_{k}.jpg")

                        # img_ = (img.cpu().numpy().transpose(1,2,0) - img.cpu().numpy().min()) * 255
                        # mmcv.imwrite(img_, f"./work_dirs/{i}_{key.split('_')[0]}_{k}.jpg")
                        
                        # save u v map
                        # if i==0:
                        #     np.save(f"./work_dirs/vt_debug/uu_{key}_{i}_{k}.npy", uu_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/vv_{key}_{i}_{k}.npy", vv_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/valid_ind_map_{key}_{i}_{k}.npy", valid_ind_map[i][k].cpu().numpy())
                        
                        voxel_feature[i][..., valid_ind_map[i][k]] = img_feats_[i][k][..., vv_, uu_]
            voxel_feature = voxel_feature.view(B, C * self.bev_z, self.bev_h, self.bev_w)
            # B * (C * Z) * H * W
            return voxel_feature