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
    def __init__(self, voxel_shape, init_cfg: dict | torch.List[dict] | None = None):
        super().__init__(init_cfg)
        self.register_buffer('voxel_feature', torch.zeros(voxel_shape))
        self.bev_z, self.bev_h, self.bev_w = voxel_shape[:-1]

    def forward(self, uu, vv, valid_map_sampled, norm_density_map, img_feats, img = None):
        # TODO: add multi-scale feature fusion
        B, N, C = img_feats.shape[:3]
        uu = uu.reshape(B, N, -1)
        vv = vv.reshape(B, N, -1)
        valid_map_sampled = valid_map_sampled.reshape(B, N, -1)
        norm_density_map = norm_density_map.reshape(B, N, -1)
        if img is not None:
            C = 3
            H, W = img.shape[-2:]
            img = img.reshape(B, N, 3, H, W)
        with autocast(enabled=False, device_type="cuda"):
            voxel_feature = self.voxel_feature.repeat(B, 1, 1, 1, C).view(B, C, -1)
            for i in range(B):
                for k in range(N):
                    uu_ = uu[i][k][valid_map_sampled[i][k]]
                    vv_ = vv[i][k][valid_map_sampled[i][k]]
                    if img is not None:
                        img_feats = F.interpolate(((img[i][k] - img[i][k].min())).unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)
                        voxel_feature[i][..., valid_map_sampled[i][k]] = img_feats[..., vv_, uu_] * norm_density_map[i][k][norm_density_map[i][k]!=0]
                        import matplotlib.pyplot as plt
                        plt.imshow(img_feats[0].cpu().numpy().transpose(1,2,0))
                        plt.savefig(f"./work_dirs/vt_debug/img_{N}_{i}_{k}.jpg")

                        # img_ = (img.cpu().numpy().transpose(1,2,0) - img.cpu().numpy().min()) * 255
                        # mmcv.imwrite(img_, f"./work_dirs/{i}_{key.split('_')[0]}_{k}.jpg")
                        
                        # save u v map
                        # if i==0:
                        #     np.save(f"./work_dirs/vt_debug/uu_{key}_{i}_{k}.npy", uu_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/vv_{key}_{i}_{k}.npy", vv_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/valid_ind_map_{key}_{i}_{k}.npy", valid_ind_map[i][k].cpu().numpy())
                    else:
                        voxel_feature[i][..., valid_map_sampled[i][k]] = img_feats[i][k][..., vv_, uu_] * norm_density_map[i][k][norm_density_map[i][k]!=0]
            voxel_feature = voxel_feature.view(B, C * self.bev_z, self.bev_h, self.bev_w)
            # B * (C * Z) * H * W
            return voxel_feature