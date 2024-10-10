import torch
from torch import nn
from mmengine.runner import autocast
from mmengine.model import BaseModule
from prefusion.registry import MODELS
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import pdb
from torch import Tensor


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
                        
                    else:
                        voxel_feature[i][..., valid_map_sampled[i][k]] = img_feats[i][k][..., vv_, uu_] * norm_density_map[i][k][norm_density_map[i][k]!=0]
                        # save u v map
                        # if i==0:
                        #     np.save(f"./work_dirs/vt_debug/uu_{N}_{i}_{k}.npy", uu_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/vv_{N}_{i}_{k}.npy", vv_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/valid_ind_map_{N}_{i}_{k}.npy", valid_map_sampled[i][k].cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/norm_density_map_{N}_{i}_{k}.npy", norm_density_map[i][k].cpu().numpy())
            voxel_feature = voxel_feature.view(B, C * self.bev_z, self.bev_h, self.bev_w)
            # B * (C * Z) * H * W
            return voxel_feature
    

class ProjectPlugin(torch.autograd.Function):
    output_size = None

    @staticmethod
    def set_output_size(output_size):
        ProjectPlugin.output_size = tuple(output_size)

    @staticmethod
    def symbolic(g, input, projection_u, projection_v, projection_valid, projection_density):
        return g.op("custom::Plugin", input, projection_u, projection_v, projection_valid, projection_density, name_s='Project2Dto3D', info_s='')

    @staticmethod
    def forward(ctx, input: Tensor, projection_u: Tensor, projection_v: Tensor, projection_valid: Tensor, projection_density: Tensor):
        # input: [bs*n_cams, h, w, c]
        # projection: [bs*n_cams, 3, 4]
        # output: [bev_h, bev_w, z, c']
        return torch.ones(ProjectPlugin.output_size).cuda()
    
class VoxelProjection(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugin = ProjectPlugin()
        self.plugin.set_output_size([1, 336 * 6, 240, 120])

        self.fish_uu = torch.nn.Parameter(torch.from_numpy(np.load("work_dirs/vt_debug/fish_uu.npy")), requires_grad=False)
        self.fish_vv  = torch.nn.Parameter(torch.from_numpy(np.load("work_dirs/vt_debug/fish_vv.npy")), requires_grad=False)
        self.fish_valid = torch.nn.Parameter(torch.from_numpy(np.load("work_dirs/vt_debug/fish_valid_map.npy")), requires_grad=False)
        self.fish_norm_density_map = torch.nn.Parameter(torch.from_numpy(np.load("work_dirs/vt_debug/fish_norm_density_map.npy")), requires_grad=False)        

        
    def forward(self, input):
        C, H, W = input.shape[-3:]
        img_bev_feats_fish = self.plugin.apply(input.reshape(4, C, H, W), self.fish_uu, self.fish_vv, 
                                                self.fish_valid, self.fish_norm_density_map
                                                )
    
        return img_bev_feats_fish