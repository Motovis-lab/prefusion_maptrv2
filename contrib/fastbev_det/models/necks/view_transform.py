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
from torch.onnx.symbolic_helper import parse_args
from torch.nn.modules.utils import _ntuple


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
                        img_feats = F.interpolate(((img[i][k] - img[i][k].min())).unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)[0]
                        img_feats = torch.ones_like(img_feats).to(img_feats.device).to(torch.float32)
                        voxel_feature[i][..., valid_map_sampled[i][k]] += img_feats[..., vv_, uu_] * norm_density_map[i][k][norm_density_map[i][k]!=0]
                        # import matplotlib.pyplot as plt
                        # plt.imshow(img_feats[0].cpu().numpy().transpose(1,2,0))
                        # plt.savefig(f"./work_dirs/vt_debug/img_{N}_{i}_{k}.jpg")

                        # img_ = (img.cpu().numpy().transpose(1,2,0) - img.cpu().numpy().min()) * 255
                        # mmcv.imwrite(img_, f"./work_dirs/{i}_{key.split('_')[0]}_{k}.jpg")
                        
                    else:
                        voxel_feature[i][..., valid_map_sampled[i][k]] += img_feats[i][k][..., vv_, uu_] * norm_density_map[i][k][norm_density_map[i][k]!=0]
                        # save u v map
                        # if i==0:
                        #     np.save(f"./work_dirs/vt_debug/uu_{N}_{i}_{k}.npy", uu_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/vv_{N}_{i}_{k}.npy", vv_.cpu().numpy())
                        #     np.save(f"./work_dirs/vt_debug/dd_{N}_{i}_{k}.npy", vv_.cpu().numpy())
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
    def symbolic(g, input, uu, vv, valid, density, length, outputsizeb_i=1, outputsizec_i=3*6, outputsizeh_i=240, outputsizew_i=120):
        
        return g.op("custom::CustomProject", input, uu, vv, valid, density, voxelnum_i=length, outputsizeb_i=outputsizeb_i, outputsizec_i=outputsizec_i, outputsizeh_i=outputsizeh_i, outputsizew_i=outputsizew_i)

    @staticmethod
    def forward(ctx, input: Tensor, uu: Tensor, vv: Tensor, valid: Tensor, density: Tensor, length):

        return torch.zeros(ProjectPlugin.output_size).cuda()
    
class ProjectPlugin_v2(torch.autograd.Function):
    output_size = None

    @staticmethod
    def set_output_size(output_size):
        ProjectPlugin.output_size = tuple(output_size)

    @staticmethod
    def symbolic(g, input, uu, vv, valid, density, length, outputsizeb_i=1, outputsizec_i=512*6, outputsizeh_i=240, outputsizew_i=120):
        
        return g.op("custom::CustomProject", input, uu, vv, valid, density, voxelnum_i=length, outputsizeb_i=outputsizeb_i, outputsizec_i=outputsizec_i, outputsizeh_i=outputsizeh_i, outputsizew_i=outputsizew_i)

    @staticmethod
    def forward(ctx, input: Tensor, uu: Tensor, vv: Tensor, valid: Tensor, density: Tensor, length):

        return torch.zeros(ProjectPlugin.output_size).cuda()    

class ProjectPlugin_v3(torch.autograd.Function):
    output_size = None

    @staticmethod
    def set_output_size(output_size):
        ProjectPlugin.output_size = tuple(output_size)

    @staticmethod
    def symbolic(g, fish_input, pv_input, front_input, uu, vv, valid, density, length, outputsizeb_i=1, outputsizec_i=64*6, outputsizeh_i=240, outputsizew_i=120):
        
        return g.op("custom::CustomProject", fish_input, pv_input, front_input, uu, vv, valid, density, voxelnum_i=length, outputsizeb_i=outputsizeb_i, outputsizec_i=outputsizec_i, outputsizeh_i=outputsizeh_i, outputsizew_i=outputsizew_i)

    @staticmethod
    def forward(ctx, fish_input: Tensor, pv_input: Tensor, front_input: Tensor, uu: Tensor, vv: Tensor, valid: Tensor, density: Tensor, length):

        return torch.zeros(ProjectPlugin.output_size).cuda()
    

class ProjectPlugin_v4(torch.autograd.Function):
    output_size = None

    @staticmethod
    def set_output_size(output_size):
        ProjectPlugin.output_size = tuple(output_size)

    @staticmethod
    def symbolic(g, fish_input, pv_input, front_input, uu, vv, dd, valid, density, length, outputsizeb_i=1, outputsizec_i=64*6, outputsizeh_i=240, outputsizew_i=120):
        
        return g.op("custom::CustomProject", fish_input, pv_input, front_input, uu, vv, dd, valid, density, voxelnum_i=length, outputsizeb_i=outputsizeb_i, outputsizec_i=outputsizec_i, outputsizeh_i=outputsizeh_i, outputsizew_i=outputsizew_i)

    @staticmethod
    def forward(ctx, fish_input: Tensor, pv_input: Tensor, front_input: Tensor, uu: Tensor, vv: Tensor, dd: Tensor, valid: Tensor, density: Tensor, length):

        return torch.zeros(ProjectPlugin.output_size).cuda()
    


class VoxelProjection_fish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #TODO 加入 C H W 为模型固定参数
        self.plugin = ProjectPlugin()
        self.plugin.set_output_size([1, 2016, 240, 120])

        self.length = 172800

        # self.uu = torch.from_numpy(np.load("work_dirs/vt_debug/fish_uu.npy")).to(torch.float32)
        # self.vv  = torch.from_numpy(np.load("work_dirs/vt_debug/fish_vv.npy")).to(torch.float32)
        # self.valid = torch.from_numpy(np.load("work_dirs/vt_debug/fish_valid_map.npy")).to(torch.float32)
        # self.norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/fish_norm_density_map.npy")).to(torch.float32)

        self.uu = torch.from_numpy(np.load("work_dirs/vt_debug/fish_uu.npy")).to(torch.float16)
        self.vv  = torch.from_numpy(np.load("work_dirs/vt_debug/fish_vv.npy")).to(torch.float16)
        self.valid = torch.from_numpy(np.load("work_dirs/vt_debug/fish_valid_map.npy")).to(torch.float16)
        self.norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/fish_norm_density_map.npy")).to(torch.float16)
        
        
    def forward(self, input):
        img_bev_feats_fish = self.plugin.apply(input, self.uu, self.vv, 
                                                self.valid, self.norm_density_map, self.length
                                                )
        # img_bev_feats_fish = self.plugin.apply(input)
        
        # for k in range(4):
        #     uu_ = self.uu[k][self.valid[k]]
        #     vv_ = self.vv[k][self.valid[k]]
        #     img_bev_feats_fish[0][..., self.valid[k]] = input[k][..., vv_, uu_] * self.norm_density_map[k][self.norm_density_map[k]!=0].float()
        return img_bev_feats_fish
    
    
class VoxelProjection_pv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugin = ProjectPlugin()
        self.plugin.set_output_size([1, 2016, 240, 120])
        
        self.length = 172800
        # self.uu = torch.from_numpy(np.load("work_dirs/vt_debug/pv_uu.npy")).to(torch.float32)
        # self.vv  = torch.from_numpy(np.load("work_dirs/vt_debug/pv_vv.npy")).to(torch.float32)
        # self.valid = torch.from_numpy(np.load("work_dirs/vt_debug/pv_valid_map.npy")).to(torch.float32)
        # self.norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/pv_norm_density_map.npy")).to(torch.float32)
        self.uu = torch.from_numpy(np.load("work_dirs/vt_debug/pv_uu.npy")).to(torch.float16)
        self.vv  = torch.from_numpy(np.load("work_dirs/vt_debug/pv_vv.npy")).to(torch.float16)
        self.valid = torch.from_numpy(np.load("work_dirs/vt_debug/pv_valid_map.npy")).to(torch.float16)
        self.norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/pv_norm_density_map.npy")).to(torch.float16)

        
    def forward(self, input):
        img_bev_feats_fish = self.plugin.apply(input, self.uu, self.vv, 
                                                self.valid, self.norm_density_map, self.length
                                                )
        # img_bev_feats_fish = self.plugin.apply(input)
    
        return img_bev_feats_fish
    
class VoxelProjection_front(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugin = ProjectPlugin()
        self.plugin.set_output_size([1, 2016, 240, 120])
        
        self.length = 172800
        # self.uu = torch.from_numpy(np.load("work_dirs/vt_debug/front_uu.npy")).to(torch.float32)
        # self.vv  = torch.from_numpy(np.load("work_dirs/vt_debug/front_vv.npy")).to(torch.float32)
        # self.valid = torch.from_numpy(np.load("work_dirs/vt_debug/front_valid_map.npy")).to(torch.float32)
        # self.norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/front_norm_density_map.npy")).to(torch.float32)

        self.uu = torch.from_numpy(np.load("work_dirs/vt_debug/front_uu.npy")).to(torch.float16)
        self.vv  = torch.from_numpy(np.load("work_dirs/vt_debug/front_vv.npy")).to(torch.float16)
        self.valid = torch.from_numpy(np.load("work_dirs/vt_debug/front_valid_map.npy")).to(torch.float16)
        self.norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/front_norm_density_map.npy")).to(torch.float16)

        
    def forward(self, input):
        img_bev_feats_fish = self.plugin.apply(input, self.uu, self.vv, 
                                                self.valid, self.norm_density_map, self.length
                                                )
        # img_bev_feats_fish = self.plugin.apply(input)
    
        return img_bev_feats_fish
    

class VoxelProjection(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugin = ProjectPlugin_v3()
        self.plugin.set_output_size([1, 2016, 240, 120])
        
        self.length = 172800
        
        self.fish_uu = torch.from_numpy(np.load("work_dirs/vt_debug/fish_uu.npy")).to(torch.float32)
        self.fish_vv  = torch.from_numpy(np.load("work_dirs/vt_debug/fish_vv.npy")).to(torch.float32)
        self.fish_valid = torch.from_numpy(np.load("work_dirs/vt_debug/fish_valid_map.npy")).to(torch.float32)
        self.fish_norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/fish_norm_density_map.npy")).to(torch.float32)

        self.pv_uu = torch.from_numpy(np.load("work_dirs/vt_debug/pv_uu.npy")).to(torch.float32)
        self.pv_vv  = torch.from_numpy(np.load("work_dirs/vt_debug/pv_vv.npy")).to(torch.float32)
        self.pv_valid = torch.from_numpy(np.load("work_dirs/vt_debug/pv_valid_map.npy")).to(torch.float32)
        self.pv_norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/pv_norm_density_map.npy")).to(torch.float32)
        
        self.front_uu = torch.from_numpy(np.load("work_dirs/vt_debug/front_uu.npy")).to(torch.float32)
        self.front_vv  = torch.from_numpy(np.load("work_dirs/vt_debug/front_vv.npy")).to(torch.float32)
        self.front_valid = torch.from_numpy(np.load("work_dirs/vt_debug/front_valid_map.npy")).to(torch.float32)
        self.front_norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/front_norm_density_map.npy")).to(torch.float32)
        
        self.uu = torch.concat([self.fish_uu, self.pv_uu, self.front_uu], dim=0).to(torch.int32).requires_grad_(False)
        self.vv = torch.concat([self.fish_vv, self.pv_vv, self.front_vv], dim=0).to(torch.int32).requires_grad_(False)
        self.valid = torch.concat([self.fish_valid, self.pv_valid, self.front_valid], dim=0).to(torch.int32).requires_grad_(False)
        self.norm_density_map = torch.concat([self.fish_norm_density_map, self.pv_norm_density_map, self.front_norm_density_map], dim=0).requires_grad_(False)

    def forward(self, fish_input, pv_input, front_input):
        img_bev_feats_fish = self.plugin.apply(fish_input, pv_input, front_input, self.uu, self.vv, 
                                                self.valid, self.norm_density_map, self.length
                                                )
        # img_bev_feats_fish = self.plugin.apply(input)
    
        return img_bev_feats_fish
    
class VoxelProjection_V2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugin = ProjectPlugin_v4()
        self.plugin.set_output_size([1, 2016, 240, 120])
        
        self.length = 172800
        
        self.fish_uu = torch.from_numpy(np.load("work_dirs/vt_debug/fish_uu.npy")).to(torch.float32)
        self.fish_vv  = torch.from_numpy(np.load("work_dirs/vt_debug/fish_vv.npy")).to(torch.float32)
        self.fish_valid = torch.from_numpy(np.load("work_dirs/vt_debug/fish_valid_map.npy")).to(torch.float32)
        self.fish_norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/fish_norm_density_map.npy")).to(torch.float32)

        self.pv_uu = torch.from_numpy(np.load("work_dirs/vt_debug/pv_uu.npy")).to(torch.float32)
        self.pv_vv  = torch.from_numpy(np.load("work_dirs/vt_debug/pv_vv.npy")).to(torch.float32)
        self.pv_valid = torch.from_numpy(np.load("work_dirs/vt_debug/pv_valid_map.npy")).to(torch.float32)
        self.pv_norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/pv_norm_density_map.npy")).to(torch.float32)
        
        self.front_uu = torch.from_numpy(np.load("work_dirs/vt_debug/front_uu.npy")).to(torch.float32)
        self.front_vv  = torch.from_numpy(np.load("work_dirs/vt_debug/front_vv.npy")).to(torch.float32)
        self.front_valid = torch.from_numpy(np.load("work_dirs/vt_debug/front_valid_map.npy")).to(torch.float32)
        self.front_norm_density_map = torch.from_numpy(np.load("work_dirs/vt_debug/front_norm_density_map.npy")).to(torch.float32)
        
        self.uu = torch.concat([self.fish_uu, self.pv_uu, self.front_uu], dim=0).to(torch.int32).requires_grad_(False)
        self.vv = torch.concat([self.fish_vv, self.pv_vv, self.front_vv], dim=0).to(torch.int32).requires_grad_(False)
        self.valid = torch.concat([self.fish_valid, self.pv_valid, self.front_valid], dim=0).to(torch.int32).requires_grad_(False)
        self.norm_density_map = torch.concat([self.fish_norm_density_map, self.pv_norm_density_map, self.front_norm_density_map], dim=0).requires_grad_(False)

    def forward(self, fish_input, pv_input, front_input):
        img_bev_feats_fish = self.plugin.apply(fish_input, pv_input, front_input, self.uu, self.vv, 
                                                self.valid, self.norm_density_map, self.length
                                                )
        # img_bev_feats_fish = self.plugin.apply(input)
    
        return img_bev_feats_fish