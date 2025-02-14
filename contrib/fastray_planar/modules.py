import math
import os
from functools import reduce

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from prefusion.registry import MODELS


__all__ = [
    "ConvBN",
    "Concat",
    "EltwiseAdd",
    "OSABlock",
    "VoVNetFPN",
    "FastRaySpatialTransform",
    "VoxelTemporalAlign",
    "VoxelStreamFusion",
    "VoVNetEncoder",
    "PlanarHead"
]


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 has_relu=True,
                 relu6=False):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.001)
        self.has_relu = has_relu
        if self.has_relu:
            if relu6:
                self.relu = nn.ReLU6(inplace=True)
            else:
                self.relu = nn.ReLU(inplace=True)
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.has_relu:
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        if hasattr(self, 'bn'):  # IMPORTANT! PREPARED FOR BN FUSION, SINCE BN WILL BE DELETED AFTER FUSED
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


@MODELS.register_module()
class EltwiseAdd(nn.Module):
    def __init__(self):
        super(EltwiseAdd, self).__init__()

    def forward(self, *inputs):
        # return torch.add(*inputs)
        return reduce(lambda x, y: torch.add(x, y), [i for i in inputs])


class OSABlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels=None,
                 stride=2,
                 dilation=1,
                 repeat=5,
                 final_dilation=1,
                 with_reduce=True,
                 has_bn=True,
                 relu6=False):
        super(OSABlock, self).__init__()
        assert stride in [1, 2]
        assert repeat >= 2
        self.repeat = repeat
        self.with_reduce = with_reduce

        self.conv1 = ConvBN(in_channels, mid_channels, stride=stride, padding=dilation, dilation=dilation, relu6=relu6)

        for i in range(repeat - 2):
            self._modules['conv{}'.format(i + 2)] = ConvBN(
                mid_channels, mid_channels, padding=dilation, dilation=dilation, relu6=relu6
            )

        self._modules['conv{}'.format(repeat)] = ConvBN(
            mid_channels, mid_channels, padding=final_dilation, dilation=final_dilation, relu6=relu6
        )

        self.concat = Concat()
        if with_reduce:
            assert out_channels is not None
            if has_bn:
                self.reduce = ConvBN(mid_channels * repeat, out_channels, kernel_size=1, padding=0, relu6=relu6)
            else:
                self.reduce = nn.Conv2d(mid_channels * repeat, out_channels, kernel_size=1, padding=0)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        layers = []
        for i in range(self.repeat):
            x = self._modules['conv{}'.format(i + 1)](x)
            layers.append(x)
        x = self.concat(*layers)
        if self.with_reduce:
            x = self.reduce(x)
        return x


@MODELS.register_module()
class VoVNetFPN(BaseModule):
    def __init__(self, out_stride=8, out_channels=128, init_cfg=None, relu6=False, last_bn=False):
        super().__init__(init_cfg=init_cfg)
        self.strides = [4, 8, 16, 32]
        assert out_stride in self.strides
        self.out_stride = out_stride

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2, relu6=relu6)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3, relu6=relu6)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2, relu6=relu6)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2, relu6=relu6)

        # NECK
        if self.out_stride <= 16:
            self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
            self.p4_fusion = Concat()
        if self.out_stride <= 8:
            self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0, relu6=relu6)
            self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
            self.p3_fusion = Concat()
        if self.out_stride <= 4:
            self.p2_linear = ConvBN(256, 96, kernel_size=1, padding=0, relu6=relu6)
            self.p2_up = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0, bias=False)
            self.p2_fusion = Concat()
        
        in_channels = {4: 192, 8: 256, 16: 384, 32: 192}
        mid_channels = {4: 96, 8: 96, 16: 128, 32: 192}
        self.out = OSABlock(
            in_channels[self.out_stride], mid_channels[self.out_stride], out_channels,
            stride=1, repeat=3, has_bn=last_bn, relu6=relu6
        )
        
        
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1) # shape: (bs, 96, 64, 176)
        osa3 = self.osa3(osa2) # shape: (bs, 128, 32, 88)
        osa4 = self.osa4(osa3) # shape: (bs, 192, 16, 44)
        osa5 = self.osa5(osa4) # shape: (bs, 192, 8, 22)

        if self.out_stride <= 32:
            out = osa5
        if self.out_stride <= 16:
            out = self.p4_fusion(self.p4_up(out), osa4)
        if self.out_stride <= 8:
            out = self.p3_fusion(self.p3_up(self.p3_linear(out)), osa3)
        if self.out_stride <= 4:
            out = self.p2_fusion(self.p2_up(self.p2_linear(out)), osa2)
        
        out = self.out(out)
        
        return out




@MODELS.register_module()
class VoVNetSlimFPN(BaseModule):
    def __init__(self, out_channels=80, init_cfg=None, relu6=False):
        super().__init__(init_cfg=init_cfg)

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2, relu6=relu6)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3, relu6=relu6)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2, relu6=relu6)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2, relu6=relu6)

        # NECK
        self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
        self.p4_fusion = Concat()
        self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0, relu6=relu6)
        self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.p3_fusion = Concat()
        self.out = OSABlock(256, 96, stride=1, repeat=3, has_bn=False, with_reduce=False)
        self.up_linear = ConvBN(288, out_channels, kernel_size=1, padding=0, relu6=relu6)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.relu6 = relu6
        if self.relu6:
            self.out_bn = nn.BatchNorm2d(out_channels)
            self.out_relu = nn.ReLU6(inplace=True)
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1)
        osa3 = self.osa3(osa2)
        osa4 = self.osa4(osa3)
        osa5 = self.osa5(osa4)

        p4 = self.p4_fusion(self.p4_up(osa5), osa4)
        p3 = self.p3_fusion(self.p3_up(self.p3_linear(p4)), osa3)
        
        out = self.up(self.up_linear(self.out(p3)))

        if self.relu6:
            out = self.out_relu(self.out_bn(out))
        return out


@MODELS.register_module()
class ResNetFPN(BaseModule):
    def __init__(
        self, 
        out_stride=8, 
        out_channels=128, 
        fpn_lateral_channel=64, 
        fpn_in_channels=(256, 512, 1024, 2048), 
        init_cfg=None, 
        **backbone_kwargs
    ):
        super().__init__(init_cfg=init_cfg)
        self.strides = [4, 8, 16, 32]
        assert out_stride in self.strides
        self.out_stride = out_stride

        # BACKBONE
        backbone_kwargs["type"] = "mmdet.ResNet"
        self.backbone = MODELS.build(backbone_kwargs)

        # NECK (FPN)
        if self.out_stride <= 16:
            self.p4_up = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0, bias=False)
            self.p4_fusion = Concat()
        if self.out_stride <= 8:
            self.p3_linear = ConvBN(2048, 512, kernel_size=1, padding=0)
            self.p3_up = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False)
            self.p3_fusion = Concat()
        if self.out_stride <= 4:
            self.p2_linear = ConvBN(1024, 256, kernel_size=1, padding=0)
            self.p2_up = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False)
            self.p2_fusion = Concat()
        
        in_channels = {4: 512, 8: 1024, 16: 2048, 32: 2048}
        mid_channels = {4: 128, 8: 256, 16: 512, 32: 1024}
        self.out = OSABlock(
            in_channels[self.out_stride], mid_channels[self.out_stride], out_channels,
            stride=1, repeat=3, has_bn=False
        )

        
    def forward(self, x):  # x: (N, 3, H, W)
        feats = self.backbone(x) # feats shape: [bs, 256, 64, 176], [bs, 512, 32, 88], [bs, 1024, 16, 44], [bs, 2048, 8, 22]
        # fpn_feats = self.fpn(feats)
        # out_feat_layer_idx = int(math.log2(self.out_stride) - 2)
        # out = fpn_feats[out_feat_layer_idx]

        if self.out_stride <= 32:
            out = feats[3]
        if self.out_stride <= 16:
            out = self.p4_fusion(self.p4_up(out), feats[2])
        if self.out_stride <= 8:
            out = self.p3_fusion(self.p3_up(self.p3_linear(out)), feats[1])
        if self.out_stride <= 4:
            out = self.p2_fusion(self.p2_up(self.p2_linear(out)), feats[0])
        
        out = self.out(out)
        
        return out



@MODELS.register_module()
class FastRaySpatialTransform(BaseModule):
    
    def __init__(self, 
                 voxel_shape, 
                 fusion_mode='weighted', 
                 bev_mode=False, 
                 reduce_channels=False,
                 in_channels=None,
                 out_channels=None,
                 dump_voxel_feats=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        assert fusion_mode in ['weighted', 'sampled', 'bilinear_weighted']
        self.fusion_mode = fusion_mode
        self.bev_mode = bev_mode
        self.dump_voxel_feats = dump_voxel_feats
        self.reduce_channels = reduce_channels and bev_mode
        if self.reduce_channels:
            self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, camera_feats_dict, camera_lookups):
        '''Output a 3d voxel tensor from 2d image features
        
        Parameters
        ----------
        camera_feats_dict : Dict[str, torch.Tensor]
            camera feature tensor of shape (N, C, H, W)
        camera_lookups : List[Dict[str, torch.Tensor]]
            camera lookup tensors of shape Z*X*Y
        
        Returns
        -------
        voxel_feats : torch.Tensor
            voxel features of shape (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        '''
        cam_ids = list(camera_feats_dict.keys())
        # cat camera features
        camera_feats = []
        for cam_id in cam_ids:
            camera_feats.append(camera_feats_dict[cam_id])
        # get sizes
        Z, X, Y = self.voxel_shape
        N, C = camera_feats[0].shape[0:2]
        # initialize voxel features, (N, C, Z*X*Y)
        voxel_feats = torch.zeros(N, C, Z*X*Y, 
                                  dtype=camera_feats[0].dtype,
                                  device=camera_feats[0].device)
        # iterate over batch
        for n in range(N):
            # iterate over cameras
            for b, cam_id in enumerate(cam_ids):
                camera_feat = camera_feats[b][n]  # C, H, W
                camera_lookup = camera_lookups[n][cam_id]
                if self.fusion_mode == 'weighted':
                    valid_map = camera_lookup['valid_map']
                    valid_uu = camera_lookup['uu'][valid_map]
                    valid_vv = camera_lookup['vv'][valid_map]
                    valid_norm_density_map = camera_lookup['norm_density_map'][valid_map]
                    voxel_feats[n, :, valid_map] += valid_norm_density_map[None] * camera_feat[:, valid_vv, valid_uu]
                if self.fusion_mode == 'sampled':
                    valid_map = camera_lookup['valid_map_sampled']
                    valid_uu = camera_lookup['uu'][valid_map]
                    valid_vv = camera_lookup['vv'][valid_map]
                    voxel_feats[n, :, valid_map] = camera_feat[:, valid_vv, valid_uu]
                if self.fusion_mode == 'bilinear_weighted':
                    valid_map = camera_lookup['valid_map_bilinear']
                    valid_norm_density_map = camera_lookup['norm_density_map'][valid_map]
                    valid_uu_floor = camera_lookup['uu_floor'][valid_map]
                    valid_uu_ceil = camera_lookup['uu_ceil'][valid_map]
                    valid_vv_floor = camera_lookup['vv_floor'][valid_map]
                    valid_vv_ceil = camera_lookup['vv_ceil'][valid_map]
                    valid_uu_bilinear_weight = camera_lookup['uu_bilinear_weight'][valid_map].to(camera_feat.dtype)
                    valid_vv_bilinear_weight = camera_lookup['vv_bilinear_weight'][valid_map].to(camera_feat.dtype)
                    interpolated_camera_feat = valid_uu_bilinear_weight[None] * (
                        camera_feat[:, valid_vv_floor, valid_uu_floor] * valid_vv_bilinear_weight[None] +
                        camera_feat[:, valid_vv_ceil, valid_uu_floor] * (1 - valid_vv_bilinear_weight)[None]
                    ) + (1 - valid_uu_bilinear_weight[None]) * (
                        camera_feat[:, valid_vv_floor, valid_uu_ceil] * valid_vv_bilinear_weight[None] +
                        camera_feat[:, valid_vv_ceil, valid_uu_ceil] * (1 - valid_vv_bilinear_weight)[None]
                    )
                    voxel_feats[n, :, valid_map] += valid_norm_density_map[None] * interpolated_camera_feat
                    
        # reshape voxel_feats
        if self.bev_mode:
            voxel_feats = voxel_feats.reshape(N, C*Z, X, Y)
            if self.reduce_channels:
                bev_feats = self.channel_reduction(voxel_feats)
                if self.dump_voxel_feats:
                    return bev_feats, voxel_feats
                return bev_feats
            return voxel_feats
        else:
            return voxel_feats.reshape(N, C, Z, X, Y)
        
            
@MODELS.register_module()
class VoxelTemporalAlign(BaseModule):
    
    def __init__(self,
                 voxel_shape,
                 voxel_range,
                 bev_mode=False,
                 interpolation='bilinear',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range
        self.bev_mode = bev_mode
        self.interpolation = interpolation
        self.voxel_intrinsics = self._get_voxel_intrinsics(voxel_shape, voxel_range)
    
    @staticmethod
    def _get_voxel_intrinsics(voxel_shape, voxel_range):
        Z, X, Y = voxel_shape    
        fx = X / (voxel_range[1][1] - voxel_range[1][0])
        fy = Y / (voxel_range[2][1] - voxel_range[2][0])
        fz = Z / (voxel_range[0][1] - voxel_range[0][0])
        cx = - voxel_range[1][0] * fx - 0.5
        cy = - voxel_range[2][0] * fy - 0.5
        cz = - voxel_range[0][0] * fz - 0.5
        return cx, cy, cz, fx, fy, fz


    def _unproject_points_from_voxel_to_ego(self):
        Z, X, Y = self.voxel_shape
        cx, cy, cz, fx, fy, fz = self.voxel_intrinsics
        if self.bev_mode:
            xx, yy = torch.meshgrid(torch.arange(X), torch.arange(Y), indexing='ij')
            xx_ego = (xx - cx) / fx
            yy_ego = (yy - cy) / fy
            return torch.stack([
                xx_ego.reshape(-1), 
                yy_ego.reshape(-1),
            ], dim=0)
        else:
            zz, xx, yy = torch.meshgrid(torch.arange(Z), torch.arange(X), torch.arange(Y), indexing='ij')
            xx_ego = (xx - cx) / fx
            yy_ego = (yy - cy) / fy
            zz_ego = (zz - cz) / fz
            return torch.stack([
                xx_ego.reshape(-1), 
                yy_ego.reshape(-1),
                zz_ego.reshape(-1),
            ], dim=0)


    def _project_points_from_ego_to_voxel(self, ego_points, normalize=True):
        """Convert points from ego coords to voxel coords.

        Parameters
        ----------
        ego_points : torch.Tensor
            shape should be (N, 2, X * Y) or (N, 3, Z * X * Y)
        
        normalize : bool
            whether to normalize the projected points to [-1, 1]
            
        Return
        ------
        output : torch.Tensor
            shape should be (N, X, Y, 2) or (N, Z, X, Y, 3)
        
        """
        Z, X, Y = self.voxel_shape
        N, _, _ = ego_points.shape
        cx, cy, cz, fx, fy, fz = self.voxel_intrinsics
        if self.bev_mode:
            xx_egos = ego_points[:, 0]
            yy_egos = ego_points[:, 1]
            xx_ = xx_egos * fx + cx
            yy_ = yy_egos * fy + cy
            if normalize:
                xx_ = 2 * xx_ / (X - 1) - 1
                yy_ = 2 * yy_ / (Y - 1) - 1
            grid = torch.stack([
                yy_.reshape(N, X, Y),
                xx_.reshape(N, X, Y) 
            ], dim=-1)
        else:
            xx_egos = ego_points[:, 0]
            yy_egos = ego_points[:, 1]
            zz_egos = ego_points[:, 2]
            xx_ = xx_egos * fx + cx
            yy_ = yy_egos * fy + cy
            zz_ = zz_egos * fz + cz
            if normalize:
                xx_ = 2 * xx_ / (X - 1) - 1
                yy_ = 2 * yy_ / (Y - 1) - 1
                zz_ = 2 * zz_ / (Z - 1) - 1
            grid = torch.stack([
                yy_.reshape(N, Z, X, Y),
                xx_.reshape(N, Z, X, Y),
                zz_.reshape(N, Z, X, Y)
            ], dim=-1)
        return grid
        
    
    def forward(self, voxel_feats_pre, delta_poses):
        """
        Output a time-aligned voxel tensor from previous voxel features.

        Parameters
        ----------
        voxel_feats_pre : torch.Tensor
            shape should be (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        delta_poses : torch.Tensor
            shape should be (N, 4, 4)
        
        Return
        ------
        output : torch.Tensor
            shape should be (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        """
        # gen ego_points from voxel
        ego_points = self._unproject_points_from_voxel_to_ego().to(
            voxel_feats_pre, non_blocking=True)[None]
        # get projection matrix
        if self.bev_mode:
            assert len(voxel_feats_pre.shape) == 4, 'must be 4-D Tensor'
            rotations = delta_poses[:, :2, :2]
            translations = delta_poses[:, :2, [3]]
        else:
            assert len(voxel_feats_pre.shape) == 5, 'must be 5-D Tensor'
            rotations = delta_poses[:, :3, :3]
            translations = delta_poses[:, :3, [3]]
        # project to previous ego coords and get grid
        ego_points_projected = rotations @ ego_points + translations
        grid = self._project_points_from_ego_to_voxel(ego_points_projected)
        # apply grid sampling
        voxel_feats_pre_aligned = nn.functional.grid_sample(
            input=voxel_feats_pre, grid=grid, mode=self.interpolation, align_corners=False
        )
        return voxel_feats_pre_aligned


@MODELS.register_module()
class VoxelStreamFusion(BaseModule):
    def __init__(self, in_channels, mid_channels=128, bev_mode=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        if bev_mode:
            self.gain = nn.Sequential(
                ConvBN(in_channels, mid_channels),
                ConvBN(mid_channels, mid_channels),
                nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.gain = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(),
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(),
                nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        self.add = EltwiseAdd()
            
    
    def forward(self, voxel_feats_cur, voxel_feats_pre):
        """Temporal fusion of voxel current and previous features.

        Parameters
        ----------
        voxel_feats_cur : torch.Tensor
            current voxel features for measurement
        voxel_feats_pre : torch.Tensor
            previously predicted voxel features

        Returns
        -------
        voxel_feats_updated : torch.Tensor
            updated voxel features
        """
        assert voxel_feats_cur.shape == voxel_feats_pre.shape
        # get gain
        gain = self.gain(voxel_feats_cur)
        # update feats
        voxel_feats_updated = self.add(
            gain * voxel_feats_cur, (1 - gain) * voxel_feats_pre
        )
        return voxel_feats_updated
            

@MODELS.register_module()
class VoxelConcatFusion(BaseModule):
    def __init__(self, in_channels, pre_nframes, bev_mode=False, dilation=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        self.cat = Concat()
        if bev_mode:
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels * (pre_nframes + 1), in_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv3d(in_channels * (pre_nframes + 1), in_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(in_channels)
            )
    
    def forward(self, *aligned_voxel_feats_cat):
        """Temporal fusion of voxel current and previous features.

        Parameters
        ----------
        aligned_voxel_feats_cat : List[torch.Tensor]
            concated nframes of voxel features

        Returns
        -------
        voxel_feats_fused : torch.Tensor
            updated voxel features
        """
        cat = self.cat(*aligned_voxel_feats_cat)
        voxel_feats_fused = self.fuse(cat)
        return voxel_feats_fused


@MODELS.register_module()
class FeatureConcatFusion(BaseModule):
    def __init__(self, in_channels, out_channels, bev_mode=False, dilation=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        self.cat = Concat()
        if bev_mode:
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,
                          dilation=dilation),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, *aligned_voxel_feats_cat):
        """Temporal fusion of voxel current and previous features.

        Parameters
        ----------
        aligned_voxel_feats_cat : List[torch.Tensor]
            concated nframes of voxel features

        Returns
        -------
        voxel_feats_fused : torch.Tensor
            updated voxel features
        """
        cat = self.cat(*aligned_voxel_feats_cat)
        voxel_feats_fused = self.fuse(cat)
        return voxel_feats_fused


@MODELS.register_module()
class VoVNetEncoder(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 out_channels,
                 repeat=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.enc_tower = OSABlock(in_channels, 
                                  mid_channels, 
                                  out_channels, 
                                  stride=1, 
                                  repeat=repeat)
    
    def forward(self, x):
        return self.enc_tower(x)


class ResModule(BaseModule):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.convbn1 = ConvBN(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, has_relu=True)
        self.convbn2 = ConvBN(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, has_relu=False)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        x = self.convbn1(x)
        x = self.convbn2(x)
        x = identity + x
        x = self.activation(x)
        return x


@MODELS.register_module()
class M2BevEncoder(BaseModule):
    def __init__(self, 
                 in_channels, 
                 shrink_channels,
                 out_channels,
                 repeat=7,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
    
        # if fuse is not None:
        #     self.fuse = nn.Conv2d(fuse["in_channels"], fuse["out_channels"], kernel_size=1)
        # else:
        #     self.fuse = None
        self.shrink = nn.Conv2d(in_channels, shrink_channels, kernel_size=1)
        model = nn.ModuleList()
        model.append(ResModule(in_channels=shrink_channels, mid_channels=shrink_channels, out_channels=shrink_channels))
        model.append(ConvBN(
                in_channels=shrink_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                has_relu=True))
        for i in range(repeat - 1):
            model.append(ResModule(in_channels=out_channels, mid_channels=out_channels, out_channels=out_channels))
            model.append(ConvBN(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_relu=True))
        self.model = nn.Sequential(*model)
        self.init_params()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        x = self.shrink(x)
        out = self.model.forward(x)
        return out
    
    def init_params(self):
        nn.init.xavier_normal_(self.shrink.weight.data)





class ResModule2D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type='BN2d'), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x


@MODELS.register_module()
class M2BevNeck(nn.Module):
    """Neck for M2BEV.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=2,
                 norm_cfg=dict(type='BN2d'),
                 stride=2,
                 is_transpose=True,
                 fuse=None,
                 with_cp=False):
        super().__init__()

        self.is_transpose = is_transpose
        self.with_cp = with_cp
        for i in range(3):
            print('neck transpose: {}'.format(is_transpose))

        if fuse is not None:
            self.fuse = nn.Conv2d(fuse["in_channels"], fuse["out_channels"], kernel_size=1)
        else:
            self.fuse = None

        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        if bool(os.getenv("DEPLOY", False)):
            N, X, Y, Z, C = x.shape
            x = x.reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)
        else:
            # N, C*T, X, Y, Z -> N, X, Y, Z, C -> N, X, Y, Z*C*T -> N, Z*C*T, X, Y
            N, C, X, Y, Z = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)

        if self.fuse is not None:
            x = self.fuse(x)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return [x.transpose(-1, -2)]
        else:
            return [x]



@MODELS.register_module()
class VoxelEncoderFPN(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels_list, 
                 out_channels,
                 repeats=3,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        assert len(mid_channels_list) == 3
        if type(repeats) is int:
            repeats = (repeats, repeats, repeats)
        else:
            assert len(repeats) == 3
        self.osa0 = OSABlock(in_channels, mid_channels_list[0], mid_channels_list[0], stride=1, repeat=repeats[0], final_dilation=2, relu6=relu6)
        self.osa1 = OSABlock(mid_channels_list[0], mid_channels_list[1], mid_channels_list[1], stride=2, repeat=repeats[1], final_dilation=2, relu6=relu6)
        self.osa2 = OSABlock(mid_channels_list[1], mid_channels_list[2], mid_channels_list[2], stride=2, repeat=repeats[2], final_dilation=2, relu6=relu6)

        self.p1_linear = ConvBN(mid_channels_list[2], mid_channels_list[1], kernel_size=1, padding=0, relu6=relu6)
        self.p1_up = nn.ConvTranspose2d(mid_channels_list[1], mid_channels_list[1], kernel_size=2, stride=2, padding=0, bias=False)
        self.p1_fusion = Concat()
        self.p0_linear = ConvBN(mid_channels_list[1] * 2, mid_channels_list[0], kernel_size=1, padding=0, relu6=relu6)
        self.p0_up = nn.ConvTranspose2d(mid_channels_list[0], mid_channels_list[0], kernel_size=2, stride=2, padding=0, bias=False)
        self.p0_fusion = Concat()

        self.out = ConvBN(mid_channels_list[0] * 2, out_channels, relu6=relu6)
    
    def forward(self, x):
        osa0 = self.osa0(x)
        osa1 = self.osa1(osa0)
        osa2 = self.osa2(osa1)

        p1 = self.p1_fusion(osa1, self.p1_up(self.p1_linear(osa2)))
        p0 = self.p0_fusion(osa0, self.p0_up(self.p0_linear(p1)))

        return self.out(p0)


@MODELS.register_module()
class PlanarHead(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 reg_scales=None,
                 repeat=3,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.seg_tower = OSABlock(in_channels, 
                                  mid_channels, 
                                  stride=1, 
                                  repeat=repeat, 
                                  with_reduce=False)
        self.reg_tower = OSABlock(in_channels, 
                                  mid_channels, 
                                  stride=1, 
                                  repeat=repeat, 
                                  with_reduce=False)
        self.cen_seg = nn.Conv2d(mid_channels * repeat, 
                                 cen_seg_channels, 
                                 kernel_size=1)
        self.reg = nn.Conv2d(mid_channels * repeat, 
                             reg_channels, 
                             kernel_size=1)
        self.has_reg_scale = reg_scales is not None
        if self.has_reg_scale:
            assert len(reg_scales) == reg_channels
            self.reg_scales = torch.tensor(reg_scales, dtype=torch.float32)[None, :, None, None]
    
    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        if self.has_reg_scale:
            reg = reg * self.reg_scales.to(reg.device)
        return cen_seg, reg


@MODELS.register_module()
class PlanarHeadSimple(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 reg_scales=None,
                 repeat=1,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        self.seg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        )
        self.reg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        )
        self.cen_seg = nn.Conv2d(mid_channels, cen_seg_channels, kernel_size=1)
        self.reg = nn.Conv2d(mid_channels, reg_channels, kernel_size=1)
        self.has_reg_scale = reg_scales is not None
        if self.has_reg_scale:
            assert len(reg_scales) == reg_channels
            self.reg_scales = torch.tensor(reg_scales, dtype=torch.float32)[None, :, None, None]
    
    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        if self.has_reg_scale:
            reg = reg * self.reg_scales.to(reg.device)
        return cen_seg, reg
        