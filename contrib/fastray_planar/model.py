
import torch
import torch.nn as nn

from functools import reduce

from torch import Tensor
from typing import Union, List, Dict, Optional

from mmengine.model import BaseModel, BaseModule
from mmengine.structures import BaseDataElement

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
    "PlanarHead",
    'FastRayPlanarSingleFrameModel',
    "FastRayPlanarStreamModel"
]


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 has_relu=True):
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
                 has_bn=True):
        super(OSABlock, self).__init__()
        assert stride in [1, 2]
        assert repeat >= 2
        self.repeat = repeat
        self.with_reduce = with_reduce

        self.conv1 = ConvBN(in_channels, mid_channels, stride=stride, padding=dilation, dilation=dilation)

        for i in range(repeat - 2):
            self._modules['conv{}'.format(i + 2)] = ConvBN(
                mid_channels, mid_channels, padding=dilation, dilation=dilation
            )

        self._modules['conv{}'.format(repeat)] = ConvBN(
            mid_channels, mid_channels, padding=final_dilation, dilation=final_dilation
        )

        self.concat = Concat()
        if with_reduce:
            assert out_channels is not None
            if has_bn:
                self.reduce = ConvBN(mid_channels * repeat, out_channels, kernel_size=1, padding=0)
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
    def __init__(self, out_stride=8, out_channels=128, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.strides = [4, 8, 16, 32]
        assert out_stride in self.strides
        self.out_stride = out_stride

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2)

        # NECK
        if self.out_stride <= 16:
            self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
            self.p4_fusion = Concat()
        if self.out_stride <= 8:
            self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0)
            self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
            self.p3_fusion = Concat()
        if self.out_stride <= 4:
            self.p2_linear = ConvBN(256, 96, kernel_size=1, padding=0)
            self.p2_up = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0, bias=False)
            self.p2_fusion = Concat()
        
        in_channels = {4: 192, 8: 256, 16: 384, 32: 192}
        mid_channels = {4: 96, 8: 96, 16: 128, 32: 192}
        self.out = OSABlock(
            in_channels[self.out_stride], mid_channels[self.out_stride], out_channels,
            stride=1, repeat=3, has_bn=False
        )
        
        
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1)
        osa3 = self.osa3(osa2)
        osa4 = self.osa4(osa3)
        osa5 = self.osa5(osa4)

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
class FastRaySpatialTransform(BaseModule):
    
    def __init__(self, 
                 voxel_shape, 
                 fusion_mode='weighted', 
                 bev_mode=False, 
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        assert fusion_mode in ['weighted', 'sampled', 'bilinear_weighted']
        self.fusion_mode = fusion_mode
        self.bev_mode = bev_mode
    
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
            return voxel_feats.reshape(N, C*Z, X, Y)
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
    def __init__(self, in_channels, pre_nframes, bev_mode=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        self.cat = Concat()
        if bev_mode:
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels * (pre_nframes + 1), in_channels, kernel_size=3, padding=1),
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


@MODELS.register_module()
class PlanarHead(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
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
    
    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        return cen_seg, reg
        


@MODELS.register_module()
class FastRayPlanarSingleFrameModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 heads,
                 loss_cfg=None,
                 debug_mode=False,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_front = MODELS.build(backbones['pv_front'])
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        self.backbone_fisheyes = MODELS.build(backbones['fisheyes'])
        # view transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # init losses
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_polyline_3d = MODELS.build(loss_cfg['polyline_3d'])
        self.loss_parkingslot_3d = MODELS.build(loss_cfg['parkingslot_3d'])

    
    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        # backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_front']:
                camera_feats_dict[cam_id] = self.backbone_pv_front(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['fisheyes']:
                camera_feats_dict[cam_id] = self.backbone_fisheyes(camera_tensors_dict[cam_id])
        # spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats = self.spatial_transform(camera_feats_dict, camera_lookups)
        # voxel encoder
        if len(voxel_feats.shape) == 5:
            N, C, Z, X, Y = voxel_feats.shape
            voxel_feats = voxel_feats.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats)
        # heads
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        # outputs
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:],
            reg=out_bbox_3d[1])
        pred_polyline_3d = dict(
            seg=out_polyline_3d[0],
            reg=out_polyline_3d[1])
        pred_parkingslot_3d = dict(
            cen=out_parkingslot_3d[0][:, 0:1],
            seg=out_parkingslot_3d[0][:, 1:],
            reg=out_parkingslot_3d[1])

        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d,
                           pred_polyline_3d,
                           pred_parkingslot_3d)
        
        if mode == 'tensor':
            return dict(
                hidden_feats=self.voxel_feats_pre,
                pred_bbox_3d=pred_bbox_3d,
                pred_polyline_3d=pred_polyline_3d,
                pred_parkingslot_3d=pred_parkingslot_3d
            )
        if mode == 'loss':
            gt_bbox_3d = batched_input_dict['annotations']['bbox_3d']
            gt_polyline_3d = batched_input_dict['annotations']['polyline_3d']
            gt_parkingslot_3d = batched_input_dict['annotations']['parkingslot_3d']

            try:
                loss_bbox_3d = self.loss_bbox_3d(pred_bbox_3d, gt_bbox_3d)
            except Exception as e:
                print(e)
                print(gt_bbox_3d)
                print(batched_input_dict['index_infos'][0])
            loss_polyline_3d = self.loss_polyline_3d(pred_polyline_3d, gt_polyline_3d)
            loss_parkingslot_3d = self.loss_parkingslot_3d(pred_parkingslot_3d, gt_parkingslot_3d)

            total_loss = sum([loss_bbox_3d['bbox_3d_loss'],
                              loss_polyline_3d['polyline_3d_loss'],
                              loss_parkingslot_3d['parkingslot_3d_loss']])

            losses = dict(
                loss=total_loss,
                seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
                seg_iou_loss_polyline_3d=loss_polyline_3d['polyline_3d_seg_iou_0_loss'],
                seg_iou_loss_parkingslot_3d=loss_parkingslot_3d['parkingslot_3d_seg_iou_0_loss']
            )

            return losses
        
        if mode == 'predict':
            raise NotImplementedError
    

def draw_out_feats(
        batched_input_dict, 
        camera_tensors_dict,
        pred_bbox_3d,
        pred_polyline_3d=None,
        pred_parkingslot_3d=None,
    ):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots(3, 10)
    fig.suptitle(batched_input_dict['index_infos'][0].scene_frame_id)
    for i, cam_id in enumerate(camera_tensors_dict):
        img = camera_tensors_dict[cam_id].detach().cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1] * 255 + 128
        img = img.astype(np.uint8)
        plt.subplot(3, 10, i+1)
        plt.title(cam_id.replace('VCAMERA_', '').lower())
        plt.imshow(img)

    gt_seg = batched_input_dict['annotations']['bbox_3d']['seg'][0][0].detach().cpu()
    pred_seg = pred_bbox_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
    plt.subplot(3, 10, 11)
    plt.imshow(gt_seg)
    plt.title('bbox_3d gt_seg')
    plt.subplot(3, 10, 12)
    plt.imshow(pred_seg)
    plt.title("bbox_3d pred_seg")
    
    gt_cen = batched_input_dict['annotations']['bbox_3d']['cen'][0][0].detach().cpu()
    pred_cen = pred_bbox_3d['cen'][0][0].to(torch.float32).sigmoid().detach().cpu()
    pred_cen *= (pred_seg > 0.5)
    plt.subplot(3, 10, 13)
    plt.imshow(gt_cen)
    plt.title("bbox_3d gt_cen")
    plt.subplot(3, 10, 14)
    plt.imshow(pred_cen)
    plt.title("bbox_3d pred_cen")
    
    gt_reg = batched_input_dict['annotations']['bbox_3d']['reg'][0][0].detach().cpu()
    pred_reg = pred_bbox_3d['reg'][0][0].to(torch.float32).detach().cpu()
    pred_reg *= (pred_seg > 0.5)
    plt.subplot(3, 10, 15)
    plt.imshow(gt_reg)
    plt.title("bbox_3d gt_reg")
    plt.subplot(3, 10, 16)
    plt.imshow(pred_reg)
    plt.title("bbox_3d pred_reg")
    
    if pred_polyline_3d is not None:
        gt_seg = batched_input_dict['annotations']['polyline_3d']['seg'][0][0].detach().cpu()
        pred_seg = pred_polyline_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
        plt.subplot(3, 10, 17)
        plt.imshow(gt_seg)
        plt.title('polyline_3d gt_seg')
        plt.subplot(3, 10, 18)
        plt.imshow(pred_seg)
        plt.title("polyline_3d pred_seg")
        
        gt_reg = batched_input_dict['annotations']['polyline_3d']['reg'][0][0].detach().cpu()
        pred_reg = pred_polyline_3d['reg'][0][0].to(torch.float32).detach().cpu()
        pred_reg *= (pred_seg > 0.5)
        plt.subplot(3, 10, 19)
        plt.imshow(gt_reg)
        plt.title("polyline_3d gt_reg")
        plt.subplot(3, 10, 20)
        plt.imshow(pred_reg)
        plt.title("polyline_3d pred_reg")
    
    if pred_parkingslot_3d is not None:
        gt_seg = batched_input_dict['annotations']['parkingslot_3d']['seg'][0][1].detach().cpu()
        pred_mask = pred_parkingslot_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
        pred_seg = pred_parkingslot_3d['seg'][0][1].to(torch.float32).sigmoid().detach().cpu()
        plt.subplot(3, 10, 21)
        plt.imshow(gt_seg)
        plt.title('parkingslot_3d gt_seg')
        plt.subplot(3, 10, 22)
        plt.imshow(pred_seg)
        plt.title("parkingslot_3d pred_seg")
        
        gt_cen = batched_input_dict['annotations']['parkingslot_3d']['cen'][0][0].detach().cpu()
        pred_cen = pred_parkingslot_3d['cen'][0][0].to(torch.float32).sigmoid().detach().cpu()
        pred_cen *= (pred_mask > 0.5)
        plt.subplot(3, 10, 23)
        plt.imshow(gt_cen)
        plt.title("parkingslot_3d gt_cen")
        plt.subplot(3, 10, 24)
        plt.imshow(pred_cen)
        plt.title("parkingslot_3d pred_cen")
        
        gt_reg = batched_input_dict['annotations']['parkingslot_3d']['reg'][0][2].detach().cpu()
        pred_reg = pred_parkingslot_3d['reg'][0][2].to(torch.float32).detach().cpu()
        pred_reg *= (pred_mask > 0.5)
        plt.subplot(3, 10, 25)
        plt.imshow(gt_reg)
        plt.title("parkingslot_3d gt_reg")
        plt.subplot(3, 10, 26)
        plt.imshow(pred_reg)
        plt.title("parkingslot_3d pred_reg")
    

    voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
    plt.subplot(3, 10, 27)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    
    gt_boxes_3d = batched_input_dict['transformables'][0]['bbox_3d']
    
    for element in gt_boxes_3d.elements:
        center = element['translation'][:, 0]
        xvec = element['size'][0] * element['rotation'][:, 0]
        yvec = element['size'][1] * element['rotation'][:, 1]
        corner_points = np.array([
            center + 0.5 * xvec - 0.5 * yvec,
            center + 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec - 0.5 * yvec
        ], dtype=np.float32)
        # print('gt: ', corner_points[:, :2])
        plt.plot(corner_points[[0, 1, 2, 3, 0], 1], corner_points[[0, 1, 2, 3, 0], 0], 'g')
    
    # gt_boxes_3d = batched_input_dict['annotations']['bbox_3d']
    plt.subplot(3, 10, 28)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    pred_bbox_3d_0 = {
        'cen': pred_bbox_3d['cen'][0].cpu().float().sigmoid(),
        'seg': pred_bbox_3d['seg'][0].cpu().float().sigmoid(),
        'reg': pred_bbox_3d['reg'][0].cpu().float()
    }
    pred_boxes_3d = get_bbox_3d(pred_bbox_3d_0)
    
    for element in pred_boxes_3d:
        # if element['confs'][0] < 0.7:
        #     continue
        if element['area_score'] < 0.5:
            continue
        center = element['translation']
        xvec = element['size'][0] * element['rotation'][:, 0]
        yvec = element['size'][1] * element['rotation'][:, 1]
        corner_points = np.array([
            center + 0.5 * xvec - 0.5 * yvec,
            center + 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec - 0.5 * yvec
        ], dtype=np.float32)
        # print('pred: ', corner_points[:, :2])
        plt.text(center[1], center[0], '{:.2f}'.format(element['area_score']), color='r',
                 ha='center', va='center')
        plt.plot(corner_points[[0, 1, 2, 3, 0], 1], corner_points[[0, 1, 2, 3, 0], 0], 'r')
    
    plt.subplot(3, 10, 29)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])

    gt_slots_3d = batched_input_dict['transformables'][0]['parkingslot_3d']
    
    for element in gt_slots_3d.elements:
        points = element['points']
        plt.plot(points[[1, 2, 3, 0], 1], points[[1, 2, 3, 0], 0], 'g')
    
    
    plt.subplot(3, 10, 30)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    
    pred_parkingslot_3d_0 = {
        'cen': pred_parkingslot_3d['cen'][0].cpu().float().sigmoid(),
        'seg': pred_parkingslot_3d['seg'][0].cpu().float().sigmoid(),
        'reg': pred_parkingslot_3d['reg'][0].cpu().float()
    }
    
    pred_slots_3d = get_parkingslot_3d(pred_parkingslot_3d_0)
    # print(pred_slots_3d)
    for slot in pred_slots_3d:
        # plt.text(points[0, 1], points[0, 0], '{:.2f}'.format(element['confs'][0]), color='r')
        plt.plot(slot[[1, 2, 3, 0], 1], slot[[1, 2, 3, 0], 0], 'r')
    
    plt.show()


def draw_aligned_voxel_feats(aligned_voxel_feats):
    import numpy as np
    import matplotlib.pyplot as plt

    n_frames = len(aligned_voxel_feats)
    plt.subplots(1, n_frames)
    for i, voxel_feats_frame in enumerate(aligned_voxel_feats):
        plt.subplot(1, n_frames, i+1)
        plt.imshow(voxel_feats_frame[0, 0].detach().cpu().to(torch.float32).numpy() > 0)
        plt.title(f'frame t({0 - i})')
    plt.show()



def get_bbox_3d(tensor_dict):
    from prefusion.dataset.tensor_smith import PlanarBbox3D
    pbox = PlanarBbox3D(
        voxel_shape=(6, 320, 160),
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        reverse_pre_conf=0.3,
        reverse_nms_ratio=1.0
    )
    return pbox.reverse(tensor_dict)
    

def get_parkingslot_3d(tensor_dict):
    from prefusion.dataset.tensor_smith import PlanarParkingSlot3D
    pslot = PlanarParkingSlot3D(
        voxel_shape=(6, 320, 160),
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        reverse_pre_conf=0.5
    )
    return pslot.reverse(tensor_dict)



@MODELS.register_module()
class FastRayPlanarMultiFrameModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 pre_nframes=1,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_front = MODELS.build(backbones['pv_front'])
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        self.backbone_fisheyes = MODELS.build(backbones['fisheyes'])
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        # self.voxel_fusion = EltwiseAdd()
        self.voxel_fusion = MODELS.build(voxel_fusion)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # hidden voxel features for temporal fusion
        self.pre_nframes = pre_nframes
        self.cached_voxel_feats = {}
        self.cached_delta_poses = {}
        # init losses
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_polyline_3d = MODELS.build(loss_cfg['polyline_3d'])
        self.loss_parkingslot_3d = MODELS.build(loss_cfg['parkingslot_3d'])

    
    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        delta_poses = batched_input_dict['delta_poses']
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_front']:
                camera_feats_dict[cam_id] = self.backbone_pv_front(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['fisheyes']:
                camera_feats_dict[cam_id] = self.backbone_fisheyes(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats_cur = self.spatial_transform(camera_feats_dict, camera_lookups)
        ## temporal transform
        cur_first_index_info = batched_input_dict['index_infos'][0]
        # init tmp pre_0 cache
        self.cached_voxel_feats[f'pre_0'] = voxel_feats_cur
        self.cached_delta_poses[f'pre_0'] = delta_poses
        for pre_i in range(self.pre_nframes):
            index_info_prev_str = 'cur_first_index_info' + ''.join(['.prev'] * (pre_i+1))
            if eval(index_info_prev_str) is None:
                for pre_j in range(pre_i, self.pre_nframes):
                    self.cached_voxel_feats[f'pre_{pre_j+1}'] = self.cached_voxel_feats[f'pre_{pre_j}'].clone().detach()
                    self.cached_delta_poses[f'pre_{pre_j+1}'] = self.cached_delta_poses[f'pre_{pre_j}'].clone().detach()
                break
        # cache delta_poses
        for pre_i in range(self.pre_nframes, 1, -1):
            self.cached_delta_poses[f'pre_{pre_i}'] = (self.cached_delta_poses[f'pre_{pre_i-1}'] @ delta_poses).clone().detach()
        self.cached_delta_poses['pre_1'] = delta_poses.clone().detach()
        # align all history frames to current frame
        aligned_voxel_feats_cat = [voxel_feats_cur]
        for pre_i in range(self.pre_nframes):
            aligned_voxel_feats_cat.append(self.temporal_transform(
                self.cached_voxel_feats[f'pre_{pre_i+1}'], self.cached_delta_poses[f'pre_{pre_i+1}']
            ))
        if self.debug_mode:
            draw_aligned_voxel_feats(aligned_voxel_feats_cat)
            # draw_aligned_voxel_feats(list(self.cached_voxel_feats.values()))
        # cache voxel features
        for pre_i in range(self.pre_nframes, 0, -1):
            self.cached_voxel_feats[f'pre_{pre_i}'] = (self.cached_voxel_feats[f'pre_{pre_i-1}']).clone().detach()
        # pop tmp pre_0 cache
        self.cached_voxel_feats.pop('pre_0')
        self.cached_delta_poses.pop('pre_0')
        ## voxel fusion
        voxel_feats_fused = self.voxel_fusion(*aligned_voxel_feats_cat)
        ## voxel encoder
        if len(voxel_feats_fused.shape) == 5:
            N, C, Z, X, Y = voxel_feats_fused.shape
            voxel_feats_fused = voxel_feats_fused.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats_fused)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:],
            reg=out_bbox_3d[1])
        pred_polyline_3d = dict(
            seg=out_polyline_3d[0],
            reg=out_polyline_3d[1])
        pred_parkingslot_3d = dict(
            cen=out_parkingslot_3d[0][:, 0:1],
            seg=out_parkingslot_3d[0][:, 1:],
            reg=out_parkingslot_3d[1])
        
        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d,
                           pred_polyline_3d,
                           pred_parkingslot_3d)
        
        if mode == 'tensor':
            return dict(
                pred_bbox_3d=pred_bbox_3d,
                pred_polyline_3d=pred_polyline_3d,
                pred_parkingslot_3d=pred_parkingslot_3d
            )
        if mode == 'loss':
            gt = batched_input_dict['annotations']
            pred = {"bbox_3d": pred_bbox_3d, "polyline_3d": pred_polyline_3d, "parkingslot_3d": pred_parkingslot_3d}
            losses = self.compute_losses(gt, pred)
            return losses

        if mode == 'predict':
            gt = batched_input_dict['annotations']
            pred = {"bbox_3d": pred_bbox_3d, "polyline_3d": pred_polyline_3d, "parkingslot_3d": pred_parkingslot_3d}
            losses = self.compute_losses(gt, pred)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt: Dict, pred: Dict):
        loss_bbox_3d = self.loss_bbox_3d(pred["bbox_3d"], gt['bbox_3d'])
        loss_polyline_3d = self.loss_polyline_3d(pred["polyline_3d"], gt['polyline_3d'])
        loss_parkingslot_3d = self.loss_parkingslot_3d(pred["parkingslot_3d"], gt['parkingslot_3d'])

        total_loss = sum([loss_bbox_3d['bbox_3d_loss'],
                            loss_polyline_3d['polyline_3d_loss'],
                            loss_parkingslot_3d['parkingslot_3d_loss']])

        losses = dict(
            loss=total_loss,
            seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
            seg_iou_loss_polyline_3d=loss_polyline_3d['polyline_3d_seg_iou_0_loss'],
            seg_iou_loss_parkingslot_3d=loss_parkingslot_3d['parkingslot_3d_seg_iou_0_loss']
        )

        return losses




@MODELS.register_module()
class FastRayPlanarStreamModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_front = MODELS.build(backbones['pv_front'])
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        self.backbone_fisheyes = MODELS.build(backbones['fisheyes'])
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        self.voxel_fusion = MODELS.build(voxel_fusion)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # hidden voxel features for temporal fusion
        self.voxel_feats_pre = None
        # init losses
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_polyline_3d = MODELS.build(loss_cfg['polyline_3d'])
        self.loss_parkingslot_3d = MODELS.build(loss_cfg['parkingslot_3d'])

    
    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        delta_poses = batched_input_dict['delta_poses']
        # backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_front']:
                camera_feats_dict[cam_id] = self.backbone_pv_front(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['fisheyes']:
                camera_feats_dict[cam_id] = self.backbone_fisheyes(camera_tensors_dict[cam_id])
        # spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats_cur = self.spatial_transform(camera_feats_dict, camera_lookups)
        # temporal transform
        if batched_input_dict['index_infos'][0].prev is None:
            self.voxel_feats_pre = voxel_feats_cur.clone().detach()
        voxel_feats_pre_aligned = self.temporal_transform(self.voxel_feats_pre, delta_poses)
        # voxel fusion
        voxel_feats_updated = self.voxel_fusion(voxel_feats_cur, voxel_feats_pre_aligned)
        self.voxel_feats_pre = voxel_feats_updated.clone().detach()
        # voxel encoder
        if len(voxel_feats_updated.shape) == 5:
            N, C, Z, X, Y = voxel_feats_updated.shape
            voxel_feats_updated = voxel_feats_updated.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats_updated)
        # heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:],
            reg=out_bbox_3d[1])
        pred_polyline_3d = dict(
            seg=out_polyline_3d[0],
            reg=out_polyline_3d[1])
        pred_parkingslot_3d = dict(
            cen=out_parkingslot_3d[0][:, 0:1],
            seg=out_parkingslot_3d[0][:, 1:],
            reg=out_parkingslot_3d[1])
        
        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d,
                           pred_polyline_3d,
                           pred_parkingslot_3d)

        if mode == 'tensor':
            return dict(
                hidden_feats=self.voxel_feats_pre,
                pred_bbox_3d=pred_bbox_3d,
                pred_polyline_3d=pred_polyline_3d,
                pred_parkingslot_3d=pred_parkingslot_3d
            )
        if mode == 'loss':
            gt = batched_input_dict['annotations']
            pred = {"bbox_3d": pred_bbox_3d, "polyline_3d": pred_polyline_3d, "parkingslot_3d": pred_parkingslot_3d}
            losses = self.compute_losses(gt, pred)
            return losses

        if mode == 'predict':
            gt = batched_input_dict['annotations']
            pred = {"bbox_3d": pred_bbox_3d, "polyline_3d": pred_polyline_3d, "parkingslot_3d": pred_parkingslot_3d}
            losses = self.compute_losses(gt, pred)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt: Dict, pred: Dict):
        loss_bbox_3d = self.loss_bbox_3d(pred["bbox_3d"], gt['bbox_3d'])
        loss_polyline_3d = self.loss_polyline_3d(pred["polyline_3d"], gt['polyline_3d'])
        loss_parkingslot_3d = self.loss_parkingslot_3d(pred["parkingslot_3d"], gt['parkingslot_3d'])

        total_loss = sum([loss_bbox_3d['bbox_3d_loss'],
                            loss_polyline_3d['polyline_3d_loss'],
                            loss_parkingslot_3d['parkingslot_3d_loss']])

        losses = dict(
            loss=total_loss,
            seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
            seg_iou_loss_polyline_3d=loss_polyline_3d['polyline_3d_seg_iou_0_loss'],
            seg_iou_loss_parkingslot_3d=loss_parkingslot_3d['parkingslot_3d_seg_iou_0_loss']
        )

        return losses


@MODELS.register_module()
class NuscenesFastRayPlanarSingleFrameModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 heads,
                 loss_cfg=None,
                 debug_mode=False,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        # view transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_bbox_3d_cylinder = MODELS.build(heads['bbox_3d_cylinder'])
        self.head_bbox_3d_oriented_cylinder = MODELS.build(heads['bbox_3d_oriented_cylinder'])
        self.head_bbox_3d_rect_cuboid = MODELS.build(heads['bbox_3d_rect_cuboid'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # init losses
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_bbox_3d_cylinder = MODELS.build(loss_cfg['bbox_3d_cylinder'])
        self.loss_bbox_3d_oriented_cylinder = MODELS.build(loss_cfg['bbox_3d_oriented_cylinder'])
        self.loss_bbox_3d_rect_cuboid = MODELS.build(loss_cfg['bbox_3d_rect_cuboid'])

    
    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        # backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
        # spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats = self.spatial_transform(camera_feats_dict, camera_lookups)

        # if self.debug_mode:
        #     draw_aligned_voxel_feats([voxel_feats])

        # voxel encoder
        if len(voxel_feats.shape) == 5:
            N, C, Z, X, Y = voxel_feats.shape
            voxel_feats = voxel_feats.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats)
        # heads
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_bbox_3d_cylinder = self.head_bbox_3d_cylinder(bev_feats)
        out_bbox_3d_oriented_cylinder = self.head_bbox_3d_oriented_cylinder(bev_feats)
        out_bbox_3d_rect_cuboid = self.head_bbox_3d_rect_cuboid(bev_feats)
        # outputs
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:],
            reg=out_bbox_3d[1])
        pred_bbox_3d_cylinder = dict(
            cen=out_bbox_3d_cylinder[0][:, 0:1],
            seg=out_bbox_3d_cylinder[0][:, 1:],
            reg=out_bbox_3d_cylinder[1])
        pred_bbox_3d_oriented_cylinder = dict(
            cen=out_bbox_3d_oriented_cylinder[0][:, 0:1],
            seg=out_bbox_3d_oriented_cylinder[0][:, 1:],
            reg=out_bbox_3d_oriented_cylinder[1])
        pred_bbox_3d_rect_cuboid = dict(
            cen=out_bbox_3d_rect_cuboid[0][:, 0:1],
            seg=out_bbox_3d_rect_cuboid[0][:, 1:],
            reg=out_bbox_3d_rect_cuboid[1])

        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d,)
        
        if mode == 'tensor':
            return dict(
                hidden_feats=self.voxel_feats_pre,
                pred_bbox_3d=pred_bbox_3d,
                pred_bbox_3d_cylinder=pred_bbox_3d_cylinder,
                pred_bbox_3d_oriented_cylinder=pred_bbox_3d_oriented_cylinder,
                pred_bbox_3d_rect_cuboid=pred_bbox_3d_rect_cuboid,
            )
        if mode == 'loss':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return losses

        if mode == 'predict':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt: Dict, pred: Dict):
        loss_bbox_3d = self.loss_bbox_3d(pred["bbox_3d"], gt['bbox_3d'])
        loss_bbox_3d_cylinder = self.loss_bbox_3d_cylinder(pred["bbox_3d_cylinder"], gt["bbox_3d_cylinder"])
        loss_bbox_3d_oriented_cylinder = self.loss_bbox_3d_oriented_cylinder(pred["bbox_3d_oriented_cylinder"], gt["bbox_3d_oriented_cylinder"])
        loss_bbox_3d_rect_cuboid = self.loss_bbox_3d_rect_cuboid(pred["bbox_3d_rect_cuboid"], gt["bbox_3d_rect_cuboid"])

        total_loss = sum([
            loss_bbox_3d['bbox_3d_loss'],
            loss_bbox_3d_cylinder['bbox_3d_cylinder_loss'],
            loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_loss'],
            loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_loss'],
        ])

        losses = dict(
            loss=total_loss,
            seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_cylinder=loss_bbox_3d_cylinder['bbox_3d_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_oriented_cylinder=loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_rect_cuboid=loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_seg_iou_0_loss'],
        )

        return losses


@MODELS.register_module()
class NuscenesFastRayPlanarMultiFrameModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 pre_nframes=1,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        # self.voxel_fusion = EltwiseAdd()
        self.voxel_fusion = MODELS.build(voxel_fusion)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # hidden voxel features for temporal fusion
        self.pre_nframes = pre_nframes
        self.cached_voxel_feats = {}
        self.cached_delta_poses = {}
        # init losses
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_bbox_3d_cylinder = MODELS.build(loss_cfg['bbox_3d_cylinder'])
        self.loss_bbox_3d_oriented_cylinder = MODELS.build(loss_cfg['bbox_3d_oriented_cylinder'])
        self.loss_bbox_3d_rect_cuboid = MODELS.build(loss_cfg['bbox_3d_rect_cuboid'])

    
    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        delta_poses = batched_input_dict['delta_poses']
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats_cur = self.spatial_transform(camera_feats_dict, camera_lookups)
        ## temporal transform
        cur_first_index_info = batched_input_dict['index_infos'][0]
        # init tmp pre_0 cache
        self.cached_voxel_feats[f'pre_0'] = voxel_feats_cur
        self.cached_delta_poses[f'pre_0'] = delta_poses
        for pre_i in range(self.pre_nframes):
            index_info_prev_str = 'cur_first_index_info' + ''.join(['.prev'] * (pre_i+1))
            if eval(index_info_prev_str) is None:
                for pre_j in range(pre_i, self.pre_nframes):
                    self.cached_voxel_feats[f'pre_{pre_j+1}'] = self.cached_voxel_feats[f'pre_{pre_j}'].clone().detach()
                    self.cached_delta_poses[f'pre_{pre_j+1}'] = self.cached_delta_poses[f'pre_{pre_j}'].clone().detach()
                break
        # cache delta_poses
        for pre_i in range(self.pre_nframes, 1, -1):
            self.cached_delta_poses[f'pre_{pre_i}'] = (self.cached_delta_poses[f'pre_{pre_i-1}'] @ delta_poses).clone().detach()
        self.cached_delta_poses['pre_1'] = delta_poses.clone().detach()
        # align all history frames to current frame
        aligned_voxel_feats_cat = [voxel_feats_cur]
        for pre_i in range(self.pre_nframes):
            aligned_voxel_feats_cat.append(self.temporal_transform(
                self.cached_voxel_feats[f'pre_{pre_i+1}'], self.cached_delta_poses[f'pre_{pre_i+1}']
            ))
        if self.debug_mode:
            draw_aligned_voxel_feats(aligned_voxel_feats_cat)
            # draw_aligned_voxel_feats(list(self.cached_voxel_feats.values()))
        # cache voxel features
        for pre_i in range(self.pre_nframes, 0, -1):
            self.cached_voxel_feats[f'pre_{pre_i}'] = (self.cached_voxel_feats[f'pre_{pre_i-1}']).clone().detach()
        # pop tmp pre_0 cache
        self.cached_voxel_feats.pop('pre_0')
        self.cached_delta_poses.pop('pre_0')
        ## voxel fusion
        voxel_feats_fused = self.voxel_fusion(*aligned_voxel_feats_cat)
        ## voxel encoder
        if len(voxel_feats_fused.shape) == 5:
            N, C, Z, X, Y = voxel_feats_fused.shape
            voxel_feats_fused = voxel_feats_fused.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats_fused)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:9],
            reg=out_bbox_3d[1][:, 0:20])
        pred_bbox_3d_cylinder = dict(
            cen=out_bbox_3d[0][:, 9:10],
            seg=out_bbox_3d[0][:, 10:12],
            reg=out_bbox_3d[1][:, 20:28])
        pred_bbox_3d_oriented_cylinder = dict(
            cen=out_bbox_3d[0][:, 12:13],
            seg=out_bbox_3d[0][:, 13:15],
            reg=out_bbox_3d[1][:, 28:41])
        pred_bbox_3d_rect_cuboid = dict(
            cen=out_bbox_3d[0][:, 15:16],
            seg=out_bbox_3d[0][:, 16:18],
            reg=out_bbox_3d[1][:, 41:55])

        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d)
        
        if mode == 'tensor':
            return dict(
                pred_bbox_3d=pred_bbox_3d,
                pred_bbox_3d_cylinder=pred_bbox_3d_cylinder,
                pred_bbox_3d_oriented_cylinder=pred_bbox_3d_oriented_cylinder,
                pred_bbox_3d_rect_cuboid=pred_bbox_3d_rect_cuboid,
            )
        if mode == 'loss':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return losses

        if mode == 'predict':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt: Dict, pred: Dict):
        loss_bbox_3d = self.loss_bbox_3d(pred["bbox_3d"], gt['bbox_3d'])
        loss_bbox_3d_cylinder = self.loss_bbox_3d_cylinder(pred["bbox_3d_cylinder"], gt["bbox_3d_cylinder"])
        loss_bbox_3d_oriented_cylinder = self.loss_bbox_3d_oriented_cylinder(pred["bbox_3d_oriented_cylinder"], gt["bbox_3d_oriented_cylinder"])
        loss_bbox_3d_rect_cuboid = self.loss_bbox_3d_rect_cuboid(pred["bbox_3d_rect_cuboid"], gt["bbox_3d_rect_cuboid"])

        total_loss = sum([
            loss_bbox_3d['bbox_3d_loss'],
            loss_bbox_3d_cylinder['bbox_3d_cylinder_loss'],
            loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_loss'],
            loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_loss'],
        ])

        losses = dict(
            loss=total_loss,
            seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_cylinder=loss_bbox_3d_cylinder['bbox_3d_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_oriented_cylinder=loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_rect_cuboid=loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_seg_iou_0_loss'],
        )

        return losses