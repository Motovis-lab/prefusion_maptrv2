
import torch
import torch.nn as nn

from functools import reduce

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
class VoxelEncoderFPN(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels_list, 
                 out_channels,
                 repeats=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(mid_channels_list) == 3
        if type(repeats) is int:
            repeats = (repeats, repeats, repeats)
        else:
            assert len(repeats) == 3
        self.osa0 = OSABlock(in_channels, mid_channels_list[0], mid_channels_list[0], stride=1, repeat=repeats[0], final_dilation=2)
        self.osa1 = OSABlock(mid_channels_list[0], mid_channels_list[1], mid_channels_list[1], stride=2, repeat=repeats[1], final_dilation=2)
        self.osa2 = OSABlock(mid_channels_list[1], mid_channels_list[2], mid_channels_list[2], stride=2, repeat=repeats[2], final_dilation=2)

        self.p1_linear = ConvBN(mid_channels_list[2], mid_channels_list[1], kernel_size=1, padding=0)
        self.p1_up = nn.ConvTranspose2d(mid_channels_list[1], mid_channels_list[1], kernel_size=2, stride=2, padding=0, bias=False)
        self.p1_fusion = Concat()
        self.p0_linear = ConvBN(mid_channels_list[1] * 2, mid_channels_list[0], kernel_size=1, padding=0)
        self.p0_up = nn.ConvTranspose2d(mid_channels_list[0], mid_channels_list[0], kernel_size=2, stride=2, padding=0, bias=False)
        self.p0_fusion = Concat()

        self.out = ConvBN(mid_channels_list[0] * 2, out_channels)
    
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
class PlanarHeadSimple(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 repeat=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.seg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels),
            * ([ConvBN(mid_channels, mid_channels)] * repeat),
        )
        self.reg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels),
            * ([ConvBN(mid_channels, mid_channels)] * repeat),
        )
        self.cen_seg = nn.Conv2d(mid_channels, cen_seg_channels, kernel_size=1)
        self.reg = nn.Conv2d(mid_channels, reg_channels, kernel_size=1)
    
    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        return cen_seg, reg
        
