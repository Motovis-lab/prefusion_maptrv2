# Copyright (c) Megvii Inc. All rights reserved.
import imp
import torch
from torch import nn
from mmengine.runner import autocast
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from .base_lss_fpn import BaseLSSFPN
from ..layers import HoriConv, DepthReducer
from ..utils import get_unproj_func
import numpy as np
import ipdb
from ..utils import FisheyeCamera, PerspectiveCamera, fish_unproject_points_from_image_to_camera, pv_unproject_points_from_image_to_camera



# NOTE Modified Lift-Splat
@MODELS.register_module()
class MatrixVT_FISH(BaseLSSFPN):

    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound_fish,
        final_dim,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf,
    ):
        """Modified from LSSFPN.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound_fish (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """
        super().__init__(
            x_bound,
            y_bound,
            z_bound,
            d_bound_fish,
            final_dim,
            downsample_factor,
            output_channels,
            img_backbone_conf,
            img_neck_conf,
            depth_net_conf,
            use_da=False,
        )
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound_fish = d_bound_fish
        self.register_buffer('bev_anchors',
                             self.create_bev_anchors(x_bound, y_bound))
        self.horiconv = HoriConv(self.output_channels, 512,
                                 self.output_channels)
        self.depth_reducer = DepthReducer(self.output_channels,
                                          self.output_channels)
        self.depth_channels = torch.arange(*self.d_bound_fish).shape[0]
        self.static_mat = None

    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """
        x_coords = ((torch.linspace(
            x_bound[0] - x_bound[2] * ds_rate,
            x_bound[1],
            self.voxel_num[0] // ds_rate,
            dtype=torch.float,
        ) + x_bound[2] * ds_rate / 2).view(self.voxel_num[0] // ds_rate,
                                           1).expand(
                                               self.voxel_num[0] // ds_rate,
                                               self.voxel_num[1] // ds_rate))
        y_coords = ((torch.linspace(
            y_bound[0] - y_bound[2] * ds_rate,
            y_bound[1],
            self.voxel_num[1] // ds_rate,
            dtype=torch.float,
        ) + y_bound[2] * ds_rate / 2).view(
            1,
            self.voxel_num[1] // ds_rate).expand(self.voxel_num[0] // ds_rate,
                                                 self.voxel_num[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return anchors

    def create_frustum(self, intrin_mat):
        self.cx, self.cy, self.fx, self.fy, p0, p1, p2, p3 = intrin_mat[0][0].cpu().detach().numpy()
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound_fish,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW).unsqueeze(-1) # 1, 16, 40, 1
        paddings = torch.ones_like(d_coords)
        D,_,_,_ = d_coords.shape
        
        x_coords, y_coords, demo_camera_point = fish_unproject_points_from_image_to_camera((self.cx, self.cy, self.fx, self.fy, p0, p1, p2, p3),
                                                                        180, ogfH, ogfW, fH, fW)
        demo_camera_point = torch.from_numpy(demo_camera_point.reshape(-1,fH, fW)).to(d_coords.device).unsqueeze(-1)
        d_coords = d_coords * demo_camera_point
        x_coords = torch.from_numpy(x_coords).unsqueeze(0).expand(D, fH, fW).unsqueeze(-1).to(d_coords.device)
        y_coords = torch.from_numpy(y_coords).unsqueeze(0).expand(D, fH, fW).unsqueeze(-1).to(d_coords.device)

        return torch.concat([x_coords * d_coords, y_coords * d_coords, d_coords, paddings], dim=-1).to(torch.float32)

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.create_frustum(intrin_mat).to(sensor2ego_mat.device)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)) # 因为本身生成的像素点points表示的图像坐标，如果有图像层面的增强需要做变换

        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)) # 图像坐标系到ego坐标系转换矩阵
        # combine = sensor2ego_mat.matmul(intrin_mat)
        points = sensor2ego_mat.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)  # 图像坐标系到ego坐标系的转换
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)  # 如果有box的增强要跟图像points的深度和位置做匹配。bda_mat本身是对空间下所有的box的旋转平移的变换矩阵
        else:
            points = points.squeeze(-1)
        return points[..., :3]  # 感觉有点像视锥空间的点

    def get_proj_mat(self, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """
        if self.static_mat is not None:
            return self.static_mat

        bev_size_w = int(self.voxel_num[0])  
        bev_size_h = int(self.voxel_num[1])
        bev_size = int(self.voxel_num[0])  
        geom_sep = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, 0, ...],
            mats_dict['intrin_mats'][:, 0, ...],
            mats_dict['ida_mats'],
            mats_dict.get('bda_mat', None),
        )  # 经过一系列固定尺寸大小的bound和图像分辨率大小的操作，和变换矩阵，生成得到相对固定的坐标矩阵
        geom_sep = (
            geom_sep +
            (self.voxel_coord + self.voxel_size / 2.0)) / self.voxel_size
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2,
                                            4).contiguous()  # B,Ncam,W,D,3    高度上进行求均值，做了压缩
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[...,
                                                                           1]) # 小于0的无效值
        invalid2 = torch.logical_or((geom_sep > (bev_size - 1))[..., 0],
                                    (geom_sep > (bev_size - 1))[..., 1])  # 超出范围的无效值
        geom_sep[(invalid1 | invalid2)] = int(bev_size / 2)  # 将所有无效值全部设置为64
        geom_idx = geom_sep[..., 1] * bev_size + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])  # B,128,128,2  根据实际的xy物理范围创建对应的格子 anchors
        B, L, L, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, L * L))  # 跟深度有关

        ray_map = geom_uni.new_zeros((B, Nc * W, L * L))  # 跟图像特征压缩完的  W有关
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1   # geom_idx[b, dir] 选中的index为1 ，未选中的为0
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1  # geom_idx[b, :, d] 选中的index为1 ，未选中的为0
        null_point = int((bev_size / 2) * (bev_size + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, L * L)
        ray_map = ray_map.view(B, -1, L * L)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]  # 归一化深度维度的大小0-1
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]  # 同理归一化图像特征W相关维度大小0-1

        return circle_map, ray_map

    def reduce_and_project(self, feature, depth, mats_dict):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        # [N,25,H,W], [N,80,H,W]  depth.shape context.shape
        # Prime Depth feature w x d
        depth = self.depth_reducer(feature, depth)  # Compress the H channel

        B = mats_dict['intrin_mats'].shape[0]

        # N, C, H, W = feature.shape
        # feature=feature.reshape(N,C*H,W)
        # horiconv -> Prime Feature w x c
        feature = self.horiconv(feature)
        # feature = feature.max(2)[0]
        # [N.25,W], [N,80,W]
        depth = depth.permute(0, 2, 1).reshape(B, -1, self.depth_channels)  # B N*W C
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.output_channels)  # B N*W C
        circle_map, ray_map = self.get_proj_mat(mats_dict)

        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1)
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.voxel_num[:2])

        return img_feat_with_depth

    def _forward_single_sweep(self,
                              sweep_idx,
                              sweep_imgs,
                              metainfo,
                              is_return_depth=False):
        (
            batch_size,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        dummy_ida_mat = torch.eye(4).unsqueeze(0).unsqueeze(1).expand(metainfo[0].shape[0], metainfo[0].shape[1], -1, -1).to(metainfo[0].device)
        mats_dict = {
            'sensor2ego_mats': metainfo[0].unsqueeze(1),
            'intrin_mats': metainfo[1].unsqueeze(1),
            'ida_mats': dummy_ida_mat, 'sensor2sensor_mats': None, 'bda_mat': None
        }
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self.depth_net(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        with autocast(device_type=depth_feature.device, enabled=False):
            feature = depth_feature[:, self.depth_channels:(
                self.depth_channels + self.output_channels)].float()
            depth = depth_feature[:, :self.depth_channels].float().softmax(1)

            img_feat_with_depth = self.reduce_and_project(
                feature, depth, mats_dict)  # [b*n, c, d, w]

            if is_return_depth:
                return img_feat_with_depth.contiguous(), depth
            return img_feat_with_depth.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        num_sweeps = 1
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs,
            mats_dict,
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)

@MODELS.register_module()
class MatrixVT_PV(BaseLSSFPN):

    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound_front,
        d_bound_pv,
        final_dim_front,
        final_dim_other,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf_front,
        depth_net_conf_other
    ):
        """Modified from LSSFPN.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound_pv (list): Boundaries for d.
            d_bound_front (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """
        super().__init__(
            x_bound,
            y_bound,
            z_bound,
            d_bound_front,
            final_dim_front,
            downsample_factor,
            output_channels,
            img_backbone_conf,
            img_neck_conf,
            depth_net_conf_front,
            use_da=False,
        )
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.final_dim_front = final_dim_front
        self.final_dim_other = final_dim_other
        self.d_bound_front = d_bound_front
        self.d_bound_pv = d_bound_pv
        self.register_buffer('bev_anchors',
                             self.create_bev_anchors(x_bound, y_bound))
        self.horiconv = HoriConv(self.output_channels, 512,
                                 self.output_channels)
        self.depth_reducer = DepthReducer(self.output_channels,
                                          self.output_channels)
        self.depth_channels_front = torch.arange(*self.d_bound_front).shape[0]
        self.depth_channels_other = torch.arange(*self.d_bound_pv).shape[0]
        self.depth_net_front = MODELS.build(depth_net_conf_front)
        self.depth_net_other = MODELS.build(depth_net_conf_other)
        self.static_mat = None

    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """
        x_coords = ((torch.linspace(
            x_bound[0] - x_bound[2] * ds_rate,
            x_bound[1],
            self.voxel_num[0] // ds_rate,
            dtype=torch.float,
        ) + x_bound[2] * ds_rate / 2).view(self.voxel_num[0] // ds_rate,
                                           1).expand(
                                               self.voxel_num[0] // ds_rate,
                                               self.voxel_num[1] // ds_rate))
        y_coords = ((torch.linspace(
            y_bound[0] - y_bound[2] * ds_rate,
            y_bound[1],
            self.voxel_num[1] // ds_rate,
            dtype=torch.float,
        ) + y_bound[2] * ds_rate / 2).view(
            1,
            self.voxel_num[1] // ds_rate).expand(self.voxel_num[0] // ds_rate,
                                                 self.voxel_num[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return anchors

    def create_frustum_front(self, intrin_mat):
        self.cx, self.cy, self.fx, self.fy = intrin_mat[0][0].cpu().detach().numpy()
        ogfH, ogfW = self.final_dim_front
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound_front,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW).unsqueeze(-1) # 1, 16, 40, 1
        paddings = torch.ones_like(d_coords)
        D,_,_,_ = d_coords.shape
        x_coords, y_coords, demo_camera_point = pv_unproject_points_from_image_to_camera((self.cx, self.cy, self.fx, self.fy),
                                                                        ogfH, ogfW, fH, fW)
        demo_camera_point /= np.linalg.norm(demo_camera_point, axis=0)
        demo_camera_point = torch.from_numpy(demo_camera_point.reshape(-1,fH, fW)).to(d_coords.device).unsqueeze(-1)      
        d_coords = d_coords * demo_camera_point
        
        x_coords = torch.from_numpy(x_coords).unsqueeze(0).expand(D, fH, fW).unsqueeze(-1).to(d_coords.device)
        y_coords = torch.from_numpy(y_coords).unsqueeze(0).expand(D, fH, fW).unsqueeze(-1).to(d_coords.device)

        return torch.concat([x_coords * d_coords, y_coords * d_coords, d_coords, paddings], dim=-1).to(torch.float32)

    def create_frustum_other(self, intrin_mat):   
        self.cx, self.cy, self.fx, self.fy = intrin_mat[0][0].cpu().detach().numpy()
        ogfH, ogfW = self.final_dim_other
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound_pv,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW).unsqueeze(-1) # 1, 16, 40, 1
        paddings = torch.ones_like(d_coords)
        D,_,_,_ = d_coords.shape
        x_coords, y_coords, demo_camera_point = pv_unproject_points_from_image_to_camera((self.cx, self.cy, self.fx, self.fy),
                                                                        ogfH, ogfW, fH, fW)
        demo_camera_point /= np.linalg.norm(demo_camera_point, axis=0)
        demo_camera_point = torch.from_numpy(demo_camera_point.reshape(-1,fH, fW)).to(d_coords.device).unsqueeze(-1) 
        d_coords = d_coords * demo_camera_point
        
        x_coords = torch.from_numpy(x_coords).unsqueeze(0).expand(D, fH, fW).unsqueeze(-1).to(d_coords.device)
        y_coords = torch.from_numpy(y_coords).unsqueeze(0).expand(D, fH, fW).unsqueeze(-1).to(d_coords.device)

        return torch.concat([x_coords * d_coords, y_coords * d_coords, d_coords, paddings], dim=-1).to(torch.float32)


    def get_geometry_front(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.create_frustum_front(intrin_mat).to(sensor2ego_mat.device)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)) # 因为本身生成的像素点points表示的图像坐标，如果有图像层面的增强需要做变换

        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)) # 图像坐标系到ego坐标系转换矩阵
        # combine = sensor2ego_mat.matmul(intrin_mat)
        points = sensor2ego_mat.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)  # 图像坐标系到ego坐标系的转换
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)  # 如果有box的增强要跟图像points的深度和位置做匹配。bda_mat本身是对空间下所有的box的旋转平移的变换矩阵
        else:
            points = points.squeeze(-1)
        return points[..., :3]  # 感觉有点像视锥空间的点

    def get_geometry_other(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.create_frustum_other(intrin_mat).to(sensor2ego_mat.device)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)) # 因为本身生成的像素点points表示的图像坐标，如果有图像层面的增强需要做变换

        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)) # 图像坐标系到ego坐标系转换矩阵
        # combine = sensor2ego_mat.matmul(intrin_mat)
        points = sensor2ego_mat.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)  # 图像坐标系到ego坐标系的转换
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)  # 如果有box的增强要跟图像points的深度和位置做匹配。bda_mat本身是对空间下所有的box的旋转平移的变换矩阵
        else:
            points = points.squeeze(-1)
        return points[..., :3]  # 感觉有点像视锥空间的点

    def get_proj_mat_front(self, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """
        if self.static_mat is not None:
            return self.static_mat

        bev_size_h = int(self.voxel_num[0])  
        bev_size_w = int(self.voxel_num[1])
        geom_sep = self.get_geometry_front(
            mats_dict['sensor2ego_mats'][:, 0, ...],
            mats_dict['intrin_mats'][:, 0, ...],
            mats_dict['ida_mats'],
            mats_dict.get('bda_mat', None),
        )  # 经过一系列固定尺寸大小的bound和图像分辨率大小的操作，和变换矩阵，生成得到相对固定的坐标矩阵
        geom_sep = (
            geom_sep +
            (self.voxel_coord + self.voxel_size / 2.0)) / self.voxel_size
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2,
                                            4).contiguous()  # B,Ncam,W,D,3    高度上进行求均值，做了压缩
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[...,
                                                                           1]) # 小于0的无效值
        invalid2 = torch.logical_or((geom_sep > (bev_size_h - 1))[..., 0],
                                    (geom_sep > (bev_size_w - 1))[..., 1])  # 超出范围的无效值
        geom_sep[(invalid1 | invalid2)] = torch.from_numpy(np.array([int(bev_size_h / 2), int(bev_size_w / 2)])).to(geom_sep.device).to(geom_sep.dtype)
        
        geom_idx = geom_sep[..., 1] * bev_size_w + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])  # B,128,128,2  根据实际的xy物理范围创建对应的格子 anchors
        B, anchor_h, anchor_w, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, anchor_h * anchor_w))  # 跟深度有关

        ray_map = geom_uni.new_zeros((B, Nc * W, anchor_h * anchor_w))  # 跟图像特征压缩完的  W有关
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1   # geom_idx[b, dir] 选中的index为1 ，未选中的为0
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1  # geom_idx[b, :, d] 选中的index为1 ，未选中的为0
        null_point = int((bev_size_h / 2) * (bev_size_w + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, anchor_h * anchor_w)
        ray_map = ray_map.view(B, -1, anchor_h * anchor_w)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]  # 归一化深度维度的大小0-1
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]  # 同理归一化图像特征W相关维度大小0-1

        return circle_map, ray_map

    def get_proj_mat_other(self, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """
        if self.static_mat is not None:
            return self.static_mat

        bev_size_h = int(self.voxel_num[0])  
        bev_size_w = int(self.voxel_num[1])
        geom_sep = self.get_geometry_other(
            mats_dict['sensor2ego_mats'][:, 0, ...],
            mats_dict['intrin_mats'][:, 0, ...],
            mats_dict['ida_mats'],
            mats_dict.get('bda_mat', None),
        )  # 经过一系列固定尺寸大小的bound和图像分辨率大小的操作，和变换矩阵，生成得到相对固定的坐标矩阵
        geom_sep = (
            geom_sep +
            (self.voxel_coord + self.voxel_size / 2.0)) / self.voxel_size
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2,
                                            4).contiguous()  # B,Ncam,W,D,3    高度上进行求均值，做了压缩
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[...,
                                                                           1]) # 小于0的无效值
        invalid2 = torch.logical_or((geom_sep > (bev_size_h - 1))[..., 0],
                                    (geom_sep > (bev_size_w - 1))[..., 1])  # 超出范围的无效值
        geom_sep[(invalid1 | invalid2)] = torch.from_numpy(np.array([int(bev_size_h / 2), int(bev_size_w / 2)])).to(geom_sep.device).to(geom_sep.dtype)
        
        geom_idx = geom_sep[..., 1] * bev_size_h + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])  # B,128,128,2  根据实际的xy物理范围创建对应的格子 anchors
        B, anchor_h, anchor_w, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, anchor_h * anchor_w))  # 跟深度有关

        ray_map = geom_uni.new_zeros((B, Nc * W, anchor_h * anchor_w))  # 跟图像特征压缩完的  W有关
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1   # geom_idx[b, dir] 选中的index为1 ，未选中的为0
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1  # geom_idx[b, :, d] 选中的index为1 ，未选中的为0
        null_point = int((bev_size_h / 2) * (bev_size_w + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, anchor_h * anchor_w)
        ray_map = ray_map.view(B, -1, anchor_h * anchor_w)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]  # 归一化深度维度的大小0-1
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]  # 同理归一化图像特征W相关维度大小0-1

        return circle_map, ray_map

    def reduce_and_project_front(self, feature, depth, mats_dict):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        # [N,25,H,W], [N,80,H,W]  depth.shape context.shape
        # Prime Depth feature w x d
        depth = self.depth_reducer(feature, depth)  # Compress the H channel

        B = mats_dict['intrin_mats'].shape[0]

        # N, C, H, W = feature.shape
        # feature=feature.reshape(N,C*H,W)
        # horiconv -> Prime Feature w x c
        feature = self.horiconv(feature)
        # feature = feature.max(2)[0]
        # [N.25,W], [N,80,W]
        depth = depth.permute(0, 2, 1).reshape(B, -1, self.depth_channels_front)  # B N*W C
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.output_channels)  # B N*W C
        circle_map, ray_map = self.get_proj_mat_front(mats_dict)

        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1)
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.voxel_num[:2])

        return img_feat_with_depth

    def reduce_and_project_other(self, feature, depth, mats_dict):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        # [N,25,H,W], [N,80,H,W]  depth.shape context.shape
        # Prime Depth feature w x d
        depth = self.depth_reducer(feature, depth)  # Compress the H channel

        B = mats_dict['intrin_mats'].shape[0]

        # N, C, H, W = feature.shape
        # feature=feature.reshape(N,C*H,W)
        # horiconv -> Prime Feature w x c
        feature = self.horiconv(feature)
        # feature = feature.max(2)[0]
        # [N.25,W], [N,80,W]
        depth = depth.permute(0, 2, 1).reshape(B, -1, self.depth_channels_other)  # B N*W C
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.output_channels)  # B N*W C
        circle_map, ray_map = self.get_proj_mat_other(mats_dict)

        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1)
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.voxel_num[:2])

        return img_feat_with_depth


    def _forward_single_sweep_front(self,
                              sweep_idx,
                              sweep_imgs,
                              metainfo,
                              is_return_depth=False):
        (
            batch_size,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        dummy_ida_mat = torch.eye(4).unsqueeze(0).unsqueeze(1).expand(metainfo[0].shape[0], metainfo[0].shape[1], -1, -1).to(metainfo[0].device)
        mats_dict = {
            'sensor2ego_mats': metainfo[0].unsqueeze(1),
            'intrin_mats': metainfo[1].unsqueeze(1),
            'ida_mats': dummy_ida_mat, 'sensor2sensor_mats': None, 'bda_mat': None
        }
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self.depth_net_front(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        with autocast(device_type=depth_feature.device, enabled=False):
            feature = depth_feature[:, self.depth_channels_front:(
                self.depth_channels_front + self.output_channels)].float()
            depth = depth_feature[:, :self.depth_channels_front].float().softmax(1)

            img_feat_with_depth = self.reduce_and_project_front(
                feature, depth, mats_dict)  # [b*n, c, d, w]

            if is_return_depth:
                return img_feat_with_depth.contiguous(), depth
            return img_feat_with_depth.contiguous()
    
    def _forward_single_sweep_other(self,
                              sweep_idx,
                              sweep_imgs,
                              metainfo,
                              is_return_depth=False):
        (
            batch_size,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        dummy_ida_mat = torch.eye(4).unsqueeze(0).unsqueeze(1).expand(metainfo[0].shape[0], metainfo[0].shape[1], -1, -1).to(metainfo[0].device)
        mats_dict = {
            'sensor2ego_mats': metainfo[0].unsqueeze(1),
            'intrin_mats': metainfo[1].unsqueeze(1),
            'ida_mats': dummy_ida_mat, 'sensor2sensor_mats': None, 'bda_mat': None
        }
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self.depth_net_other(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        with autocast(device_type=depth_feature.device, enabled=False):
            feature = depth_feature[:, self.depth_channels_other:(
                self.depth_channels_other + self.output_channels)].float()
            depth = depth_feature[:, :self.depth_channels_other].float().softmax(1)

            img_feat_with_depth = self.reduce_and_project_other(
                feature, depth, mats_dict)  # [b*n, c, d, w]

            if is_return_depth:
                return img_feat_with_depth.contiguous(), depth
            return img_feat_with_depth.contiguous()

    def forward(self,
                sweep_imgs_front,
                mats_dict_fornt,
                sweep_imgs_other,
                mats_dict_other,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_cams, num_channels, img_height, \
            img_width = sweep_imgs_front.shape
        num_sweeps = 1
        key_frame_res_front = self._forward_single_sweep_front(
            0,
            sweep_imgs_front,
            mats_dict_fornt,
            is_return_depth=is_return_depth)
        key_frame_res_other = self._forward_single_sweep_other(
            0,
            sweep_imgs_other,
            mats_dict_other,
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res_front, key_frame_res_other

        key_frame_feature_front = key_frame_res_front[
            0] if is_return_depth else key_frame_res_front

        key_frame_feature_other = key_frame_res_other[
            0] if is_return_depth else key_frame_res_other

        ret_feature_list_front = [key_frame_feature_front]
        ret_feature_list_other = [key_frame_feature_other]
        # for sweep_index in range(1, num_sweeps):
        #     with torch.no_grad():
        #         feature_map = self._forward_single_sweep(
        #             sweep_index,
        #             sweep_imgs[:, sweep_index:sweep_index + 1, ...],
        #             mats_dict,
        #             is_return_depth=False)
        #         ret_feature_list.append(feature_map)

        if is_return_depth:
            return (torch.cat(ret_feature_list_front, 1), key_frame_res_front[1]), (torch.cat(ret_feature_list_other, 1), key_frame_res_other[1])
        else:
            return torch.cat(ret_feature_list_front, 1), torch.cat(ret_feature_list_front, 1)

if __name__ == '__main__':
    backbone_conf = {
        'x_bound': [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        'y_bound': [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        'z_bound': [-5, 3, 8],  # BEV grids bounds and size (m)
        'd_bound': [2.0, 58.0,
                    0.5],  # Categorical Depth bounds and division (m)
        'final_dim': (256, 704),  # img size for model input (pix)
        'output_channels':
        80,  # BEV feature channels
        'downsample_factor':
        16,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)  # noqa
        'img_backbone_conf':
        dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained',
                          checkpoint='torchvision://resnet50'),
        ),
        'img_neck_conf':
        dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
        'depth_net_conf':
        dict(type='Fish_DepthNet', in_channels=512, mid_channels=512, context_channels=80, depth_channels=25)
    }

    model = MatrixVT_FISH(**backbone_conf)
    # for inference and deployment where intrin & extrin mats are static
    # model.static_mat = model.get_proj_mat(mats_dict)
    ipdb.set_trace()
    bev_feature, depth = model(
        torch.rand((2, 1, 6, 3, 256, 704)), {
            'sensor2ego_mats': torch.rand((2, 1, 6, 4, 4)),
            'intrin_mats': torch.rand((2, 1, 6, 4, 4)),
            'ida_mats': torch.rand((2, 1, 6, 4, 4)),
            'sensor2sensor_mats': torch.rand((2, 1, 6, 4, 4)),
            'bda_mat': torch.rand((2, 4, 4)),
        },
        is_return_depth=True)

    print(bev_feature.shape, depth.shape)
