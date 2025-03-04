from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn

from mmengine.structures import BaseDataElement

from prefusion import BaseModel
from prefusion import SegIouLoss, DualFocalLoss
from prefusion.registry import MODELS
from . import ParkingFastRayPlanarSingleFrameModelAPALidar
from .misc_draw import draw_results_planar_lidar, draw_results_ps_lidar

from .modules import *
from .model_utils import *
import torch.nn.functional as F

@MODELS.register_module()
class ParkingFastRayPlanarSingleFrameModelAPALidarBigModel(BaseModel):

    def __init__(self,
                 backbone,
                 spatial_transform,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 picked_category=None,
                 category2head_mapping=None,
                 **kwargs
                 ):
        super().__init__(data_preprocessor, init_cfg)
        self.with_lidar = True
        if self.with_lidar:
            self.pts_middle_encoder = MODELS.build(kwargs['pts_middle_encoder'])
            self.pts_backbone = MODELS.build(kwargs['pts_backbone'])
            self.pts_neck = MODELS.build(kwargs['pts_neck'])
            self.lidar_voxel_fusion = MODELS.build(kwargs['lidar_voxel_fusion'])
        self.planar_losses_dict = {}
        self.heads_list = set()
        self.heads = deepcopy(heads)

        if picked_category is not None and category2head_mapping is not None:
            self.picked_category = picked_category
            self.category2head_mapping = category2head_mapping
            # then heads is also needed
            for head, val in heads.items():
                val['cen_seg_channels'] = sum([v for k, v in val['cen_seg_channels'].items() if k in picked_category])
                val['reg_channels'] = sum([v for k, v in val['reg_channels'].items() if k in picked_category])
                val['cen_seg_channels'] = 56  # 16
                val['reg_channels'] = 86  # 20
            # calculate the head channels
            st_cen_dict, st_reg_dict= defaultdict(int), defaultdict(int)
            self.categorys = {}
            for cate in picked_category:
                head = category2head_mapping[cate]
                self.categorys[cate] = {
                    'head': head,
                    'cen': [st_cen_dict[head],
                            st_cen_dict[head] + self.heads[head]['cen_seg_channels'][cate]],
                    'reg': [st_reg_dict[head],
                            st_reg_dict[head] + self.heads[head]['reg_channels'][cate]],
                }
                st_cen_dict[head] += self.heads[head]['cen_seg_channels'][cate]
                st_reg_dict[head] += self.heads[head]['reg_channels'][cate]

            for i in picked_category:
                branch = category2head_mapping[i]
                if branch == 'bbox_3d':
                    self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
                    self.heads_list.add('bbox_3d')
                elif branch == 'head_polyline_3d':
                    self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
                    self.heads_list.add('head_polyline_3d')
                elif branch == 'parkingslot_3d':
                    self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
                    self.heads_list.add('parkingslot_3d')
                elif branch == 'occ_sdf_bev':
                    self.head_occ_sdf_bev = MODELS.build(heads['occ_sdf_bev'])
                    self.occ_seg_iou_loss = SegIouLoss(method='linear')
                    self.occ_seg_dfl_loss = DualFocalLoss()
                    self.occ_sdf_l1_loss = nn.L1Loss(reduction='none')
                    self.occ_height_l1_loss = nn.L1Loss(reduction='none')
                    self.heads_list.add('occ_sdf_bev')
                else:
                    NotImplementedError
                self.planar_losses_dict[i] = MODELS.build(loss_cfg[i])
        else:
            NotImplementedError
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder, voxel related
        self.voxel_encoder = MODELS.build(voxel_encoder)


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
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            camera_feats_dict[cam_id] = self.backbone(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        bev_feats = self.spatial_transform(camera_feats_dict, camera_lookups)
        if len(bev_feats.shape) == 5:
            N, C, Z, X, Y = bev_feats.shape
            bev_feats = bev_feats.reshape(N, C * Z, X, Y)
        ## voxel encoder
        bev_feats = self.voxel_encoder(bev_feats)

        if self.with_lidar:
            lidar_data = batched_input_dict['lidar_points']
            batch_size = len(lidar_data['res_voxels'])
            coors = []
            voxel_features = []
            for val, voxel in enumerate(range(batch_size)):
                voxel_features += [lidar_data['res_voxels'][val].clone()]
                coors += [F.pad(lidar_data['res_coors'][val].clone(), (1, 0), mode='constant', value=val)]
            voxels_features = torch.cat(voxel_features, dim=0)
            coors_batch = torch.cat(coors, dim=0)
            x = self.pts_middle_encoder(voxels_features, coors_batch, batch_size)
            x = self.pts_backbone(x)
            if True:
                lidar_features = self.pts_neck(x)
                lidar_features = torch.cat(lidar_features, dim=0)
            bev_feats = self.lidar_voxel_fusion(bev_feats, lidar_features)

        ## heads & outputs: from self.picked_category
        out_3d_dict = {}
        if 'bbox_3d' in self.heads_list:
            out_3d_dict['bbox_3d'] = out_bbox_3d = self.head_bbox_3d(bev_feats)
        if 'polyline_3d' in self.heads_list:
            out_3d_dict['polyline_3d'] =out_polyline_3d = self.head_polyline_3d(bev_feats)
        if 'parkingslot_3d' in self.heads_list:
            out_3d_dict['parkingslot_3d'] =out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        if 'occ_sdf_bev' in self.heads_list:
            out_3d_dict['occ_sdf_bev'] =out_occ_sdf_bev = self.head_occ_sdf_bev(bev_feats)
        pred_dict = {}
        for cate, v in self.categorys.items():
            head = self.category2head_mapping[cate]
            st_cen, ed_cen = self.categorys[cate]['cen'][0], self.categorys[cate]['cen'][1]
            st_reg, ed_reg = self.categorys[cate]['reg'][0], self.categorys[cate]['reg'][1]
            pred_dict[cate] = dict(
                cen = out_3d_dict[head][0][:, st_cen:st_cen+1],
                seg = out_3d_dict[head][0][:, st_cen+1:ed_cen],
                reg = out_3d_dict[head][1][:, st_reg: ed_reg]
            )

        # pred_dict = dict(
        #     bbox_3d_heading=dict(
        #         cen=out_bbox_3d[0][:, 0:1],
        #         seg=out_bbox_3d[0][:, 1:16],
        #         reg=out_bbox_3d[1][:, 0:20]),
        #     # bbox_3d_plane_heading=dict(
        #     #     cen=out_bbox_3d[0][:, 14:15],
        #     #     seg=out_bbox_3d[0][:, 15:26],
        #     #     reg=out_bbox_3d[1][:, 20:40]),
        #     # bbox_3d_no_heading=dict(
        #     #     cen=out_bbox_3d[0][:, 26:27],
        #     #     seg=out_bbox_3d[0][:, 27:34],
        #     #     reg=out_bbox_3d[1][:, 40:54]),
        #     # bbox_3d_square=dict(
        #     #     cen=out_bbox_3d[0][:, 34:35],
        #     #     seg=out_bbox_3d[0][:, 35:40],
        #     #     reg=out_bbox_3d[1][:, 54:65]),
        #     # bbox_3d_cylinder=dict(
        #     #     cen=out_bbox_3d[0][:, 40:41],
        #     #     seg=out_bbox_3d[0][:, 41:51],
        #     #     reg=out_bbox_3d[1][:, 65:73]),
        #     # bbox_3d_oriented_cylinder=dict(
        #     #     cen=out_bbox_3d[0][:, 51:52],
        #     #     seg=out_bbox_3d[0][:, 52:54],
        #     #     reg=out_bbox_3d[1][:, 73:86]),
        #     # polyline_3d=dict(
        #     #     seg=out_polyline_3d[0][:, 0:9],
        #     #     reg=out_polyline_3d[1][:, 0:7]),
        #     # polygon_3d=dict(
        #     #     seg=out_polyline_3d[0][:, 9:15],
        #     #     reg=out_polyline_3d[1][:, 7:14]),
        #     # parkingslot_3d=dict(
        #     #     cen=out_parkingslot_3d[0][:, :1],
        #     #     seg=out_parkingslot_3d[0][:, 1:],
        #     #     reg=out_parkingslot_3d[1]),
        # )
        # # pred_occ_sdf_bev = dict(
        # #     seg=out_occ_sdf_bev[0],
        # #     sdf=out_occ_sdf_bev[1][:, 0:1],
        # #     height=out_occ_sdf_bev[1][0:, 1:2],
        # # )

        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']
            # gt_occ_sdf_bev = gt_dict['occ_sdf_bev']

        if self.debug_mode:
            import matplotlib.pyplot as plt
            pass

        if mode == 'infer':
            assert self.debug_mode
            # images = draw_results_planar_lidar(pred_dict, batched_input_dict, save_im=False)
            # images = draw_results_ps_lidar(pred_dict, batched_input_dict, save_im=False)
            images = draw_results_ps_lidar(pred_dict, batched_input_dict, save_im=False)
            return images
        if mode == 'tensor':
            # pred_dict['occ_sdf_bev'] = pred_occ_sdf_bev
            return pred_dict
        if mode == 'loss':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            # losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            # losses['loss'] += losses['occ_sdf_bev_loss']
            return losses
        if mode == 'predict':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            # losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            # losses['loss'] += losses['occ_sdf_bev_loss']
            return (
                *[{branch: {t: v.cpu() for t, v in _pred.items()}} for branch, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_planar_losses(self, pred_dict, gt_dict):
        losses = dict(loss=0)
        for branch in self.planar_losses_dict:
            losses_branch = self.planar_losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            losses.update(losses_branch)

        return losses

    def compute_occ_sdf_losses(self, pred_occ_sdf_bev, gt_occ_sdf_bev):
        # seg_im = np.stack([freespace, occ_edge])
        # sdf_im = np.stack([sdf])
        # height_im = np.stack([height, heigh_mask])
        losses = {}
        losses['occ_seg_iou_0_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 0:1],
                                                             gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_dfl_0_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 0:1],
                                                             gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_iou_1_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 1:2],
                                                             gt_occ_sdf_bev['seg'][:, 1:2])
        losses['occ_seg_dfl_1_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 1:2],
                                                             gt_occ_sdf_bev['seg'][:, 1:2])

        sdf_mask = gt_occ_sdf_bev['seg'][:, 0:1] + 0.1
        sdf_loss = self.occ_sdf_l1_loss(pred_occ_sdf_bev['sdf'] * sdf_mask, gt_occ_sdf_bev['sdf'] * sdf_mask)
        losses['occ_sdf_loss'] = sdf_loss.sum() / sdf_mask.sum()

        gt_height = gt_occ_sdf_bev['height'][:, 0:1]
        gt_height_mask = gt_occ_sdf_bev['height'][:, 1:2]
        pred_height = pred_occ_sdf_bev['height']
        occ_height_loss = self.occ_height_l1_loss(pred_height * gt_height_mask, gt_height * gt_height_mask)
        losses['occ_height_loss'] = occ_height_loss.sum() / gt_height_mask.sum()

        losses['occ_sdf_bev_loss'] = 2 * (
                5 * losses['occ_seg_iou_0_loss'] + 10 * losses['occ_seg_iou_1_loss'] + \
                10 * losses['occ_seg_dfl_0_loss'] + 20 * losses['occ_seg_dfl_1_loss'] + \
                20 * losses['occ_sdf_loss'] + 20 * losses['occ_height_loss'])

        return losses

@MODELS.register_module()
class ParkingFastRayPlanarMultiFrameModelAPALidarBigModel(BaseModel):

    def __init__(self,
                 backbone,
                 spatial_transform,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 picked_category=None,
                 category2head_mapping=None,
                 temporal_transform=None,
                 voxel_fusion=None,
                 **kwargs
                 ):
        super().__init__(data_preprocessor, init_cfg)
        self.with_lidar = True
        if self.with_lidar:
            self.pts_middle_encoder = MODELS.build(kwargs['pts_middle_encoder'])
            self.pts_backbone = MODELS.build(kwargs['pts_backbone'])
            self.pts_neck = MODELS.build(kwargs['pts_neck'])
            self.lidar_voxel_fusion = MODELS.build(kwargs['lidar_voxel_fusion'])
        self.planar_losses_dict = {}
        self.heads_list = set()
        self.heads = deepcopy(heads)

        if picked_category is not None and category2head_mapping is not None:
            self.picked_category = picked_category
            self.category2head_mapping = category2head_mapping
            # then heads is also needed
            for head, val in heads.items():
                val['cen_seg_channels'] = sum([v for k, v in val['cen_seg_channels'].items() if k in picked_category])
                val['reg_channels'] = sum([v for k, v in val['reg_channels'].items() if k in picked_category])
                val['cen_seg_channels'] = 56  # 16
                val['reg_channels'] = 86  # 20
            # calculate the head channels
            st_cen_dict, st_reg_dict= defaultdict(int), defaultdict(int)
            self.categorys = {}
            for cate in picked_category:
                head = category2head_mapping[cate]
                self.categorys[cate] = {
                    'head': head,
                    'cen': [st_cen_dict[head],
                            st_cen_dict[head] + self.heads[head]['cen_seg_channels'][cate]],
                    'reg': [st_reg_dict[head],
                            st_reg_dict[head] + self.heads[head]['reg_channels'][cate]],
                }
                st_cen_dict[head] += self.heads[head]['cen_seg_channels'][cate]
                st_reg_dict[head] += self.heads[head]['reg_channels'][cate]

            for i in picked_category:
                branch = category2head_mapping[i]
                if branch == 'bbox_3d':
                    self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
                    self.heads_list.add('bbox_3d')
                elif branch == 'head_polyline_3d':
                    self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
                    self.heads_list.add('head_polyline_3d')
                elif branch == 'parkingslot_3d':
                    self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
                    self.heads_list.add('parkingslot_3d')
                elif branch == 'occ_sdf_bev':
                    self.head_occ_sdf_bev = MODELS.build(heads['occ_sdf_bev'])
                    self.occ_seg_iou_loss = SegIouLoss(method='linear')
                    self.occ_seg_dfl_loss = DualFocalLoss()
                    self.occ_sdf_l1_loss = nn.L1Loss(reduction='none')
                    self.occ_height_l1_loss = nn.L1Loss(reduction='none')
                    self.heads_list.add('occ_sdf_bev')
                else:
                    NotImplementedError
                self.planar_losses_dict[i] = MODELS.build(loss_cfg[i])
        else:
            NotImplementedError
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder, voxel related
        self.voxel_encoder = MODELS.build(voxel_encoder)  # visual encoder
        self.multi_frame_fusion = kwargs.get('multi_frame_fusion', False)  # every mode has a temporal fusion module, and only when it is used
        if self.multi_frame_fusion:
            ## temporal
            self.temporal_transform = MODELS.build(temporal_transform)
            self.voxel_fusion = MODELS.build(voxel_fusion)
            self.cached_voxel_feats = None
            self.cached_delta_poses = None

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

        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            camera_feats_dict[cam_id] = self.backbone(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        bev_feats = self.spatial_transform(camera_feats_dict, camera_lookups)
        if len(bev_feats.shape) == 5:
            N, C, Z, X, Y = bev_feats.shape
            bev_feats = bev_feats.reshape(N, C * Z, X, Y)

        ## voxel encoder
        bev_feats = self.voxel_encoder(bev_feats)


        if self.with_lidar:
            lidar_data = batched_input_dict['lidar_points']
            batch_size = len(lidar_data['res_voxels'])
            coors = []
            voxel_features = []
            for val, voxel in enumerate(range(batch_size)):
                voxel_features += [lidar_data['res_voxels'][val].clone()]
                coors += [F.pad(lidar_data['res_coors'][val].clone(), (1, 0), mode='constant', value=val)]
            voxels_features = torch.cat(voxel_features, dim=0)
            coors_batch = torch.cat(coors, dim=0)
            x = self.pts_middle_encoder(voxels_features, coors_batch, batch_size)
            x = self.pts_backbone(x)
            if True:
                lidar_features = self.pts_neck(x)
                lidar_features = torch.cat(lidar_features, dim=0)
            bev_feats = self.lidar_voxel_fusion(bev_feats, lidar_features)


        # after modal fusion feature cache
        if self.multi_frame_fusion:  # stream feature fusion
            # delta pose: Tpre-curr, like use F.grid sample to
            if batched_input_dict['index_infos'][0].prev is not None:  # fuse previous feat
                delta_poses = batched_input_dict['delta_poses'].clone().detach()
                prev_bev_feats = self.temporal_transform(self.cached_voxel_feats, delta_poses)  # pose: Last frame;
                bev_feats = self.voxel_fusion(bev_feats, prev_bev_feats)
            elif batched_input_dict['index_infos'][0].prev is None:
                bev_feats = self.voxel_fusion(bev_feats, bev_feats.clone().detach())  # stream line temporal

            if batched_input_dict['index_infos'][0].next is not None:  # cached feat is next frame exists
                self.cached_voxel_feats = bev_feats.clone.detach()
                self.cached_delta_poses = batched_input_dict['delta_poses'].clone().detach()

        ## heads & outputs: from self.picked_category
        out_3d_dict = {}
        for head_name in self.heads_list:
            out_3d_dict[head_name] = getattr(self, 'head_' + head_name)(bev_feats)

        pred_dict = {}
        for cate, v in self.categorys.items():
            head = self.category2head_mapping[cate]
            st_cen, ed_cen = self.categorys[cate]['cen'][0], self.categorys[cate]['cen'][1]
            st_reg, ed_reg = self.categorys[cate]['reg'][0], self.categorys[cate]['reg'][1]
            pred_dict[cate] = dict(
                cen = out_3d_dict[head][0][:, st_cen:st_cen+1],
                seg = out_3d_dict[head][0][:, st_cen+1:ed_cen],
                reg = out_3d_dict[head][1][:, st_reg: ed_reg]
            )

        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']

        if self.debug_mode:
            import matplotlib.pyplot as plt
            pass

        if mode == 'infer':
            assert self.debug_mode
            # images = draw_results_planar_lidar(pred_dict, batched_input_dict, save_im=False)
            # images = draw_results_ps_lidar(pred_dict, batched_input_dict, save_im=False)
            images = draw_results_ps_lidar(pred_dict, batched_input_dict, save_im=False)
            return images
        if mode == 'tensor':
            # pred_dict['occ_sdf_bev'] = pred_occ_sdf_bev
            return pred_dict
        if mode == 'loss':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            # losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            # losses['loss'] += losses['occ_sdf_bev_loss']
            return losses
        if mode == 'predict':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            # losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            # losses['loss'] += losses['occ_sdf_bev_loss']
            return (
                *[{branch: {t: v.cpu() for t, v in _pred.items()}} for branch, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_planar_losses(self, pred_dict, gt_dict):
        losses = dict(loss=0)
        for branch in self.planar_losses_dict:
            losses_branch = self.planar_losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            losses.update(losses_branch)

        return losses

    def compute_occ_sdf_losses(self, pred_occ_sdf_bev, gt_occ_sdf_bev):
        # seg_im = np.stack([freespace, occ_edge])
        # sdf_im = np.stack([sdf])
        # height_im = np.stack([height, heigh_mask])
        losses = {}
        losses['occ_seg_iou_0_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 0:1],
                                                             gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_dfl_0_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 0:1],
                                                             gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_iou_1_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 1:2],
                                                             gt_occ_sdf_bev['seg'][:, 1:2])
        losses['occ_seg_dfl_1_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 1:2],
                                                             gt_occ_sdf_bev['seg'][:, 1:2])

        sdf_mask = gt_occ_sdf_bev['seg'][:, 0:1] + 0.1
        sdf_loss = self.occ_sdf_l1_loss(pred_occ_sdf_bev['sdf'] * sdf_mask, gt_occ_sdf_bev['sdf'] * sdf_mask)
        losses['occ_sdf_loss'] = sdf_loss.sum() / sdf_mask.sum()

        gt_height = gt_occ_sdf_bev['height'][:, 0:1]
        gt_height_mask = gt_occ_sdf_bev['height'][:, 1:2]
        pred_height = pred_occ_sdf_bev['height']
        occ_height_loss = self.occ_height_l1_loss(pred_height * gt_height_mask, gt_height * gt_height_mask)
        losses['occ_height_loss'] = occ_height_loss.sum() / gt_height_mask.sum()

        losses['occ_sdf_bev_loss'] = 2 * (
                5 * losses['occ_seg_iou_0_loss'] + 10 * losses['occ_seg_iou_1_loss'] + \
                10 * losses['occ_seg_dfl_0_loss'] + 20 * losses['occ_seg_dfl_1_loss'] + \
                20 * losses['occ_sdf_loss'] + 20 * losses['occ_height_loss'])

        return losses

