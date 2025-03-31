import torch

from torch import Tensor
from typing import Union, List, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import BaseDataElement
from copious.cv.geometry import Box3d as CopiousBox3d

from prefusion import BaseModel
from prefusion.registry import MODELS


@MODELS.register_module()
class FastBEVModel(BaseModel):
    def __init__(self,
                 camera_groups: List,
                 backbones,
                 neck,
                 neck_fuse,
                 neck_3d,
                 spatial_transform,
                 head_bbox_3d,
                 train_cfg,
                 test_cfg,
                 loss_cfg=None,
                 debug_mode=False,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone = MODELS.build(backbones)
        self.neck = MODELS.build(neck)
        self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        # view transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.neck_3d = MODELS.build(neck_3d)
        # voxel heads
        head_bbox_3d.update(train_cfg=train_cfg)
        head_bbox_3d.update(test_cfg=test_cfg)
        self.head_bbox_3d = MODELS.build(head_bbox_3d)
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # init losses
        self.losses_dict = {}
        if loss_cfg is not None:
            for branch in loss_cfg:
                self.losses_dict[branch] = MODELS.build(loss_cfg[branch])

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
            if cam_id in self.camera_groups:
                mlvl_feats = self.backbone(camera_tensors_dict[cam_id])
                mlvl_feats = self.neck(mlvl_feats) # normally it is FPN
                camera_feats_dict[cam_id] = self.fuse_multi_level_feats(mlvl_feats)

        # spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats = self.spatial_transform(camera_feats_dict, camera_lookups)

        # if self.debug_mode:
        #     draw_aligned_voxel_feats([voxel_feats])

        # voxel encoder
        voxel_feats = voxel_feats.permute(0, 1, 3, 4, 2) # (N, C, Z, X, Y) -> (N, C, X, Y, Z)
        bev_feats = self.neck_3d(voxel_feats)
        # heads
        pred = self.head_bbox_3d(bev_feats) # out: cls_score, bbox_pred, dir_cls_preds
        pred_dict = {
            "pred_cls_score": pred[0], 
            "pred_bbox": pred[1], 
            "pred_dir_cls": pred[2],
        }

        # FIXME:
        # import matplotlib.pyplot as plt
        # import numpy as np
        # from pathlib import Path
        # from scipy.spatial.transform import Rotation
        # from copious.cv.geometry import Box3d as CopiousBox3d, points3d_to_homo
        # from prefusion.dataset.utils import T4x4
        # num_imgs = len(pred_dict['pred_bbox'])
        # pred_bbox_decoded, pred_bbox_score, pred_bbox_class = self.head_bbox_3d.get_bboxes(
        #     pred_dict['pred_cls_score'], 
        #     pred_dict['pred_bbox'],
        #     pred_dict['pred_dir_cls'],
        #     num_imgs,
        #     valid=None,
        # )[0]
        # credible_idx = pred_bbox_score > 0.50
        # credible_pred_bbox_elements = []
        # for bx in pred_bbox_decoded[credible_idx]:
        #     credible_pred_bbox_elements.append(
        #         {
        #             "translation": bx[:3].cpu().numpy(),
        #             "size": bx[3:6].cpu().numpy(),
        #             "rotation": Rotation.from_euler("XYZ", [0, 0, bx[6].item()], degrees=False).as_matrix()
        #         }
        #     )

        # gt_boxes = batched_input_dict['transformables'][0]['bbox_3d']
        # save_dir = Path("./vis/box_project_onto_image") / batched_input_dict['index_infos'][0].scene_id
        # save_dir.mkdir(parents=True, exist_ok=True)
        # nrows, ncols = 3, 2
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 24))

        # def _bbox_to_corners(bx):
        #     copious_box3d = CopiousBox3d(
        #         position=bx['translation'].flatten(), 
        #         scale=np.array(bx['size']), 
        #         rotation=Rotation.from_matrix(bx['rotation'])
        #     )
        #     return copious_box3d.corners

        # def im_pts_within_image(pts, im_size):
        #     w, h = im_size
        #     return (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)

        # def K3x3(cx, cy, fx, fy):
        #     return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # def check_camera_coords_visibility_on_image(cam_coords, conservative=True):
        #     if conservative:
        #         if (cam_coords[:, 2] < 0).any():
        #             raise ValueError
        #     else:
        #         if (cam_coords[:, 2] < 0).all():
        #             raise ValueError
        #         cam_coords = cam_coords[cam_coords[:, 2] >= 0]
        #     return cam_coords


        # def check_im_coords_visibility_on_image(im_coords, im_size, conservative=True):
        #     if conservative:
        #         if not im_pts_within_image(im_coords, im_size).all():
        #             raise ValueError
        #     else:
        #         if not im_pts_within_image(im_coords, im_size).any():
        #             raise ValueError
        #         im_coords = im_coords[im_pts_within_image(im_coords, im_size)]
        #     return im_coords

        # def _3d_pts_to_uv(pts, extr, intr, im_size):
        #     points_homo = points3d_to_homo(pts)
        #     cam_extr, cam_intr = T4x4(*extr), K3x3(*intr[:4])
        #     T_cam_ego = np.linalg.inv(cam_extr)
        #     cam_coords = (T_cam_ego @ points_homo.T).T[:, :3]
        #     cam_coords = check_camera_coords_visibility_on_image(cam_coords, True)
        #     normalized_cam_coords = cam_coords[:, :2] / cam_coords[:, 2:3]
        #     im_coords = (cam_intr[:2, :2] @ normalized_cam_coords.T).T + cam_intr[:2, 2]
        #     im_coords = check_im_coords_visibility_on_image(im_coords, im_size, True)
        #     return im_coords

        # def _within_image(uv, im_size):
        #     w, h = im_size
        #     return uv[0] < w and uv[0] > 0 and uv[1] < h and uv[1] > 0

        # def _plot_bbox(ax, corners_uv, color='red'):
        #     edges = [
        #         (0, 1),
        #         (1, 2),
        #         (2, 3),
        #         (3, 0),
        #         (4, 5),
        #         (5, 6),
        #         (6, 7),
        #         (7, 4),
        #         (0, 4),
        #         (1, 5),
        #         (2, 6),
        #         (3, 7),
        #     ]

        #     for i, edge in enumerate(edges):
        #         # first 4 edges are the front face of the box, draw them with border lines
        #         # the rest 8 edges should be draw with thiner lines
        #         if i < 4:
        #             ax.plot(
        #                 (int(corners_uv[edge[0], 0]), int(corners_uv[edge[1], 0])),
        #                 (int(corners_uv[edge[0], 1]), int(corners_uv[edge[1], 1])),
        #                 color=color,
        #                 linewidth=2,
        #                 alpha=0.5,
        #             )
        #         else:
        #             ax.plot(
        #                 (int(corners_uv[edge[0], 0]), int(corners_uv[edge[1], 0])),
        #                 (int(corners_uv[edge[0], 1]), int(corners_uv[edge[1], 1])),
        #                 color=color,
        #                 linewidth=1,
        #                 alpha=0.5
        #             )


        # for i, (cam_id, im_tensor) in enumerate(camera_tensors_dict.items()):
        #     _im = (im_tensor[0].cpu().numpy().transpose(1, 2, 0) * np.array([58.395, 57.12, 57.375])[None, None, :] + np.array([123.675, 116.28, 103.53])[None, None]).astype(np.uint8)
        #     _ax = ax[i // ncols][i % ncols]
        #     _ax.imshow(_im)
        #     _ax.set_title(cam_id)
        #     extr = batched_input_dict['transformables'][0]['camera_images'].transformables[cam_id].extrinsic
        #     intr = batched_input_dict['transformables'][0]['camera_images'].transformables[cam_id].intrinsic
        #     # for bx in gt_boxes_reversed:
        #     #     corners_3d = _bbox_to_corners(bx)

        #     #     try:
        #     #         corners_uv = _3d_pts_to_uv(corners_3d, extr, intr, (880, 320))
        #     #     except ValueError:
        #     #         continue
        #     #     _plot_bbox(_ax, corners_uv, color='red')

        #     for j, bx in enumerate(credible_pred_bbox_elements):
        #         corners_3d = _bbox_to_corners(bx)

        #         try:
        #             corners_uv = _3d_pts_to_uv(corners_3d, extr, intr, (880, 320))
        #         except ValueError:
        #             continue
        #         _plot_bbox(_ax, corners_uv, color='red')


        # plt.savefig(save_dir / f"{batched_input_dict['index_infos'][0].frame_id}.jpg")
        # plt.close()

        # def get_box_bev_corners(_boxes):
        #     corners = []
        #     for _bx in _boxes:
        #         copious_box3d = CopiousBox3d(
        #             position=_bx['translation'].flatten(), 
        #             scale=np.array(_bx['size']), 
        #             rotation=Rotation.from_matrix(_bx['rotation'])
        #         )
        #         _corners = copious_box3d.corners[[0, 1, 5, 4], :2]
        #         rot = np.array([
        #             [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
        #             [np.sin(np.pi / 2), np.cos(np.pi / 2)],
        #         ])
        #         _corners = _corners @ rot.T
        #         corners.append(_corners)
        #     return corners

        # def plot_bev_box(bev_corners, color='red', linewidth=1, alpha=0.3):
        #     for corners in bev_corners:
        #         for i in range(4):
        #             plt.plot(
        #                 [corners[i][0], corners[(i + 1) % 4][0]],
        #                 [corners[i][1], corners[(i + 1) % 4][1]],
        #                 color=color,
        #                 linewidth=linewidth,
        #             )

        # gt_box_bev_corners = get_box_bev_corners(gt_boxes.elements)
        # pred_box_bev_corners = get_box_bev_corners(credible_pred_bbox_elements)
        # fig = plt.figure(figsize=(8, 8))
        # plot_bev_box(gt_box_bev_corners, color='green')
        # plot_bev_box(pred_box_bev_corners, color='red')
        # plt.gca().set_xlim([-50, 50])
        # plt.gca().set_ylim([-50, 50])
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig(save_dir / f"{batched_input_dict['index_infos'][0].frame_id}_bev.jpg")
        # plt.close()


        # FIXME:

        if mode == 'tensor':
            return pred_dict
        if mode == 'loss':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict, batched_input_dict['index_infos'][0].frame_id)
            return losses

        if mode == 'predict':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def fuse_multi_level_feats(self, mlvl_feats, target_lvl=0):
        if target_lvl != 0:
            raise NotImplementedError

        fuse_feats = [mlvl_feats[target_lvl]]
        for i in range(target_lvl + 1, len(mlvl_feats)):
            resized_feat = F.interpolate(
                mlvl_feats[i], 
                size=mlvl_feats[target_lvl].size()[2:], 
                mode="bilinear", 
                align_corners=False)
            fuse_feats.append(resized_feat)
    
        if len(fuse_feats) > 1:
            fuse_feats = torch.cat(fuse_feats, dim=1)
        else:
            fuse_feats = fuse_feats[0]

        fuse_feats = self.neck_fuse(fuse_feats)
        return fuse_feats

    def compute_losses(self, gt_dict: Dict, pred_dict: Dict, ts='-1'):
        losses = dict(loss=0)
        # aligned_bboxes = gt_dict['bbox_3d_basic']['xyz_lwh_yaw_vx_vy'].cpu().numpy()
        # aligned_bboxes[0, :, 0] -= 0.985793
        # aligned_bboxes[0, :, 2] -= 1.84019
        # aligned_bboxes.round(1)
        gt_bboxes_3d = gt_dict['bbox_3d_basic']['xyz_lwh_yaw_vx_vy']
        gt_labels_3d = gt_dict['bbox_3d_basic']['classes']
        num_imgs = len(pred_dict['pred_bbox'][0]) # 0 means the only level of the multi-level features
        bbox_losses = self.head_bbox_3d.loss(
            pred_dict['pred_cls_score'], 
            pred_dict['pred_bbox'],
            pred_dict['pred_dir_cls'],
            gt_bboxes_3d, 
            gt_labels_3d, 
            num_imgs
        )
        for loss_item, loss_value in bbox_losses.items():
            losses["loss"] += loss_value
            losses.update({loss_item: loss_value})

        # for branch in self.losses_dict:
        #     losses_branch = self.losses_dict[branch](pred_dict[branch], gt_dict[branch])
        #     losses['loss'] += losses_branch[branch + '_loss']
        #     # losses.update(losses_branch)
        #     for loss_item, loss_value in losses_branch.items():
        #         if loss_item == branch + '_loss':
        #             new_loss_name = f"{branch}/{loss_item[len(branch) + 1:]}"
        #         else:
        #             parts = loss_item[len(branch) + 1:].split("_")
        #             new_loss_name = f"{branch}/{parts[0]}/{'_'.join(parts[1:])}"
        #         losses.update({new_loss_name: loss_value})
        return losses
