import numba
import numpy as np
import torch
from mmengine.registry import MODELS
from mmdet3d.models.utils.gaussian import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead, circle_nms
from mmdet3d.models.utils import clip_sigmoid
from mmdet.utils import reduce_mean
from mmengine.runner import autocast
from mmdet3d.models.layers import nms_bev
from mmdet3d.structures.bbox_3d.utils import xywhr2xyxyr
import copy
import torch.nn as nn
from scipy.spatial.transform import Rotation
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from prefusion.utils.utils import get_cam_corners, intrinsics_matrix, get_3d_lines, get_bev_lines, get_corners_with_angles, get_bev_lines_cylinder
import cv2
import time


@MODELS.register_module()
class MVBEVHead(CenterHead):
    """Head for BevDepth.

    Args:
        in_channels(int): Number of channels after bev_neck.
        tasks(dict): Tasks for head.
        bbox_coder(dict): Config of bbox coder.
        common_heads(dict): Config of head for each task.
        loss_cls(dict): Config of classification loss.
        loss_bbox(dict): Config of regression loss.
        gaussian_overlap(float): Gaussian overlap used for `get_targets`.
        min_radius(int): Min radius used for `get_targets`.
        train_cfg(dict): Config used in the training process.
        test_cfg(dict): Config used in the test process.
        bev_backbone_conf(dict): Cnfig of bev_backbone.
        bev_neck_conf(dict): Cnfig of bev_neck.
    """

    def __init__(
        self,
        in_channels=256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        bev_backbone_conf=None,
        bev_neck_conf=None,
        separate_head=dict(type='SeparateHead',
                           init_bias=-2.19,
                           final_kernel=3),
        num_class_maps_convs=2,
        share_conv_channel: int = 64,
    ):
        super(MVBEVHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel
        )
        self.trunk = MODELS.build(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = MODELS.build(bev_neck_conf)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.common_head = common_heads

        self.task_heads = nn.ModuleList()
        for common_head_id, num_cls in enumerate(self.num_classes):
            heads = copy.deepcopy(common_heads[common_head_id])
            heads.update(dict(class_maps=(num_cls, num_class_maps_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(MODELS.build(separate_head))

    # @autocast(False)
    def forward(self, x):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        x = x.float()
        # FPN
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        ret_values = super().forward(fpn_output)
        return ret_values

    def loss(self, targets, preds_dicts, **kwargs):
        """Loss function for BEVDepthHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return_loss = 0
        target_keys = list(targets)
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            target = targets[target_keys[task_id]]
            heatmaps, anno_boxes, gridzs, class_maps = target['heatmap'], target['anno_boxes'], target['gridzs'], target['class_maps'],  
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps.eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps.new_tensor(num_pos)),
                                         min=1).item()
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps,
                                         avg_factor=cls_avg_factor)
            
            preds_dict[0]['class_maps'] = clip_sigmoid(preds_dict[0]['class_maps'])
            num_pos = class_maps.eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                class_maps.new_tensor(num_pos)),
                                         min=1).item()
            loss_class_maps = self.loss_cls(preds_dict[0]['class_maps'],
                                         class_maps,
                                         avg_factor=cls_avg_factor)

            mask = torch.ones_like(anno_boxes).to(device=anno_boxes.device)
            num = torch.as_tensor(mask.float().sum()).to(anno_boxes.device)
            num = torch.clamp(reduce_mean(num),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(anno_boxes)).float()
            mask *= isnotnan
            # code_weights = self.train_cfg['code_weights']
            bbox_weights = mask #  * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(preds_dict[0]['anno_boxes'],
                                       anno_boxes,
                                       bbox_weights,
                                       avg_factor=num)

            mask = torch.ones_like(gridzs).to(device=gridzs.device)
            num = torch.as_tensor(mask.float().sum()).to(anno_boxes.device)
            num = torch.clamp(reduce_mean(num),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(gridzs)).float()
            mask *= isnotnan
            # code_weights = self.train_cfg['code_weights']
            bbox_weights = mask #  * mask.new_tensor(code_weights)
            loss_gridzs = self.loss_bbox(preds_dict[0]['gridzs'],
                                       gridzs,
                                       bbox_weights,
                                       avg_factor=num)

            return_loss += loss_bbox
            return_loss += loss_heatmap
            return_loss += loss_class_maps
            return_loss += loss_gridzs
        return return_loss, dict()

    def get_bboxes(self, preds_dicts, img_metas, debug_target=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        voxel_size = [0.2, 0.2, 8]
        all_pred = []
        all_classes = []
        all_scores = []
        task_head_names = list(debug_target.keys())
        for b in range(preds_dicts[0][0]['heatmap'].shape[0]):
            single_frame_preds = []
            single_frame_class = []
            single_frame_scores = []
            for pred_idx, preds_dict in enumerate(preds_dicts):
                heatmaps, anno_box, _, class_maps = clip_sigmoid(preds_dict[0]['heatmap'])[b], preds_dict[0]['anno_boxes'][b], preds_dict[0]['gridzs'][b], clip_sigmoid(preds_dict[0]['class_maps'])[b]
                # heatmaps, anno_box, _, class_maps = debug_target[task_head_names[pred_idx]]['heatmap'][b], debug_target[task_head_names[pred_idx]]['anno_boxes'][b], debug_target[task_head_names[pred_idx]]['gridzs'][b], debug_target[task_head_names[pred_idx]]['class_maps'][b]
                batch_heatmaps = (heatmaps * class_maps[0])[0]
                val_point_map = torch.zeros_like(batch_heatmaps).to(heatmaps.device).to(torch.int32)
                topk_scores, inds = torch.topk(batch_heatmaps.view(-1), 10)
                for ind in inds:
                    val_point_map[ind // 120, ind % 120] = 1
                val_point_map = ((batch_heatmaps > 0.1) & val_point_map).to(torch.bool)
                # cls_point  = class_maps[val_point_map]
                scores = batch_heatmaps[val_point_map]
                seg_scores = class_maps[:, val_point_map]
                xx,yy = torch.meshgrid(torch.arange(120), torch.arange(240), indexing='xy')
                points_grid = torch.stack([xx, yy], dim=0).to(heatmaps.device).to(torch.int32)
                val_points_grid=points_grid[:, val_point_map]
                scores = batch_heatmaps[val_point_map]
                batch_anno_boxes = anno_box[:, val_point_map]
                batch_offsetxy = batch_anno_boxes[:2]
                batch_xy = (val_points_grid + batch_offsetxy) * voxel_size[0]
                batch_z = batch_anno_boxes[2].unsqueeze(0)
                xyz = torch.concat([batch_xy, batch_z], axis=0)
                if pred_idx == 0 or pred_idx==1:
                    batch_dim = torch.exp(batch_anno_boxes[3:6])
                    batch_gridz = batch_anno_boxes[6:10]
                    batch_yaw = batch_anno_boxes[10:12]
                    batch_2yaw = batch_anno_boxes[12:14]
                    # yaw_2rot = torch.arctan2(np.sqrt((1-batch_2yaw[1])/2), np.sqrt((batch_2yaw[1]+1)/2))
                    yaw_rot = torch.arctan2(batch_yaw[0], batch_yaw[1])
                    batch_velocity = batch_anno_boxes[14:]
                    res = self.rbox_group_nms(xyz, batch_dim, scores, seg_scores, batch_gridz, yaw_rot, batch_velocity)
                    for c in range(len(res)):
                        single_frame_class.append(self.class_names[pred_idx][res[c][-1].to(torch.int32)])
                        tmp_center = res[c][:2].clone()# [-(res[c][1] - self.train_cfg['to_mv_coord'][0]), -(res[c][0] - self.train_cfg['to_mv_coord'][1]), res[c][2]]
                        res[c][0] = -(tmp_center[1] - self.train_cfg['to_mv_coord'][0])
                        res[c][1] = -(tmp_center[0] - self.train_cfg['to_mv_coord'][1])
                        # center, dims, gridz(4), yaw, velocity
                        single_frame_preds.append(res[c][:-2].unsqueeze(0))
                        single_frame_scores.append(res[c][-2].unsqueeze(0))
                    # R = Rotation.from_euler("XYZ", angles=(0, 0, yaw_rot[0, 218, 100].cpu().numpy()), degrees=False).as_matrix()
                    # xyz lwh z1 z2 z3 z4 yaw vx vy vz
                elif pred_idx == 2:
                    dim_w = torch.exp(batch_anno_boxes[3])
                    dim_h = torch.exp(batch_anno_boxes[4])
                    batch_gridz = batch_anno_boxes[5:9]
                    batch_yaw = batch_anno_boxes[9:11]
                    batch_2yaw = batch_anno_boxes[11:13]
                    # yaw_2rot = np.arctan2(np.sqrt((1-batch_2yaw[1])/2), np.sqrt((batch_2yaw[1]+1)/2))
                    yaw_rot = torch.arctan2(batch_yaw[0], batch_yaw[1])
                    batch_velocity = batch_anno_boxes[13:16]
                    res = self.rbox_group_nms_cube(xyz, [dim_w, dim_h], scores, seg_scores, batch_gridz, yaw_rot, batch_velocity)
                    for c in range(len(res)):
                        single_frame_class.append(self.class_names[pred_idx][res[c][-1].to(torch.int32)])
                        tmp_center = res[c][:2].clone()
                        res[c][0] = -(tmp_center[1] - self.train_cfg['to_mv_coord'][0])
                        res[c][1] = -(tmp_center[0] - self.train_cfg['to_mv_coord'][1])
                        # center, dims, gridz(4), yaw, velocity
                        single_frame_preds.append(res[c][:-2].unsqueeze(0))
                        single_frame_scores.append(res[c][-2].unsqueeze(0))
                else:
                    # this code maybe have a bug 443
                    batch_dim = torch.exp(batch_anno_boxes[3])  # r
                    cylinder_height = torch.exp(batch_anno_boxes[4]) # cylinder_height
                    batch_yaw = batch_anno_boxes[5:7]
                    yaw_rot = torch.arctan2(batch_yaw[0], batch_yaw[1])
                    batch_velocity = batch_anno_boxes[9:12]
                    res = self.rbox_group_nms_cylinder(xyz, [batch_dim, cylinder_height], scores, seg_scores, yaw_rot, batch_velocity)
                    for c in range(len(res)):
                        single_frame_class.append(self.class_names[pred_idx][res[c][-1].to(torch.int32)])
                        tmp_center = res[c][:2].clone()
                        res[c][0] = -(tmp_center[1] - self.train_cfg['to_mv_coord'][0])
                        res[c][1] = -(tmp_center[0] - self.train_cfg['to_mv_coord'][1])
                        # center, dims, gridz(4), yaw, velocity
                        single_frame_preds.append(res[c][:-2].unsqueeze(0))
                        single_frame_scores.append(res[c][-2].unsqueeze(0))
            if len(single_frame_preds)==0:
                all_pred.append(torch.zeros(0,14).to(heatmaps.device))
                all_classes.append([])
                all_scores.append(torch.zeros(0).to(heatmaps.device))
            else:
                all_pred.append(torch.concat(single_frame_preds, axis=0))
                all_classes.append(single_frame_class)
                all_scores.append(torch.concat(single_frame_scores, axis=0))
        return all_pred, all_classes, all_scores

    def rotate_points_xyz(self, points, r_matrix):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:
        """
        r_matrix = r_matrix.reshape(1,3,3)
        points_rot = np.matmul(points[:, :, 0:3], r_matrix)
        points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
        return points_rot

    def get_corners_with_angles(self, boxes3d, R_matrix):
        template = (np.array((
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
        )) / 2)
        # R_matrix = Rotation.from_euler("xyz",angles=Rotation.from_matrix(R_matrix).as_euler("XYZ", degrees=False), degrees=False).as_matrix()
        corners3d = np.tile(boxes3d[:, None, 3:6],
                            [1, 8, 1]) * template[None, :, :]
        corners3d = self.rotate_points_xyz(corners3d.reshape(-1, 8, 3),
                                        R_matrix).reshape(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]

        return corners3d

    def is_in_ellipse3d(self, a,b,c,xyz, xyz0, yaw_rot):
        # if (xyz[0]-xyz0[0])**2/a**2 + (xyz[1]-xyz0[1])**2/b**2 + (xyz[2]-xyz0[2])**2/c**2 > 1:
        #     return False
        # else:
        #     return True
        x, y, z = xyz0[0], xyz0[1], xyz0[2]
        x_c, y_c, z_c = xyz[0],xyz[1],xyz[2]
        
        # 旋转角度theta的cos和sin值
        cos_theta = torch.cos(yaw_rot)
        sin_theta = torch.sin(yaw_rot)
        
        # 将点旋转回椭球未旋转位置的坐标
        x_prime = cos_theta * (x - x_c) + sin_theta * (y - y_c)
        y_prime = -sin_theta * (x - x_c) + cos_theta * (y - y_c)
        z_prime = z - z_c
        
        # 判断点是否在椭球内部
        value = (x_prime**2 / a**2) + (y_prime**2 / b**2) + (z_prime**2 / c**2)
        return value <= 1

    def is_in_cuboid(self, length, width, height, xyz, xyz0, yaw_rot):
        """
        判断一个点是否在旋转的长方体内部
        
        参数:
        length: 长方体的长度（x轴方向）
        width: 长方体的宽度（y轴方向）
        height: 长方体的高度（z轴方向）
        xyz: 长方体中心的坐标 [x_c, y_c, z_c]
        xyz0: 待判断点的坐标 [x, y, z]
        yaw_rot: 绕z轴的旋转角度（偏航角，单位：弧度）
        
        返回:
        Boolean: 如果点在长方体内部或表面上返回True，否则返回False
        """
        x, y, z = xyz0
        x_c, y_c, z_c = xyz
        
        # 旋转角度theta的cos和sin值
        cos_theta = torch.cos(yaw_rot)
        sin_theta = torch.sin(yaw_rot)
        
        # 将点的坐标转换到长方体的局部坐标系
        x_prime = cos_theta * (x - x_c) + sin_theta * (y - y_c)
        y_prime = -sin_theta * (x - x_c) + cos_theta * (y - y_c)
        z_prime = z - z_c
        
        # 长方体的半长、半宽、半高
        half_length = length / 2
        half_width = width / 2
        half_height = height / 2
        
        # 判断点是否在长方体内部
        return (torch.abs(x_prime) <= half_length and 
                torch.abs(y_prime) <= half_width and 
                torch.abs(z_prime) <= half_height)

    def rbox_group_nms(self, xyz, dims, scores, seg_scores, gridz, yaw_rot, velocity, ratio=2):
        ranked_ind = torch.argsort(scores, descending=True)

        kept_groups = []
        kept_inds = []

        for i in ranked_ind[:]:
            if i not in kept_inds:
                center_i = xyz[:, i]
                dim = dims[:, i]
                # axis_length = min(dim[0], dim[1], dim[2])
                kept_inds.append(i)
                grouped_inds = [i]
                yaw = yaw_rot[i]
                kept_groups.append(grouped_inds)
                for j in ranked_ind[:]:
                    if j not in kept_inds:
                        center_j = xyz[:, j]
                        # dist_ij = np.linalg.norm(center_i - center_j)
                        # if dist_ij < axis_length * ratio: 
                        #     kept_inds.append(j)
                        #     grouped_inds.append(j)
                        if self.is_in_cuboid(dim[0], dim[1], dim[2], center_i, center_j, yaw):
                            kept_inds.append(j)
                            grouped_inds.append(j)

        pred_objs = []

        for group in kept_groups[:]:
            mean_confs = scores.unsqueeze(0)[:, group].mean(1)
            box_class = seg_scores[:, group].mean(1)
            mean_center_xyz = xyz[:, group].mean(1)
            mean_gridz = gridz[:, group].mean(1)
            mean_yaw_rot = yaw_rot.unsqueeze(0)[:, group].mean(1)
            mean_dim = dims[:, group].mean(1)
            mean_velocity = velocity[:, group].mean(1)
            # pred_objs.append([*mean_center_xyz.tolist(), *mean_dim.tolist(), *mean_gridz.tolist(), mean_yaw_rot, *mean_velocity.tolist(), mean_confs, box_class])
            pred_objs.append(torch.concat([mean_center_xyz, mean_dim, mean_gridz, mean_yaw_rot, mean_velocity, mean_confs, box_class[1:].argmax(0).unsqueeze(0)]))
        return pred_objs

    def rbox_group_nms_cube(self, xyz, dims, scores, seg_scores, gridz, yaw_rot, velocity, ratio=2):
        ranked_ind = torch.argsort(scores, descending=True)

        kept_groups = []
        kept_inds = []

        for i in ranked_ind[:]:
            if i not in kept_inds:
                center_i = xyz[:, i]
                dim_w = dims[0][i]
                dim_h = dims[1][i]
                # axis_length = min(dim[0], dim[1], dim[2])
                kept_inds.append(i)
                grouped_inds = [i]
                yaw = yaw_rot[i]
                kept_groups.append(grouped_inds)
                for j in ranked_ind[:]:
                    if j not in kept_inds:
                        center_j = xyz[:, j]
                        # dist_ij = np.linalg.norm(center_i - center_j)
                        # if dist_ij < axis_length * ratio: 
                        #     kept_inds.append(j)
                        #     grouped_inds.append(j)
                        if self.is_in_cuboid(dim_w, dim_w, dim_h, center_i, center_j, yaw):
                            kept_inds.append(j)
                            grouped_inds.append(j)

        pred_objs = []

        for group in kept_groups[:]:
            mean_confs = scores.unsqueeze(0)[:, group].mean(1)
            box_class = seg_scores[:, group].mean(1)
            mean_center_xyz = xyz[:, group].mean(1)
            mean_gridz = gridz[:, group].mean(1)
            mean_yaw_rot = yaw_rot.unsqueeze(0)[:, group].mean(1)
            mean_dim = [dims[0].unsqueeze(0)[:, group].mean(1), dims[0].unsqueeze(0)[:, group].mean(1), dims[1].unsqueeze(0)[:, group].mean(1)]
            mean_velocity = velocity[:, group].mean(1)
            # pred_objs.append([*mean_center_xyz.tolist(), *mean_dim, *mean_gridz.tolist(), mean_yaw_rot, *mean_velocity.tolist(), mean_confs, box_class])
            pred_objs.append(torch.concat([mean_center_xyz, torch.concat(mean_dim, axis=0), mean_gridz, mean_yaw_rot, mean_velocity, mean_confs, box_class[1:].argmax(0).unsqueeze(0)]))
        return pred_objs

    def rbox_group_nms_cylinder(self, xyz, dims, scores, seg_scores, yaw_rot, velocity, ratio=2):
        ranked_ind = torch.argsort(scores, descending=True)

        kept_groups = []
        kept_inds = []

        for i in ranked_ind[:]:
            if i not in kept_inds:
                center_i = xyz[:, i]
                dim_r = dims[0][i]
                dim_h = dims[1][i]
                kept_inds.append(i)
                grouped_inds = [i]
                yaw = yaw_rot[i]
                kept_groups.append(grouped_inds)
                for j in ranked_ind[:]:
                    if j not in kept_inds:
                        center_j = xyz[:, j]
                        # dist_ij = np.linalg.norm(center_i - center_j)
                        # if dist_ij < axis_length * ratio: 
                        #     kept_inds.append(j)
                        #     grouped_inds.append(j)
                        if self.is_in_cuboid(dim_r, dim_r, dim_h, center_i, center_j, yaw):
                            kept_inds.append(j)
                            grouped_inds.append(j)

        pred_objs = []

        for group in kept_groups[:]:
            mean_confs = scores.unsqueeze(0)[:, group].mean(1)
            box_class = seg_scores[:, group].mean(1)
            mean_center_xyz = xyz[:, group].mean(1)
            mean_yaw_rot = yaw_rot.unsqueeze(0)[:, group].mean(1)
            mean_dim = [dims[0].unsqueeze(0)[:, group].mean(1), dims[1].unsqueeze(0)[:, group].mean(1)]
            mean_velocity = velocity[:, group].mean(1)
            mean_gridz = torch.tensor([0,0,0,0]).to(mean_confs.device)
            # pred_objs.append([*mean_center_xyz.tolist(), *mean_dim, mean_yaw_rot, *mean_velocity.tolist(), mean_confs, box_class])
            pred_objs.append(torch.concat([mean_center_xyz, mean_dim, mean_gridz, mean_yaw_rot, mean_velocity, mean_confs, box_class[1:].argmax(0).unsqueeze(0)]))

        return pred_objs

    def show_results(self, all_pred, all_classes, data, frame_ids, **kwargs):
        """Visualize the results to an image.

        Args:
            results (list[dict]): Predicted results.
            img_metas (list[dict]): Point cloud and image's meta info.
            out_dir (str): Path of output directory, None means not saving
                the image.
        """
        for b in range(len(all_pred)):
            gt_corners = []
            gt_cylinders = []
            # for i,sub_boxes in enumerate(all_pred[b]):
            # for i in range(all_pred[b].shape[0]):
            #     sub_cla = all_classes[b]
            #     sub_boxes = all_pred[b][i]
            for j in range(len(all_pred[b])):
                # class_cat = sub_cla[j]
                pred_box = all_pred[b][j].cpu().numpy().tolist()
                # tmp_center = [-(pred_box[1] - 36), -(pred_box[0] - 12), pred_box[2]]
                tmp_center = pred_box[:3]
                tmp_box = np.array(tmp_center + pred_box[3:6] + [0] + [0, 0])
                tmp_r = Rotation.from_euler('xyz', angles=[0, 0, pred_box[10]], degrees=False).as_matrix()
                corners = get_corners_with_angles(tmp_box[None], tmp_r.T)[0][:, :3]
                corners[:4, 2] += np.array(pred_box[6:10])
                corners[4:, 2] += np.array(pred_box[6:10])
                gt_corners.append(corners)
            # else:
            #     for j in range(len(sub_boxes)):
            #         class_cat = sub_cla[j]
            #         pred_box = sub_boxes[j]
            #         tmp_center = [-(pred_box[1] - 36), -(pred_box[0] - 12), pred_box[2]]
            #         tmp_r = Rotation.from_euler('xyz', angles=[0, 0, pred_box[5]], degrees=False).as_matrix()
                    
            #         corners = plot_cylinder(tmp_center, np.array([0,0,1]), radius=pred_box[3], height=pred_box[4])
            #         lines = get_bev_lines_cylinder(corners)
            #         gt_cylinders.append(corners)
                
            # gt_corners = [] 
            # gt_cylinders = []
            # for label_type in ['bbox_3d', 'bbox_bev', 'square_3d']:
            #     label_data = ori_data[b]['transformables'][label_type].data['elements']
            #     for box_3d in label_data:
            #         box = np.array(box_3d['translation'].reshape(-1).tolist() + box_3d['size'] + [Quaternion(matrix=box_3d['rotation']).yaw_pitch_roll[0]] + [0, 0])
            #         if np.linalg.norm(box[:2]) <= 1e5:
            #             corners = get_corners_with_angles(box[None], box_3d['rotation'].T)[0]
            #             gt_corners.append(corners)
            
            plt.figure(figsize=(24, 8))
            row = 5 

            # for i, k in enumerate(IMG_KEYS):
            img_num = 0
            for key_camera in data.keys():
                imgs = torch.split(data[key_camera]['imgs'], self.each_camera_nums[key_camera.split('_')[0]])[b]
                extrinsics = torch.split(data[key_camera]['extrinsic'], self.each_camera_nums[key_camera.split('_')[0]])[b]
                intrinsics = torch.split(data[key_camera]['intrinsic'], self.each_camera_nums[key_camera.split('_')[0]])[b]
                for idx in range(len(imgs)):
                    # Draw camera views
                    fig_idx = img_num + 1 if img_num < row else img_num + 2
                    img_num += 1
                    plt.subplot(2, 6, fig_idx)

                    # Set camera attributes
                    plt.title(f"{key_camera} {idx}")
                    plt.axis('off')
                    H, W = imgs.shape[-2:]
                    plt.xlim(0, W)
                    plt.ylim(H, 0)
                    img = imgs[idx].cpu().numpy().transpose(1, 2, 0) - (imgs[idx].min().cpu().numpy())
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # plt.imsave(f"{key_camera}_{idx}.jpg",img)

                    # Draw images
                    plt.imshow(img)

                    # Draw 3D gt
                    for corners in gt_corners:
                        cam_corners = get_cam_corners(
                            corners,
                            extrinsics[idx][:3, 3].cpu().numpy(),
                            Quaternion(matrix=extrinsics[idx][:3, :3].cpu().numpy(), atol=1e-06),
                            intrinsics_matrix(intrinsics[idx][:4].cpu().numpy()))
                        lines = get_3d_lines(cam_corners)
                        for line in lines:
                            plt.plot(line[0],
                                    line[1],
                                    c=cm.get_cmap('tab10')(4)
                                    )
                    for corners in gt_cylinders:
                        cam_corners = get_cam_corners(
                            corners,
                            extrinsics[idx][:3, 3].cpu().numpy(),
                            Quaternion(matrix=extrinsics[idx][:3, :3].cpu().numpy(), atol=1e-06),
                            intrinsics_matrix(intrinsics[idx][:4].cpu().numpy()))
                        
                        bottom_cam_corner = cam_corners[:100, :]
                        up_cam_corner = cam_corners[100:, :]

                        plt.plot(bottom_cam_corner[:, 0],
                                bottom_cam_corner[:, 1],
                                c=cm.get_cmap('tab10')(4)
                                )
                        plt.plot(up_cam_corner[:, 0],
                                up_cam_corner[:, 1],
                                c=cm.get_cmap('tab10')(4)
                                )
                        for i in range(0, 100, 10):
                            plt.plot([bottom_cam_corner[i, 0], up_cam_corner[i, 0]], [bottom_cam_corner[i, 1], up_cam_corner[i, 1]], 'g-')
                    # for box in info['box_2d'][k]['box']:
                    #     x1, y1, x2, y2 = box
                    #     w = x2 - x1
                    #     h = y2 - y1
                    #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    #     plt.gca().add_patch(rect)
            # Draw BEV
            plt.subplot(1, 6, 6)

            # Set BEV attributes
            plt.title('LIDAR_TOP')
            plt.axis('equal')
            plt.xlim(-15, 15)
            plt.ylim(-15, 40)

            # BEV box ego 是x朝前 y朝左,  可视化出来到图上是x朝右，y朝前，对应到图上x=-y,y=x 
            # Draw BEV GT boxes
            for corners in gt_corners:
                lines = get_bev_lines(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')
            for corners in gt_cylinders:
                lines = get_bev_lines_cylinder(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')
            # Set legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(),
                    by_label.keys(),
                    loc='upper right',
                    framealpha=1)

            # Save figure
            plt.tight_layout(w_pad=0, h_pad=2)
            # plt.show()
            plt.savefig(f"work_dirs/vis_data/{frame_ids[b]}.jpg")
            plt.close()