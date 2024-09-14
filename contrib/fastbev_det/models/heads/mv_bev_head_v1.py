import numpy as np
import torch
from mmdet3d.models import draw_heatmap_gaussian, gaussian_radius
from prefusion.registry import MODELS
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead, circle_nms
from mmdet3d.models.utils import clip_sigmoid
from mmdet.utils import reduce_mean
from torch.cuda.amp import autocast
from mmdet3d.models.layers import nms_bev
from mmdet3d.structures import xywhr2xyxyr
from mmdet.models.utils import multi_apply
from scipy.spatial.transform import Rotation
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from prefusion.utils.utils import get_cam_corners, intrinsics_matrix, get_3d_lines, get_bev_lines, get_corners_with_angles, get_bev_lines_cylinder, \
                                get_corners_with_angles
import cv2
from copy import deepcopy
from typing import List, DefaultDict
import copy
import torch.nn as nn
from shapely.geometry import Polygon

@MODELS.register_module()
class MVBEVHeadV1(CenterHead):
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
        loss_seg_iou=dict(type='SegIouLoss',method='linear', pred_logits=False),
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
        super(MVBEVHeadV1, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            common_heads=common_heads
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
        self.loss_seg_iou = MODELS.build(loss_seg_iou)
        self.tasks = tasks
        self.all_classes = []
        for sub_classes in self.class_names:
            self.all_classes.extend(sub_classes)

        # self.task_heads = nn.ModuleList()
        # for common_head_id, num_cls in enumerate(self.num_classes):
        #     heads = copy.deepcopy(common_heads[common_head_id])
        #     heads.update(dict(class_maps=(num_cls, num_class_maps_convs)))
        #     separate_head.update(
        #         in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
        #     self.task_heads.append(MODELS.build(separate_head))
        
    
    @autocast(False)
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

    def get_targets(self, batch_data):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """
        gt_bboxes_3d, gt_labels_3d = [], []
        bev_front = 180 
        bev_left = 60
        for data in batch_data:
            bboxes_3d = []
            labels_3d = []
            for task_name in self.tasks:
                task_bbox = []
                task_label = []
                for boxes in data['transformables'][task_name['label_type']].data['elements']:
                    tmp_box = [*boxes['translation'].tolist(), *boxes['size'], Rotation.from_matrix(boxes['rotation']).as_euler("xyz", degrees=False).tolist()[-1], *boxes['velocity'].tolist()[:2]]
                    corners = get_corners_with_angles(np.array(tmp_box)[None], boxes['rotation'].T)[0][:4, :2] / 0.2
                    corners_ = deepcopy(corners)
                    corners[:, 0] = -corners_[:, 1] + bev_left
                    corners[:, 1] = -corners_[:, 0] + bev_front
                    if self.train_cfg['point_cloud_range'][0] <= tmp_box[0] <= self.train_cfg['point_cloud_range'][1] and self.train_cfg['point_cloud_range'][2] <= tmp_box[1] <= self.train_cfg['point_cloud_range'][3]:
                        task_bbox.append(corners)
                        task_label.append(boxes['class'])
                bboxes_3d.append(task_bbox)
                labels_3d.append(task_label)
            # bboxes_3d = torch.as_tensor(bboxes_3d, device='cuda')
            # labels_3d = torch.as_tensor(labels_3d, device='cuda')
            gt_bboxes_3d.append(bboxes_3d)
            gt_labels_3d.append(labels_3d)
        targets = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)

        targets = [torch.stack(target).cuda() for target in targets]

        gt_bboxes_3d, gt_labels_3d = [], []
        for data in batch_data:
            bboxes_3d = []
            labels_3d = []
            for task_name in self.tasks:
                for boxes in data['transformables'][task_name['label_type']].data['elements']:
                    labels_3d.append(self.all_classes.index(boxes['class']))
                    bboxes_3d.append([*boxes['translation'].tolist(), *boxes['size'], Rotation.from_matrix(boxes['rotation']).as_euler("xyz", degrees=False).tolist()[-1], *boxes['velocity'].tolist()[:2]])
            bboxes_3d = torch.as_tensor(bboxes_3d, device='cuda')
            labels_3d = torch.as_tensor(labels_3d, device='cuda')
            gt_bboxes_3d.append(bboxes_3d)
            gt_labels_3d.append(labels_3d)
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single_centerhead, gt_bboxes_3d, gt_labels_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        return (targets, heatmaps)


    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        # reg: 8 + z h 
        
        lW, lH = self.train_cfg['grid_size'][0], self.train_cfg['grid_size'][1]
        # results = DefaultDict(list)
        targets = []

        for indx, branch_class_name in enumerate(self.class_names):
            seg_im = np.zeros((len(branch_class_name), lH, lW))
            cen_im = np.zeros((1, lH, lW))
            reg_im = np.zeros((8, lH, lW))
            xx, yy = np.meshgrid(np.arange(lW), np.arange(lH))
            points_grid = np.array([xx, yy])

            ign_mask = np.ones((1, lH, lW))
            for points, class_name in zip(gt_bboxes_3d[indx], gt_labels_3d[indx]):
                if class_name not in branch_class_name:
                    continue
                points = np.float32(points[:, :2])
                points_int = np.round(points).astype(int)

                # gen segmentation
                cv2.fillPoly(seg_im[0], [points_int], 1)
            
                ind = branch_class_name.index(class_name)
                cv2.fillPoly(seg_im[ind], [points_int], 1)

                region_obj = cv2.fillPoly(
                    np.zeros((lH, lW)), [points_int], 1
                )

                center_point = points.mean(0)
                center_line_long = np.float32([points[[0, 1]].mean(0),
                                                points[[2, 3]].mean(0)])
                center_line_short = np.float32([points[[1, 2]].mean(0),
                                                points[[0, 3]].mean(0)])

                direction_long = np.float32(center_line_long[0] - center_line_long[1])
                direction_short = np.float32(center_line_short[0] - center_line_short[1])
                length_long = np.linalg.norm(direction_long)
                length_short = np.linalg.norm(direction_short)

                # gen regression
                norm_vec_long = direction_long / max(length_long, 1.)
                abs_direction_long = np.concatenate([
                    np.abs(direction_long), 
                    norm_vec_long[0, None] * norm_vec_long[1, None]
                ], axis=-1)

                direction_to_center_point = center_point[..., None, None] - points_grid

                reg_im[[0, 1]] = reg_im[[0, 1]] * (1 - region_obj) + region_obj * direction_to_center_point

                reg_im[2] = reg_im[2] * (1 - region_obj) + region_obj * abs_direction_long[0]
                reg_im[3] = reg_im[3] * (1 - region_obj) + region_obj * abs_direction_long[1]
                reg_im[4] = reg_im[4] * (1 - region_obj) + region_obj * length_short
                reg_im[5] = reg_im[5] * (1 - region_obj) + region_obj * abs_direction_long[2]

                reg_im[6] = reg_im[6] * (1 - region_obj) + region_obj * norm_vec_long[0]
                reg_im[7] = reg_im[7] * (1 - region_obj) + region_obj * norm_vec_long[1]

                # gen centerness
                dist2front = self.dist_point2line_along_direction(points_grid, points[[0, 1]], direction_long)
                dist2rear = self.dist_point2line_along_direction(points_grid, points[[2, 3]], direction_long)
                dist2left = self.dist_point2line_along_direction(points_grid, points[[1, 2]], direction_short)
                dist2right = self.dist_point2line_along_direction(points_grid, points[[0, 3]], direction_short)

                min_ds = np.minimum(dist2front, dist2rear) * region_obj
                # print("-"*20, min_ds.max(), "-"*20)
                if min_ds.max() >= 0.1:
                    min_ds /= min_ds.max()
                else:
                    min_ds /= 0.1
                min_dl = np.minimum(dist2left, dist2right) * region_obj
                # print("-"*20, min_dl.max(), "-"*20)
                if min_dl.max() >= 0.1:
                    min_dl /= min_dl.max()
                else:
                    min_dl /= 0.1
                

                centerness = min_ds * min_dl
                if centerness.max()>=0.1:
                    centerness /= centerness.max()
                else:
                    centerness /= 0.1

                cen_im[0] = cen_im[0] * (1 - region_obj) + region_obj * centerness
            # results[self.tasks[indx]['label_type']].append(cen_im)
            # results[self.tasks[indx]['label_type']].append(seg_im)
            # results[self.tasks[indx]['label_type']].append(reg_im)
            targets.append(torch.concat([torch.as_tensor(cen_im), torch.as_tensor(seg_im), torch.as_tensor(reg_im)], dim=0))
            
        return targets
    
    def get_targets_single_centerhead(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['bev_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device='cuda')

            anno_box = gt_bboxes_3d.new_zeros(
                (max_objs, len(self.train_cfg['code_weights'])),
                dtype=torch.float32,
                device='cuda')

            ind = gt_labels_3d.new_zeros((max_objs),
                                         dtype=torch.int64,
                                         device='cuda')
            mask = gt_bboxes_3d.new_zeros((max_objs),
                                          dtype=torch.uint8,
                                          device='cuda')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        -x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        -y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device='cuda')
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[1]
                            and 0 <= center_int[1] < feature_map_size[0]):
                        continue

                    draw_gaussian(heatmap[cls_id], [center_int[1], center_int[0]], radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert x * feature_map_size[0] + y < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = x * feature_map_size[0] + y
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    if len(task_boxes[idx][k]) > 7:
                        vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if len(task_boxes[idx][k]) > 7:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

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
        heatmaps = targets[1]
        targets = targets[0]
        return_loss = 0
        record_loss = DefaultDict(float)
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            # num_pos = targets[task_id][:, 1,...].eq(1).float().sum().item()
            # cls_avg_factor = torch.clamp(reduce_mean(
            #     targets[task_id].new_tensor(num_pos)),
            #                              min=1).item()
            
            # loss_center = self.loss_cls(preds_dict[0]['center'],
            #                              targets[task_id][:, 0, ...].unsqueeze(1),
            #                              )
            
            # for pred_class_id in range(self.num_classes[task_id]):
            #     num_pos = targets[task_id][:, [1 + pred_class_id],...].eq(1).float().sum().item()
            #     sub_cls_avg_factor = torch.clamp(reduce_mean(
            #         targets[task_id].new_tensor(num_pos)),
            #                                 min=1).item()
                
            #     loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'][:, [pred_class_id], ...],
            #                              targets[task_id][:, [1 + pred_class_id], ...],
            #                              )
            #     # loss_seg_iou = self.loss_seg_iou(preds_dict[0]['heatmap'][:, [pred_class_id], ...],
            #     #                                 targets[task_id][:, [1 + pred_class_id], ...]
            #     #                                 )
            #     return_loss += loss_heatmap
            #     # return_loss += loss_seg_iou

            #     record_loss['loss_heatmap'] += loss_heatmap
            #     # record_loss['loss_seg_iou'] += loss_seg_iou
            
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                                         min=1).item()
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         avg_factor=cls_avg_factor)

            return_loss += loss_heatmap

            mask_cen = (targets[task_id][:, 0, ...] * (0.5 * targets[task_id][:, 0, ...] + 0.5))
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor((mask_cen!=0).sum())),
                                         min=1).item()
            reg_box = targets[task_id][:, -8:, ...]
            reg_box_exp = reg_box[:, 2:5,...]
            reg_box_lin = reg_box[:, [0,1,5,6,7], ...]
            # Regression loss for reg_exp, reg_lin
            loss_reg_side = self.loss_bbox(mask_cen.unsqueeze(1) * preds_dict[0]['reg'][:, :3,...].exp(),
                                       mask_cen.unsqueeze(1) * reg_box_exp,
                                       avg_factor=cls_avg_factor)
            loss_reg_center = self.loss_bbox(mask_cen.unsqueeze(1) * preds_dict[0]['reg'][:, 3:5,...],
                                       mask_cen.unsqueeze(1) * reg_box_lin[:, 0:2, ...],
                                       avg_factor=cls_avg_factor)
            loss_reg_side_dir = self.loss_bbox(mask_cen.unsqueeze(1) * preds_dict[0]['reg'][:, 5,...],
                                       mask_cen.unsqueeze(1) * reg_box_lin[:, 2, ...],
                                       avg_factor=cls_avg_factor)

            loss_reg_head = self.loss_bbox(mask_cen.unsqueeze(1) * preds_dict[0]['reg'][:, 6:8,...],
                                       mask_cen.unsqueeze(1) * reg_box_lin[:, 3:5, ...],
                                       avg_factor=cls_avg_factor)

            return_loss += loss_reg_side
            return_loss += loss_reg_center
            return_loss += 10 * loss_reg_side_dir
            return_loss += 10 * loss_reg_head

            record_loss['loss_heatmap'] = loss_heatmap
            record_loss['loss_reg_center'] += loss_reg_center
            record_loss['loss_reg_side'] += loss_reg_side
            record_loss['loss_reg_side_dir'] += 10 * loss_reg_side_dir
            record_loss['loss_reg_head'] += 10 * loss_reg_head

        return return_loss, record_loss

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        lW, lH = self.test_cfg['grid_size'][1], self.test_cfg['grid_size'][0]
        xx, yy = np.meshgrid(np.arange(lW), np.arange(lH))
        points_grid = np.array([xx, yy])
        results = []
        for b in range(preds_dicts[0][0]['heatmap'].shape[0]):
            result_bboxes = []
            result_labels = []
            for type_index in range(len(preds_dicts)):
                cls_pred = preds_dicts[type_index][0]['heatmap'][b].sigmoid().cpu().numpy()
                reg_pred = preds_dicts[type_index][0]['reg'][b]
                pred_objs = self.solve_rdet_from_feat(cls_pred, None, reg_pred[:3, ...].exp().cpu().numpy(), reg_pred[3:, ...].cpu().numpy(), points_grid, scale=1)
                for pred_obj, confs in pred_objs:
                    result_bboxes.append(pred_obj)  # TODO convert to ego coord
                    result_labels.append(self.class_names[type_index][confs[1:].argmax(0)-1])
            results.append([result_bboxes, result_labels])    
        
        return results
    
    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
    
    def _add_new_axis(self, arr, n):
        for _ in range(n):
            arr = arr[..., None]
        return arr

    def vec_point2line_along_direction(self, point, line, direction):
        point = np.float32(point)
        n_extra_dim = len(point.shape) - 1
        line = self._add_new_axis(np.float32(line), n_extra_dim)
        vec = self._add_new_axis(np.float32(direction), n_extra_dim)
        vec_l = line[1] - line[0]
        vec_p = line[1] - point
        C1 = vec[1] * vec_l[0] - vec[0] * vec_l[1]
        if np.abs(C1) < 1e-5:
            return np.full_like(point, 0)
        C2 = (vec[1] * vec_p[0] - vec[0] * vec_p[1]) / C1
        return vec_p - vec_l * C2

    def dist_point2line_along_direction(self, point, line, direction):
        vec = self.vec_point2line_along_direction(point, line, direction)
        return np.linalg.norm(vec, axis=0)

    def solve_rdet_from_feat(self, cls_pred, cen_pred, reg_exp_pred, reg_lin_pred, points_grid, pre_conf=0.3, scale=1):
        valid_centerness = cen_pred[0] * cls_pred[0]

        valid_points_map = valid_centerness > pre_conf
        valid_points = points_grid[:, valid_points_map]

        # conf = cls_pred[0][valid_points_map]
        seg_scores = cls_pred[:, valid_points_map]

        dc_x = reg_lin_pred[0][valid_points_map]
        dc_y = reg_lin_pred[1][valid_points_map]

        abs_l_x = reg_exp_pred[0][valid_points_map]
        abs_l_y = reg_exp_pred[1][valid_points_map]
        ss = reg_exp_pred[2][valid_points_map]
        nl_xy = reg_lin_pred[2][valid_points_map] / scale

        head_x = reg_lin_pred[3][valid_points_map] / scale
        head_y = reg_lin_pred[4][valid_points_map] / scale

        center_x = valid_points[0] + dc_x
        center_y = valid_points[1] + dc_y

        l_x = abs_l_x
        l_y = self.sign(nl_xy) * abs_l_y

        vec_l = np.float32([l_x, l_y])
        vec_head = np.float32([head_x, head_y])
        vec_cross = np.sum(vec_l * vec_head, axis=0)

        head_sign = self.sign(vec_cross)

        length_l = np.linalg.norm(vec_l, axis=0)
        norm_vec_l = vec_l / length_l

        norm_vec_s = norm_vec_l[[1, 0]] * [[-1], [1]]

        points_0 = [center_x, center_y] + 0.5 * (vec_l + norm_vec_s * ss)
        points_1 = [center_x, center_y] + 0.5 * (vec_l - norm_vec_s * ss)
        points_2 = [center_x, center_y] - 0.5 * (vec_l + norm_vec_s * ss)
        points_3 = [center_x, center_y] - 0.5 * (vec_l - norm_vec_s * ss)

        points = np.float32([points_0, points_1, points_2, points_3])

        # SOFT GROUP NMS
        scores = valid_centerness[valid_points_map]
        pred_objs = self.polygon_group_nms(points, scores, seg_scores, head_sign)

        return pred_objs

    def polygon_group_nms(self, points, scores, seg_scores, head_sign, iou_thresh=0.5):
        ranked_ind = np.argsort(scores)[::-1]

        kept_groups = []
        kept_inds = []

        for i in ranked_ind[:]:
            if i not in kept_inds:
                quad_i = points[:, :2, i]
                quad_i_polygon = Polygon(quad_i)
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_ind[:]:
                    if j not in kept_inds:
                        quad_j = points[:, :2, j]
                        quad_j_polygon = Polygon(quad_j)

                        inter_area = quad_i_polygon.intersection(quad_j_polygon).area
                        union_area = quad_i_polygon.union(quad_j_polygon).area
                        iou = inter_area / max([union_area, 1e-5])
                        if iou > iou_thresh:
                            kept_inds.append(j)
                            grouped_inds.append(j)

        pred_objs = []

        for group in kept_groups[:]:    
            ind_0 = group[0]
            obj_0 = points[:, :2, ind_0]
            obj_0_line_01 = obj_0[1, :2] - obj_0[0, :2]

            obj_group = [obj_0]
            for ind in group[1:]:
                obj_line_01 = points[1, :2, ind] - points[0, :2, ind]
                if np.sum(obj_line_01 * obj_0_line_01) < 0:
                    obj = points[[0,1,2,3], :2, ind]
                else:
                    obj = points[:, :2, ind]
                obj_group.append(obj)

            obj_group = np.float32(obj_group)        
            mean_obj = obj_group.mean(0)
            # print(seg_scores[:, group].shape)
            mean_confs = seg_scores[:, group].mean(1)
            mean_head = head_sign[group].mean(0)
            if mean_head < 0:
                mean_obj = mean_obj[[0,1,2,3]]
            # print(mean_confs)
            # print(mean_obj)
            pred_objs.append([mean_obj, mean_confs])

        return pred_objs

    def sign(self, x):
        return 2 * (x >= 0) - 1

    def show_results(self, all_pred, data, **kwargs):
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
            for j in range(len(all_pred[b][0])):
                # class_cat = sub_cla[j]
                pred_box = all_pred[b][0][j].tolist()
                # tmp_center = [-(pred_box[1] - 36), -(pred_box[0] - 12), pred_box[2]]
                tmp_center = pred_box[:3]
                tmp_box = np.array(tmp_center + pred_box[3:6] + [0] + [0, 0])
                tmp_r = Rotation.from_euler('xyz', angles=[0, 0, pred_box[6]], degrees=False).as_matrix()
                corners = get_corners_with_angles(tmp_box[None], tmp_r.T)[0][:, :3]
                # corners[:4, 2] += np.array(pred_box[6:10])
                # corners[4:, 2] += np.array(pred_box[6:10])
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
            # plt.savefig(f"work_dirs/vis_data/{frame_ids[b]}.jpg")
            plt.close()