from mmengine.optim import OptimWrapper
import torch
import torch.nn.functional as F
import mmengine
import mmcv
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from mmengine.runner import autocast
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmengine import ConfigDict
from mmdet3d.models import Base3DDetector
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from prefusion.utils.utils import get_cam_corners, intrinsics_matrix, get_3d_lines, get_bev_lines, get_corners_with_angles, get_bev_lines_cylinder
import cv2
from ..utils.utils import transformation_from_parameters
import pdb


@MODELS.register_module()
class MatrixVT_Det(BaseModel):
    """Implementation of MatrixVT for Object Detection.

        Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    def __init__(self, 
                 backbone_conf_fish,
                 backbone_conf_pv,
                 head_conf, 
                 pv_bev_fusion, 
                 is_train_depth=False, 
                 data_preprocessor=None, 
                 mono_depth=None,
                 frame_start_end_ids=[1,0,2],
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone_fish = MODELS.build(backbone_conf_fish)
        self.backbone_pv = MODELS.build(backbone_conf_pv)
        self.head = MODELS.build(head_conf)
        self.PV_BEV_Fusion = MODELS.build(pv_bev_fusion)
        self.is_train_depth = is_train_depth
        self.downsample_factor_fish = backbone_conf_fish.get('downsample_factor', 16)
        self.dbound_fish = backbone_conf_fish.get("d_bound_fish", [2.0, 58.0, 0.5])
        self.depth_channels_fish = int(
            (self.dbound_fish[1] - self.dbound_fish[0]) / self.dbound_fish[2])
        self.downsample_factor_pv = self.downsample_factor_front = backbone_conf_pv.get('downsample_factor', 16)
        self.dbound_pv = backbone_conf_pv.get("d_bound_pv", [2.0, 58.0, 0.5])
        self.depth_channels_pv = int(
            (self.dbound_pv[1] - self.dbound_pv[0]) / self.dbound_pv[2])
        self.dbound_front = backbone_conf_pv.get("d_bound_front", [2.0, 58.0, 0.5])
        self.depth_channels_front = int(
            (self.dbound_front[1] - self.dbound_front[0]) / self.dbound_front[2])
        self.mono_depth_net = MODELS.build(mono_depth)
        self.frame_start_end_ids = frame_start_end_ids
        self.mono_depth_net.frame_start_end_ids = frame_start_end_ids
     
    def get_depth_loss(self, depth_labels, depth_preds, camera_type):
        depth_labels = self.get_downsampled_gt_depth(depth_labels, camera_type)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, getattr(self, f"depth_channels_{camera_type}"))
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        assert depth_preds.device == depth_labels.device == fg_mask.device
        with autocast(device_type=depth_preds.device, enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return depth_loss

    def get_downsampled_gt_depth(self, gt_depths, camera_type):
        """
        Input:
            gt_depths: [B, N, H, W]
            camera_type: [fish, pv, front]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // getattr(self, f"downsample_factor_{camera_type}"),
            getattr(self, f"downsample_factor_{camera_type}"),
            W // getattr(self, f"downsample_factor_{camera_type}"),
            getattr(self, f"downsample_factor_{camera_type}"),
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, getattr(self, f"downsample_factor_{camera_type}") * getattr(self, f"downsample_factor_{camera_type}"))
        gt_depths_tmp = torch.where((gt_depths <= 0.0) | (gt_depths==-1),
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // getattr(self, f"downsample_factor_{camera_type}"),
                                   W // getattr(self, f"downsample_factor_{camera_type}"))
        dbound = getattr(self, f"dbound_{camera_type}")
        gt_depths = (gt_depths -
                     (dbound[0] - dbound[2])) / dbound[2]
        depth_channels = getattr(self, f"depth_channels_{camera_type}")
        gt_depths = torch.where(
            (gt_depths < depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=depth_channels + 1).view(
                                  -1, depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def forward(self, batch_data: dict, targets, frame_ids, frame_exists, ori_data, mode):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
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
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if mode == 'tensor':
            return self.extract_feat(batch_data)
        
        elif mode == 'loss':
            fish_bev_feature, fish_depth_pred, front_bev_feature, front_depth_pred, pv_bev_feature, pv_depth_pred = self.extract_feat(batch_data)
            pv_bev_fusion = self.PV_BEV_Fusion(pv_bev_feature, front_bev_feature, fish_bev_feature)
            preds = self.head(pv_bev_fusion)
            losses = dict()
            detection_loss = self.loss(targets, preds)
            # depth_features = {(self.frame_start_end_ids[0], 'fish'): fish_depth_pred, (self.frame_start_end_ids[0], 'pv'): pv_depth_pred, (self.frame_start_end_ids[0], 'front'): front_depth_pred}
            # for f_i in self.frame_start_end_ids[1:]:
            #     _, _fish_depth_pred, _, _front_depth_pred, _, _pv_depth_pred = self.extract_feat_depth(batch_data[f_i])
            #     depth_features[(f_i, 'fish')] = _fish_depth_pred
            #     depth_features[(f_i, 'pv')] = _pv_depth_pred
            #     depth_features[(f_i, 'front')] = _front_depth_pred
            
            # mono_losses, mono_total_losses = self.mono_depth_net(batch_data, depth_features)
            # losses.update(mono_losses)

            supervised_depth_loss = 0
            # the varible batch_data[self.frame_start_end_ids[0]] include fish data, front data and pv data
            for camera_type, depth_preds in zip(batch_data, [fish_depth_pred, front_depth_pred, pv_depth_pred]):
                depth_label = batch_data[camera_type]['depth']
                depth_loss = self.get_depth_loss(depth_label, depth_preds, camera_type.split('_')[0])
                supervised_depth_loss += depth_loss
            
            losses['loss'] = detection_loss # + mono_total_losses + supervised_depth_loss
            losses['detection_loss'] = detection_loss
            # losses['mono_total_loss'] = mono_total_losses
            losses['supervised_depth_loss'] = supervised_depth_loss
            
            return losses
        
        elif mode == 'predict':
            fish_bev_feature, fish_depth_pred, front_bev_feature, front_depth_pred, pv_bev_feature, pv_depth_pred = self.extract_feat(batch_data[self.frame_start_end_ids[0]])
            pv_bev_fusion = self.PV_BEV_Fusion(pv_bev_feature, front_bev_feature, fish_bev_feature)
            preds = self.head(pv_bev_fusion)
            all_pred, all_classes, all_scores = self.get_bboxes(preds, batch_data[self.frame_start_end_ids[0]], targets[self.frame_start_end_ids[0]])
            if False:
                self.show_results(all_pred, all_classes, batch_data[self.frame_start_end_ids[0]])
            
            return (all_pred, all_classes, all_scores)

    def extract_feat(self, batch_data):
        fish_bev_feature, fish_depth_pred = self.backbone_fish(deepcopy(batch_data['fish_data']['imgs']), batch_data['fish_data']['metainfo'], is_return_depth=self.is_train_depth)
        (front_bev_feature, front_depth_pred), (pv_bev_feature, pv_depth_pred) = self.backbone_pv(deepcopy(batch_data['front_data']['imgs']), batch_data['front_data']['metainfo'],\
                                                         deepcopy(batch_data['pv_data']['imgs']), batch_data['pv_data']['metainfo'],
                                                         is_return_depth=self.is_train_depth)

        return fish_bev_feature, fish_depth_pred, front_bev_feature, front_depth_pred, pv_bev_feature, pv_depth_pred

    # @torch.no_grad()
    def extract_feat_depth(self, batch_data):
        return self.extract_feat(batch_data)    
     
    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, debug=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, debug, img, rescale)
    
    def show_results(self, all_pred, all_classes, data, **kwargs):
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
            for i,sub_boxes in enumerate(all_pred[b]):
                sub_cla = all_classes[b]
                # if i != len(TASKS)-1:
                for j in range(len(sub_boxes)):
                    class_cat = sub_cla[j]
                    pred_box = sub_boxes[j]
                    tmp_center = [-(pred_box[1] - 36), -(pred_box[0] - 12), pred_box[2]]
                    tmp_box = np.array(tmp_center + pred_box[3:6] + [0] + [0, 0])
                    tmp_r = Rotation.from_euler('xyz', angles=[0, 0, pred_box[10][0]], degrees=False).as_matrix()
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
                imgs = data[key_camera]['imgs'][b]
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
                            data[key_camera]['metainfo'][0][b][idx][:3, 3].cpu().numpy(),
                            Quaternion(matrix=data[key_camera]['metainfo'][0][b][idx][:3, :3].cpu().numpy(), atol=1e-06),
                            intrinsics_matrix(data[key_camera]['metainfo'][1][b][idx][:4].cpu().numpy()))
                        lines = get_3d_lines(cam_corners)
                        for line in lines:
                            plt.plot(line[0],
                                    line[1],
                                    c=cm.get_cmap('tab10')(4)
                                    )
                    for corners in gt_cylinders:
                        cam_corners = get_cam_corners(
                            corners,
                            data[key_camera]['metainfo'][0][b][idx][:3, 3].cpu().numpy(),
                            Quaternion(matrix=data[key_camera]['metainfo'][0][b][idx][:3, :3].cpu().numpy(), atol=1e-06),
                            intrinsics_matrix(data[key_camera]['metainfo'][1][b][idx][:4].cpu().numpy()))
                        
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
            plt.show()