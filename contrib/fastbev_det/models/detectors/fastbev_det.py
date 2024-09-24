import torch
import torch.nn.functional as F
from copy import deepcopy
from mmengine.runner import autocast
from prefusion.registry import MODELS
from mmengine.model import BaseModel
import pdb


@MODELS.register_module()
class FastBEV_Det(BaseModel):
    """Implementation of MatrixVT for Object Detection.

        Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    def __init__(self, 
                 backbone_conf,
                 head_conf, 
                 is_train_depth=False, 
                 data_preprocessor=None, 
                 mono_depth=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone_conf)
        self.head = MODELS.build(head_conf)
        self.head_conf = head_conf
        self.is_train_depth = is_train_depth
        self.downsample_factor = backbone_conf.get('downsample_factor', 16)
        self.each_camera_nums = self.backbone.each_camera_nums if hasattr(self.backbone, 'each_camera_nums') else None
        self.head.each_camera_nums = self.each_camera_nums
        if mono_depth is not None:
            self.mono_depth_net = MODELS.build(mono_depth)
            self.mono_depth_net.each_camera_nums = self.each_camera_nums
     
    def get_depth_loss(self, depth_labels, depth_preds, camera_type):
        depth_labels = self.get_downsampled_gt_depth(depth_labels, camera_type)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, getattr(self.backbone.depth_net, f"depth_channels_{camera_type}"))
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        assert depth_preds.device == depth_labels.device == fg_mask.device
        with autocast(device_type=depth_preds.device, enabled=False):
            depth_loss = (F.binary_cross_entropy_with_logits(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return depth_loss

    def get_downsampled_gt_depth(self, gt_depths, camera_type):
        """
        Input:
            gt_depths: [B*N, H, W]
            camera_type: [fish, pv, front]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        BN, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            BN,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where((gt_depths <= 0.0) | (gt_depths==-1),
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(BN, H // self.downsample_factor,
                                   W // self.downsample_factor)
        dbound = getattr(self.backbone.depth_net, f"d_bound_{camera_type}")
        gt_depths = (gt_depths -
                     (dbound[0] - dbound[2])) / dbound[2]
        depth_channels = getattr(self.backbone.depth_net, f"depth_channels_{camera_type}")
        gt_depths = torch.where(
            (gt_depths < depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=depth_channels + 1).view(
                                  -1, depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def forward(self, batch_data: dict, frame_ids, frame_exists, ori_data, mode):
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
            features = self.extract_feat(batch_data)
            preds = self.head(features['bev_img_feats'])
            losses = dict()
            if self.head_conf['type'] != "MVBEVHead":
                targets = self.head.get_targets(ori_data)
            detection_loss, record_loss = self.loss(targets, preds)

            # mono_depth_input = self.extract_mono_depth_feature(batch_data)
            # mono_losses, mono_total_losses = self.mono_depth_net(mono_depth_input)
            
            supervised_depth_loss = 0
            
            for camera_type in batch_data:
                depth_label = batch_data[camera_type]['depth']
                depth_preds = features['depth_feats'][f"{camera_type.split('_')[0]}_feats"]
                depth_loss = self.get_depth_loss(depth_label, depth_preds, camera_type.split('_')[0])
                supervised_depth_loss += depth_loss
            
            losses['loss'] = detection_loss + supervised_depth_loss # + mono_total_losses 
            losses['detection_loss'] = detection_loss
            # losses['mono_total_loss'] = mono_total_losses
            losses['supervised_depth_loss'] = supervised_depth_loss
            
            losses.update(record_loss)
            # losses.update(mono_losses)  # only for record

            return losses
        
        elif mode == 'predict':
            features = self.extract_feat(batch_data)
            preds = self.head(features['bev_img_feats'])
            if self.head_conf['type'] != "MVBEVHead":
                results = self.get_bboxes(preds)
                for i in range(len(results)):
                    results[i][0][:, :2] = -results[i][0][:, :2]
                    results[i][0] = results[i][0].detach().cpu().numpy()
                    results[i][1] = results[i][1].detach().cpu().numpy()
                    results[i][2] = results[i][2].detach().cpu().numpy()
                    
                if False:
                    self.head.show_results(results, batch_data, frame_ids)
            else:
                all_pred, all_classes, all_scores = self.head.get_bboxes(preds, batch_data, targets)
                self.head.show_results(all_pred, all_classes, batch_data, frame_ids)

            return results

    def extract_feat(self, batch_data):
        features = self.backbone(deepcopy(batch_data), is_return_depth=self.is_train_depth, is_return_bev=True)

        return features

    def extract_mono_depth_feature(self, batch_data):
        B = batch_data['pv_data']['imgs'].shape[0]//self.each_camera_nums['pv']
        mono_depth_input = dict()
        mono_depth_input['pv_data'] = dict(imgs=batch_data['pv_data']['mono_imgs'].clone(), extrinsic=batch_data['pv_data']['mono_extrinsic'].clone(), intrinsic=batch_data['pv_data']['mono_intrinsic'].clone())
        mono_depth_input['front_data'] = dict(imgs=batch_data['front_data']['mono_imgs'].clone(), extrinsic=batch_data['front_data']['mono_extrinsic'].clone(), intrinsic=batch_data['front_data']['mono_intrinsic'].clone())
        mono_depth_feats = self.backbone(mono_depth_input, is_return_depth=self.is_train_depth, is_return_bev=False)['mono_depth_feats'] 
        pv_imgs_split = torch.split(batch_data['pv_data']['mono_imgs'], B*self.each_camera_nums['pv'], dim=0)
        front_imgs_split = torch.split(batch_data['front_data']['mono_imgs'], B*self.each_camera_nums['front'], dim=0)
        pv_intrinsic_split = torch.split(batch_data['pv_data']['mono_intrinsic'], B*self.each_camera_nums['pv'], dim=0)
        front_intrinsic_split = torch.split(batch_data['front_data']['mono_intrinsic'], B*self.each_camera_nums['front'], dim=0)
        pv_feat_split = torch.split(mono_depth_feats['pv_mono_depth_feats'], B*self.each_camera_nums['pv'], dim=0)
        front_feat_split = torch.split(mono_depth_feats['front_mono_depth_feats'], B*self.each_camera_nums['front'], dim=0)
        for i in range(0, 3):
            K = torch.eye(4, 4).repeat(pv_intrinsic_split[i].shape[0], 1, 1).to(pv_intrinsic_split[i].device)
            K[:, 0, 0] = pv_intrinsic_split[i][:, 2] # fx
            K[:, 1, 1] = pv_intrinsic_split[i][:, 3] # fy
            K[:, 0, 2] = pv_intrinsic_split[i][:, 0] # cx
            K[:, 1, 2] = pv_intrinsic_split[i][:, 1] # cy
            mono_depth_input[('pv_img', i-1)] = pv_imgs_split[i]   
            mono_depth_input[('pv_K', i-1)] = K
            mono_depth_input[('pv_feat', i-1)] = pv_feat_split[i]
            K = torch.eye(4, 4).repeat(front_intrinsic_split[i].shape[0], 1, 1).to(front_intrinsic_split[i].device)
            K[:, 0, 0] = front_intrinsic_split[i][:, 2] # fx
            K[:, 1, 1] = front_intrinsic_split[i][:, 3] # fy
            K[:, 0, 2] = front_intrinsic_split[i][:, 0] # cx
            K[:, 1, 2] = front_intrinsic_split[i][:, 1] # cy
            mono_depth_input[('front_img', i-1)] = front_imgs_split[i]   
            mono_depth_input[('front_K', i-1)] = K
            mono_depth_input[('front_feat', i-1)] = front_feat_split[i]
        # plt.imshow(pv_split[0][0].cpu().detach().numpy().transpose(1,2,0) - pv_split[0][0].cpu().detach().numpy().transpose(1,2,0).min())
        # mono_depth_input['pv_data']['imgs'] = torch.split()
        return mono_depth_input
     
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
        return self.head.get_bboxes(preds_dicts)
    
