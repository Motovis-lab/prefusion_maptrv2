import torch
import torch.nn.functional as F
from copy import deepcopy
from mmengine.runner import autocast
from prefusion.registry import MODELS
from mmengine.model import BaseModel
import pdb

__all__ = ['MonoDepth']

@MODELS.register_module()
class MonoDepth(BaseModel):
    """Implementation of MatrixVT for Object Detection.

        Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    def __init__(self, 
                 data_preprocessor=None,
                 fish_img_backbone_neck_conf=None,
                 pv_img_backbone_neck_conf=None,
                 front_img_backbone_neck_conf=None,
                 depth_net_fish_conf=None,
                 depth_net_pv_conf=None,
                 depth_net_front_conf=None,
                 depth_decoder_conf=None,
                 mono_depth=None,
                 downsample_factor=4,
                 split_num=3,
                 each_camera_nums=dict(fish=4, pv=5, front=1),
                 depth_weight=1.,
                 eval_only=False,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.downsample_factor = downsample_factor
        self.img_backbone_neck_fish = MODELS.build(fish_img_backbone_neck_conf)
        self.img_backbone_neck_pv = MODELS.build(pv_img_backbone_neck_conf)
        self.img_backbone_neck_front = MODELS.build(front_img_backbone_neck_conf)

        self.depth_net_fish = MODELS.build(depth_net_fish_conf)
        self.depth_net_pv = MODELS.build(depth_net_pv_conf)
        self.depth_net_front = MODELS.build(depth_net_front_conf)
        self.frame_ids = [1, 0, 2]
        self.each_camera_nums = each_camera_nums
        if mono_depth is not None:
            self.mono_depth_net = MODELS.build(mono_depth)
            self.mono_depth_net.each_camera_nums = each_camera_nums
            self.mono_depth_net.frame_ids = self.frame_ids
        self.split_num = split_num
        
        self.dummy_loss = torch.tensor([0.], dtype=torch.float32, requires_grad=True).cuda()
        self.fish_cached = dict(
            
        )
        self.pv_cached = dict(
            
        )
        self.front_cached = dict(
            
        )
    def get_depth_loss(self, depth_labels, depth_preds, camera_type):
        depth_labels = self.get_downsampled_gt_depth(depth_labels, camera_type)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, getattr(getattr(self.backbone, f"depth_net_{camera_type}"), "depth_channels"))
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        assert depth_preds.device == depth_labels.device == fg_mask.device
        with autocast(device_type=depth_preds.device, enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask].float(),
                depth_labels[fg_mask].float(),
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
        dbound = getattr(getattr(self.backbone, f"depth_net_{camera_type}"), "d_bound")
        gt_depths = (gt_depths -
                     (dbound[0] - dbound[2])) / dbound[2]
        depth_channels = getattr(getattr(self.backbone, f"depth_net_{camera_type}"), "depth_channels")
        gt_depths = torch.where(
            (gt_depths < depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=depth_channels + 1).view(
                                  -1, depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def forward(self, batch_data: dict, frame_timestamp, frame_exists, delta_pose, ori_data, mode):
        """Forward function for Mono Depth
        
        """
        if mode == 'tensor':
            return self.extract_feat(batch_data)
        
        elif mode == 'loss':
            if self.split_num % 3 < 2 and None not in frame_exists['next_exists']:
                self.prepare_data(batch_data, delta_pose)
                self.split_num += 1
                return dict(loss = self.dummy_loss)
            if None in frame_exists['next_exists']:
                self.fish_cached = dict()
                self.pv_cached = dict()
                self.front_cached = dict()
                return dict(loss = self.dummy_loss)
            if len(self.fish_cached) // 3 == 2:
                fish_mono_depth_features = self.fish_mono_feat(self.fish_cached)
                pv_mono_depth_features = self.pv_mono_feat(self.pv_cached)
                front_mono_depth_features = self.front_mono_feat(self.front_cached)
                
                losses = dict()
                mono_losses = self.mono_depth_net(fish_mono_depth_features, pv_mono_depth_features, front_mono_depth_features,
                                                                     self.fish_cached, self.pv_cached, self.front_cached, self.split_num + 1)
                mono_total_losses = mono_losses['mono_loss_fish'] + mono_losses['mono_loss_pv'] + mono_losses['mono_loss_front']
                supervised_depth_loss = 0
                
                # for camera_type in batch_data:
                #     depth_label = batch_data[camera_type]['depth']
                #     depth_preds = features['depth_feats'][f"{camera_type.split('_')[0]}_feats"]
                #     depth_loss = self.get_depth_loss(depth_label, depth_preds, camera_type.split('_')[0])
                #     supervised_depth_loss += depth_loss * self.depth_weight
                # losses['supervised_depth_loss'] = supervised_depth_loss
                
                losses['loss'] = supervised_depth_loss + mono_total_losses 
                losses['mono_total_loss'] = mono_total_losses
                
                losses.update(mono_losses)  # only for record
                self.split_num = 0
                return losses
        
        elif mode == 'predict':
            pass

    def fish_mono_feat(self, fish_data, sweep_infos):
        fish_imgs = torch.cat([fish_data['color', i] for i in self.frame_ids], dim=0)
        img_feats_fish = self.get_cam_feats_fish(fish_imgs)
        intrinsic=sweep_infos["fish_data"]['intrinsic'].view(-1, 8)
        extrinsic=sweep_infos["fish_data"]['extrinsic'][:, :3, :].view(-1, 12)
        _,_, mono_depth_feats_fish = self.depth_net_fish(img_feats_fish, intrinsic, extrinsic, return_mono_depth=True)
        
        return mono_depth_feats_fish

    def pv_mono_feat(self, pv_data, sweep_infos):
        pv_imgs = torch.cat([pv_data['color', i] for i in self.frame_ids], dim=0)
        img_feats_pv = self.get_cam_feats_fish(pv_imgs)
        intrinsic = torch.zeros(sweep_infos["pv_data"]['extrinsic'].shape[0], 8).to(sweep_infos["pv_data"]['extrinsic'].device)
        intrinsic[:, :4] = sweep_infos["pv_data"]['intrinsic']
        extrinsic=sweep_infos["pv_data"]['extrinsic'][:, :3, :].view(-1, 12)
        _,_, mono_depth_feats_pv = self.depth_net_pv(img_feats_pv, intrinsic, extrinsic, return_mono_depth=True)
        
        return mono_depth_feats_pv

    def front_mono_feat(self, front_data, sweep_infos):
        front_imgs = torch.cat([front_data['color', i] for i in self.frame_ids], dim=0)
        img_feats_front = self.get_cam_feats_fish(front_imgs)
        intrinsic = torch.zeros(sweep_infos["pv_data"]['extrinsic'].shape[0], 8).to(sweep_infos["pv_data"]['extrinsic'].device)
        intrinsic[:, :4] = sweep_infos["pv_data"]['intrinsic']
        extrinsic=sweep_infos["pv_data"]['extrinsic'][:, :3, :].view(-1, 12)
        _,_, mono_depth_feats_front = self.depth_net_pv(img_feats_front, intrinsic, extrinsic, return_mono_depth=True)
    
        return mono_depth_feats_front

    def prepare_data(self, batch_data: dict, delta_pose):
        self.fish_cached[('color', self.split_num % 3)] = batch_data['fish_data']['imgs']
        self.fish_cached[('gt_depth', self.split_num % 3)] = batch_data['fish_data']['depth']

        self.pv_cached[('color', self.split_num % 3)] = batch_data['pv_data']['imgs']
        self.pv_cached[('gt_depth', self.split_num % 3)] = batch_data['pv_data']['depth']
        pv_intrinsic = batch_data['pv_data']['intrinsic']
        K = torch.eye(4, 4).repeat(pv_intrinsic.shape[0], 1, 1).to(pv_intrinsic.device)
        K[:, 0, 0] = pv_intrinsic[:, 2] # fx
        K[:, 1, 1] = pv_intrinsic[:, 3] # fy
        K[:, 0, 2] = pv_intrinsic[:, 0] # cx
        K[:, 1, 2] = pv_intrinsic[:, 1] # cy
        self.pv_cached['K', self.split_num % 3] = K
        self.pv_cached['inv_K', self.split_num % 3] = K.inverse()

        self.front_cached[('color', self.split_num % 3)] = batch_data['front_data']['imgs']
        self.front_cached[('gt_depth', self.split_num % 3)] = batch_data['front_data']['depth']
        front_intrinsic = batch_data['front_data']['intrinsic']
        K = torch.eye(4, 4).repeat(front_intrinsic.shape[0], 1, 1).to(front_intrinsic.device)
        K[:, 0, 0] = front_intrinsic[:, 2] # fx
        K[:, 1, 1] = front_intrinsic[:, 3] # fy
        K[:, 0, 2] = front_intrinsic[:, 0] # cx
        K[:, 1, 2] = front_intrinsic[:, 1] # cy
        self.front_cached['K', self.split_num % 3] = K
        self.front_cached['inv_K', self.split_num % 3] = K.inverse()
     
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
   
    def get_cam_feats_fish(self, sweep_imgs):
        img_feats = self.img_backbone_neck_fish(sweep_imgs)[0]
        
        return img_feats

    def get_cam_feats_pv(self, sweep_imgs):
        img_feats = self.img_backbone_neck_pv(sweep_imgs)[0]
        
        return img_feats
    
    def get_cam_feats_front(self, sweep_imgs):
        img_feats = self.img_backbone_neck_front(sweep_imgs)[0]
        
        return img_feats
