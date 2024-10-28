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
                 mono_depth=None,
                 downsample_factor=4,
                 mono_frame_nums=3,
                 each_camera_nums=dict(fish=4, pv=5, front=1),
                 supervised_depth_weight=1.,
                 mono_depth_weight=1.,
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
        self.frame_nums = 3
        self.each_camera_nums = each_camera_nums
        if mono_depth is not None:
            self.mono_depth_head = MODELS.build(mono_depth)
            self.mono_depth_head.each_camera_nums = each_camera_nums
            self.mono_depth_head.frame_ids = self.frame_ids
            self.mono_depth_head.frame_nums = self.frame_nums
        self.split_num = 0
        self.supervised_depth_weight = supervised_depth_weight
        self.mono_depth_weight = mono_depth_weight    
        self.fish_cached_info = ['color', 'gt_depth', 'delta_pose', 'ego_mask']
        self.mono_frame_nums = mono_frame_nums
        self.fish_cached = dict()
        self.pv_cached = dict()
        self.front_cached = dict()
    def get_depth_loss(self, depth_labels, depth_preds, camera_type):
        depth_labels = self.get_downsampled_gt_depth(depth_labels, camera_type)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, getattr(getattr(self, f"depth_net_{camera_type}"), "depth_channels"))
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
        dbound = getattr(getattr(self, f"depth_net_{camera_type}"), "d_bound")
        gt_depths = (gt_depths -
                     (dbound[0] - dbound[2])) / dbound[2]
        depth_channels = getattr(getattr(self, f"depth_net_{camera_type}"), "depth_channels")
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
        # TODO add supervised_depth_loss and dynamic region mask
        losses = dict(loss = torch.tensor([0.], dtype=torch.float32, requires_grad=True).cuda())
        if mode == 'tensor':
            return self.extract_feat(batch_data)
        
        elif mode == 'loss':
            if None not in frame_exists['next_exists'] and None not in frame_exists['prev_exists']:  # 中间帧，前后都有
                self.prepare_data(batch_data, delta_pose)
                self.split_num += 1
            elif None in frame_exists['next_exists']:  # 后没有
                if None not in frame_exists['prev_exists']:  # 如果前有
                    # group size的最后一帧
                    self.prepare_data(batch_data, delta_pose)
                    self.split_num += 1
                else:  # 前没有
                    self.fish_cached = dict()
                    self.pv_cached = dict()
                    self.front_cached = dict()
                    self.split_num = 0
            elif None in frame_exists['prev_exists']:  # 前没有
                if None not in frame_exists['next_exists']: # 后有
                    # group size的第一帧
                    self.prepare_data(batch_data, delta_pose)
                    self.split_num += 1
                else:  # 后没有
                    self.fish_cached = dict()
                    self.pv_cached = dict()
                    self.front_cached = dict()
                    self.split_num = 0
            
            if len(self.fish_cached) // len(self.fish_cached_info) == self.mono_frame_nums:
                fish_supervised_depth_feats, fish_mono_depth_features = self.fish_mono_feat(self.fish_cached, batch_data)
                pv_supervised_depth_feats, pv_mono_depth_features = self.pv_mono_feat(self.pv_cached, batch_data)
                front_supervised_depth_feats, front_mono_depth_features = self.front_mono_feat(self.front_cached, batch_data)
                
                mono_losses = self.mono_depth_head(fish_mono_depth_features, pv_mono_depth_features, front_mono_depth_features,
                                                                    self.fish_cached, self.pv_cached, self.front_cached)
                mono_total_losses = (mono_losses['mono_loss_fish'] + mono_losses['mono_loss_pv'] + mono_losses['mono_loss_front']) * self.mono_depth_weight
                supervised_depth_loss = dict()
                supervised_total_loss = 0
                for camera_type in ["fish", "pv", "front"]:
                    depth_label = getattr(self, f"{camera_type}_cached")[('gt_depth', 1)]
                    depth_preds = eval(f"{camera_type}_supervised_depth_feats").split(depth_label.shape[0])[1]
                    depth_loss = self.get_depth_loss(depth_label, depth_preds, camera_type.split('_')[0])
                    supervised_depth_loss[f"supervised_depth_loss_{camera_type}"] = depth_loss
                    supervised_total_loss += depth_loss * self.supervised_depth_weight
                
                losses['loss'] = supervised_total_loss + mono_total_losses 
                losses['supervised_depth_total_loss'] = supervised_total_loss
                losses['mono_total_loss'] = mono_total_losses

                losses.update(mono_losses)  # only for record
                losses.update(supervised_depth_loss)
                # self.update_cache_data()
                # self.split_num -= 1

            return losses
        
        elif mode == 'predict':
            pass

    def fish_mono_feat(self, fish_data, sweep_infos):
        fish_imgs = torch.cat([fish_data['color', i] for i in range(self.split_num)], dim=0)
        img_feats_fish = self.get_cam_feats_fish(fish_imgs)
        intrinsic=sweep_infos["fish_data"]['intrinsic'].view(-1, 8).repeat(self.frame_nums, 1)
        extrinsic=sweep_infos["fish_data"]['extrinsic'][:, :3, :].view(-1, 12).repeat(self.frame_nums, 1)
        _, supervised_depth_feats, mono_depth_feats_fish = self.depth_net_fish(img_feats_fish, intrinsic, extrinsic, return_mono_depth=True)
        
        return supervised_depth_feats, mono_depth_feats_fish

    def pv_mono_feat(self, pv_data, sweep_infos):
        pv_imgs = torch.cat([pv_data['color', i] for i in range(self.split_num)], dim=0)
        img_feats_pv = self.get_cam_feats_fish(pv_imgs)
        intrinsic = torch.zeros(sweep_infos["pv_data"]['extrinsic'].shape[0], 8).to(sweep_infos["pv_data"]['extrinsic'].device)
        intrinsic[:, :4] = sweep_infos["pv_data"]['intrinsic']
        extrinsic=sweep_infos["pv_data"]['extrinsic'][:, :3, :].view(-1, 12).repeat(self.frame_nums, 1)
        _, supervised_depth_feats, mono_depth_feats_pv = self.depth_net_pv(img_feats_pv, intrinsic.repeat(self.frame_nums, 1), extrinsic, return_mono_depth=True)
        
        return supervised_depth_feats, mono_depth_feats_pv

    def front_mono_feat(self, front_data, sweep_infos):
        front_imgs = torch.cat([front_data['color', i] for i in range(self.split_num)], dim=0)
        img_feats_front = self.get_cam_feats_fish(front_imgs)
        intrinsic = torch.zeros(sweep_infos["front_data"]['extrinsic'].shape[0], 8).to(sweep_infos["front_data"]['extrinsic'].device)
        intrinsic[:, :4] = sweep_infos["front_data"]['intrinsic']
        extrinsic=sweep_infos["front_data"]['extrinsic'][:, :3, :].view(-1, 12).repeat(self.frame_nums, 1)
        _, supervised_depth_feats, mono_depth_feats_front = self.depth_net_pv(img_feats_front, intrinsic.repeat(self.frame_nums, 1), extrinsic, return_mono_depth=True)
    
        return supervised_depth_feats, mono_depth_feats_front

    def prepare_data(self, batch_data: dict, delta_pose):
        self.fish_cached[('color', self.split_num % 3)] = batch_data['fish_data']['imgs']
        self.fish_cached[('gt_depth', self.split_num % 3)] = batch_data['fish_data']['depth']
        self.fish_cached[('delta_pose', self.split_num % 3)] = delta_pose
        self.fish_cached[('ego_mask', self.split_num % 3)] = batch_data['fish_data']['ego_mask']

        self.pv_cached[('color', self.split_num % 3)] = batch_data['pv_data']['imgs']
        self.pv_cached[('gt_depth', self.split_num % 3)] = batch_data['pv_data']['depth']
        self.pv_cached[('delta_pose', self.split_num % 3)] = delta_pose
        self.pv_cached[('ego_mask', self.split_num % 3)] = batch_data['pv_data']['ego_mask']
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
        self.front_cached[('delta_pose', self.split_num % 3)] = delta_pose
        self.front_cached[('ego_mask', self.split_num % 3)] = batch_data['front_data']['ego_mask']
        front_intrinsic = batch_data['front_data']['intrinsic']
        K = torch.eye(4, 4).repeat(front_intrinsic.shape[0], 1, 1).to(front_intrinsic.device)
        K[:, 0, 0] = front_intrinsic[:, 2] # fx
        K[:, 1, 1] = front_intrinsic[:, 3] # fy
        K[:, 0, 2] = front_intrinsic[:, 0] # cx
        K[:, 1, 2] = front_intrinsic[:, 1] # cy
        self.front_cached['K', self.split_num % 3] = K
        self.front_cached['inv_K', self.split_num % 3] = K.inverse()
     
    def update_cache_data(self):
        for id in range(self.split_num):
            if id == 0:
                self.fish_cached.pop(('color', id))
                self.fish_cached.pop(('gt_depth', id))
                self.fish_cached.pop(('delta_pose', id))
                self.fish_cached.pop(('ego_mask', id))

                self.pv_cached.pop(('color', id))
                self.pv_cached.pop(('gt_depth', id))
                self.pv_cached.pop(('delta_pose', id))
                self.pv_cached.pop(('ego_mask', id))
                self.pv_cached.pop(('K', id))
                self.pv_cached.pop(('inv_K', id))

                self.front_cached.pop(('color', id))
                self.front_cached.pop(('gt_depth', id))
                self.front_cached.pop(('delta_pose', id))
                self.front_cached.pop(('ego_mask', id))
                self.front_cached.pop(('K', id))
                self.front_cached.pop(('inv_K', id))
            else:
                self.fish_cached[('color', id - 1)] = self.fish_cached[('color', id)]
                self.fish_cached[('gt_depth', id - 1)] = self.fish_cached[('gt_depth', id)]
                self.fish_cached[('delta_pose', id - 1)] = self.fish_cached[('delta_pose', id)]
                self.fish_cached[('ego_mask', id - 1)] = self.fish_cached[('ego_mask', id)]
                
                self.pv_cached[('color', id - 1)] = self.pv_cached[('color', id)]
                self.pv_cached[('gt_depth', id - 1)] = self.pv_cached[('gt_depth', id)]
                self.pv_cached[('delta_pose', id - 1)] = self.pv_cached[('delta_pose', id)]
                self.pv_cached[('ego_mask', id - 1)] = self.pv_cached[('ego_mask', id)]
                self.pv_cached['K', id - 1] = self.pv_cached['K', id]
                self.pv_cached['inv_K', id - 1] = self.pv_cached['inv_K', id]
                
                self.front_cached[('color', id - 1)] = self.front_cached[('color', id)]
                self.front_cached[('gt_depth', id - 1)] = self.front_cached[('gt_depth', id)]
                self.front_cached[('delta_pose', id - 1)] = self.front_cached[('delta_pose', id)]
                self.front_cached[('ego_mask', id - 1)] = self.front_cached[('ego_mask', id)]
                self.front_cached['K', id - 1] = self.front_cached['K', id]
                self.front_cached['inv_K', id - 1] = self.front_cached['inv_K', id]

    def get_cam_feats_fish(self, sweep_imgs):
        img_feats = self.img_backbone_neck_fish(sweep_imgs)[0]
        
        return img_feats

    def get_cam_feats_pv(self, sweep_imgs):
        img_feats = self.img_backbone_neck_pv(sweep_imgs)[0]
        
        return img_feats
    
    def get_cam_feats_front(self, sweep_imgs):
        img_feats = self.img_backbone_neck_front(sweep_imgs)[0]
        
        return img_feats
