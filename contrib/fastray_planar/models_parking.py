import torch
import torch.nn as nn

from mmengine.structures import BaseDataElement

from prefusion import BaseModel
from prefusion import SegIouLoss, DualFocalLoss
from prefusion.registry import MODELS
from .misc_draw import draw_results_planar_lidar,draw_results_parking,draw_results_parking_IPM,draw_results_3dboxes

from .modules import *
from .model_utils import *
import torch.nn.functional as F

def build_folder_set(text):
    return set(line.strip() for line in text.strip().splitlines())

mask_960_folders = build_folder_set("""
SLOT_BIRD-APS-20240809-19830
SLOT_BIRD-APS-20240812-6437
""")
mask_1600_folders = build_folder_set("""
SLOT_BIRD-APS-20240110-393
SLOT_BIRD-APS-20240110-605
""")
mask_640_folders = build_folder_set("""
SLOT_BIRD-APS-20230607-1880
SLOT_BIRD-APS-20230608-807
SLOT_BIRD-APS-20230613-543
SLOT_BIRD-APS-20230830-41-SLOT_BIRD-APS-20230830-41
SLOT_BIRD-APS-20230915-1743
SLOT_BIRD-APS-20231114-141
SLOT_BIRD-APS-20231218-1065
SLOT_BIRD-APS-20231219-696
SLOT_BIRD-APS-20240112-687
SLOT_BIRD-APS-20240202-2889
SLOT_BIRD-APS-20240220-1987
SLOT_BIRD-APS-20240328-1346
SLOT_BIRD-APS-20240329-1329
SLOT_BIRD-APS-20240403-35
SLOT_BIRD-APS-20240403-92
SLOT_BIRD-APS-20240407-1688
SLOT_BIRD-APS-20240408-4568
SLOT_BIRD-APS-20240513-178
SLOT_BIRD-APS-20240513-207
SLOT_BIRD-APS-20240513-89
SLOT_BIRD-APS-20240603-1329
SLOT_BIRD-APS-20240604-143
SLOT_BIRD-APS-20240604-416
SLOT_BIRD-APS-20240704-112
SLOT_BIRD-APS-20240704-137
SLOT_BIRD-APS-20240715-185
SLOT_BIRD-APS-20240725-1123
SLOT_BIRD-APS-20240805-307
SLOT_BIRD-APS-20241122_2993
SLOT_BIRD-APS-20241125_2286
SLOT_BIRD-APS-20241126_226
""")

def extract_project_folder_name(folder_name: str):
    """
    提取 SLOT_BIRD-APS-2024xxxx-xxxx 或 SLOT_BIRD-APS-2024xxxx_xxxx 的子串
    """
    import re
    match = re.search(r"SLOT_BIRD-APS-\d{8}[-_]\d+", folder_name)
    if match:
        return match.group(0)
    else:
        return None

def get_mask_resolution(folder_name):
    project_name = extract_project_folder_name(folder_name)
    if project_name in mask_640_folders:
        return (640, 640)
    elif project_name in mask_960_folders:
        return (960, 960)
    elif project_name in mask_1600_folders:
        return (1600, 1600)
    else:
        return (960, 1280)

def get_centered_mask(full_H=1280, full_W=960, valid_H=640, valid_W=640, margin=8, device='cpu'):
    mask = torch.zeros((1, 1, full_H, full_W), dtype=torch.float32, device=device)
    
    w_margin_left = (full_W - valid_W) // 2
    w_margin_right = (full_W - valid_W) - w_margin_left
    h_margin_top = (full_H - valid_H) // 2
    h_margin_bottom = (full_H - valid_H) - h_margin_top
    
    w_start = w_margin_left + margin
    w_end = full_W - w_margin_right - margin
    h_start = h_margin_top + margin
    h_end = full_H - h_margin_bottom - margin

    mask[:, :, h_start:h_end, w_start:w_end] = 1.0
    return mask




@MODELS.register_module()
class ParkingFastRayPlanarMultiFrameModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 pre_nframes=1,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_front = MODELS.build(backbones['pv_front'])
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        self.backbone_fisheyes = MODELS.build(backbones['fisheyes'])
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        # self.voxel_fusion = EltwiseAdd()
        self.voxel_fusion = MODELS.build(voxel_fusion)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # hidden voxel features for temporal fusion
        self.pre_nframes = pre_nframes
        self.cached_voxel_feats = {}
        self.cached_delta_poses = {}
        # init losses
        self.losses_dict = {}
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
        delta_poses = batched_input_dict['delta_poses']
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_front']:
                camera_feats_dict[cam_id] = self.backbone_pv_front(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
            if cam_id in self.camera_groups['fisheyes']:
                camera_feats_dict[cam_id] = self.backbone_fisheyes(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats_cur = self.spatial_transform(camera_feats_dict, camera_lookups)
        ## temporal transform
        cur_first_index_info = batched_input_dict['index_infos'][0]
        # init tmp pre_0 cache
        self.cached_voxel_feats[f'pre_0'] = voxel_feats_cur
        self.cached_delta_poses[f'pre_0'] = delta_poses
        for pre_i in range(self.pre_nframes):
            index_info_prev_str = 'cur_first_index_info' + ''.join(['.prev'] * (pre_i+1))
            if eval(index_info_prev_str) is None:
                for pre_j in range(pre_i, self.pre_nframes):
                    self.cached_voxel_feats[f'pre_{pre_j+1}'] = self.cached_voxel_feats[f'pre_{pre_j}'].clone().detach()
                    self.cached_delta_poses[f'pre_{pre_j+1}'] = self.cached_delta_poses[f'pre_{pre_j}'].clone().detach()
                break
        # cache delta_poses
        for pre_i in range(self.pre_nframes, 1, -1):
            self.cached_delta_poses[f'pre_{pre_i}'] = (self.cached_delta_poses[f'pre_{pre_i-1}'] @ delta_poses).clone().detach()
        self.cached_delta_poses['pre_1'] = delta_poses.clone().detach()
        # align all history frames to current frame
        aligned_voxel_feats_cat = [voxel_feats_cur]
        for pre_i in range(self.pre_nframes):
            aligned_voxel_feats_cat.append(self.temporal_transform(
                self.cached_voxel_feats[f'pre_{pre_i+1}'], self.cached_delta_poses[f'pre_{pre_i+1}']
            ))
        # if self.debug_mode:
        #     draw_aligned_voxel_feats(aligned_voxel_feats_cat)
        # cache voxel features
        for pre_i in range(self.pre_nframes, 0, -1):
            self.cached_voxel_feats[f'pre_{pre_i}'] = (self.cached_voxel_feats[f'pre_{pre_i-1}']).clone().detach()
        # pop tmp pre_0 cache
        self.cached_voxel_feats.pop('pre_0')
        self.cached_delta_poses.pop('pre_0')
        ## voxel fusion
        voxel_feats_fused = self.voxel_fusion(*aligned_voxel_feats_cat)
        ## voxel encoder
        if len(voxel_feats_fused.shape) == 5:
            N, C, Z, X, Y = voxel_feats_fused.shape
            voxel_feats_fused = voxel_feats_fused.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats_fused)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        
        pred_dict = dict(
            bbox_3d_heading=dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:14],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_plane_heading=dict(
                cen=out_bbox_3d[0][:, 14:15],
                seg=out_bbox_3d[0][:, 15:26],
                reg=out_bbox_3d[1][:, 20:40]),
            bbox_3d_no_heading=dict(
                cen=out_bbox_3d[0][:, 26:27],
                seg=out_bbox_3d[0][:, 27:34],
                reg=out_bbox_3d[1][:, 40:54]),
            bbox_3d_square=dict(
                cen=out_bbox_3d[0][:, 34:35],
                seg=out_bbox_3d[0][:, 35:40],
                reg=out_bbox_3d[1][:, 54:65]),
            bbox_3d_cylinder=dict(
                cen=out_bbox_3d[0][:, 40:41],
                seg=out_bbox_3d[0][:, 41:51],
                reg=out_bbox_3d[1][:, 65:73]),
            bbox_3d_oriented_cylinder=dict(
                cen=out_bbox_3d[0][:, 51:52],
                seg=out_bbox_3d[0][:, 52:54],
                reg=out_bbox_3d[1][:, 73:86]),
            polyline_3d=dict(
                seg=out_polyline_3d[0][:, 0:9],
                reg=out_polyline_3d[1][:, 0:7]),
            polygon_3d=dict(
                seg=out_polyline_3d[0][:, 9:15],
                reg=out_polyline_3d[1][:, 7:14]),
            parkingslot_3d=dict(
                cen=out_parkingslot_3d[0][:, :1],
                seg=out_parkingslot_3d[0][:, 1:],
                reg=out_parkingslot_3d[1])
        )
        
        if self.debug_mode:
            draw_outputs(pred_dict, batched_input_dict)
        
        if mode == 'tensor':
            return pred_dict
        if mode == 'loss':
            gt_dict = batched_input_dict['annotations']
            losses = self.compute_losses(pred_dict, gt_dict)
            return losses

        if mode == 'predict':
            gt_dict = batched_input_dict['annotations']
            losses = self.compute_losses(pred_dict, gt_dict)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, pred_dict, gt_dict):
        losses = dict(loss=0)
        for branch in self.losses_dict:
            losses_branch = self.losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            losses.update(losses_branch)

        return losses
    

@MODELS.register_module()
class ParkingFastRayPlanarSingleFrameModelAPA(BaseModel):
    
    def __init__(self,
                 backbone,
                 spatial_transform,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        self.head_occ_sdf_bev = MODELS.build(heads['occ_sdf_bev'])
        # init losses
        self.planar_losses_dict = {}
        if loss_cfg is not None:
            for branch in loss_cfg:
                self.planar_losses_dict[branch] = MODELS.build(loss_cfg[branch])
        self.occ_seg_iou_loss = SegIouLoss(method='linear')
        self.occ_seg_dfl_loss = DualFocalLoss()
        self.occ_sdf_l1_loss = nn.L1Loss(reduction='none')
        self.occ_height_l1_loss = nn.L1Loss(reduction='none')

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
            bev_feats = bev_feats.reshape(N, C*Z, X, Y)
        ## voxel encoder
        bev_feats = self.voxel_encoder(bev_feats)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        out_occ_sdf_bev = self.head_occ_sdf_bev(bev_feats)
        
        pred_dict = dict(
            bbox_3d_heading=dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:14],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_plane_heading=dict(
                cen=out_bbox_3d[0][:, 14:15],
                seg=out_bbox_3d[0][:, 15:26],
                reg=out_bbox_3d[1][:, 20:40]),
            bbox_3d_no_heading=dict(
                cen=out_bbox_3d[0][:, 26:27],
                seg=out_bbox_3d[0][:, 27:34],
                reg=out_bbox_3d[1][:, 40:54]),
            bbox_3d_square=dict(
                cen=out_bbox_3d[0][:, 34:35],
                seg=out_bbox_3d[0][:, 35:40],
                reg=out_bbox_3d[1][:, 54:65]),
            bbox_3d_cylinder=dict(
                cen=out_bbox_3d[0][:, 40:41],
                seg=out_bbox_3d[0][:, 41:51],
                reg=out_bbox_3d[1][:, 65:73]),
            bbox_3d_oriented_cylinder=dict(
                cen=out_bbox_3d[0][:, 51:52],
                seg=out_bbox_3d[0][:, 52:54],
                reg=out_bbox_3d[1][:, 73:86]),
            polyline_3d=dict(
                seg=out_polyline_3d[0][:, 0:9],
                reg=out_polyline_3d[1][:, 0:7]),
            polygon_3d=dict(
                seg=out_polyline_3d[0][:, 9:15],
                reg=out_polyline_3d[1][:, 7:14]),
            parkingslot_3d=dict(
                cen=out_parkingslot_3d[0][:, :1],
                seg=out_parkingslot_3d[0][:, 1:],
                reg=out_parkingslot_3d[1]),
        )
        pred_occ_sdf_bev=dict(
            seg=out_occ_sdf_bev[0],
            sdf=out_occ_sdf_bev[1][:, 0:1],
            height=out_occ_sdf_bev[1][0:, 1:2],
        )
        
        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']
            gt_occ_sdf_bev = gt_dict['occ_sdf_bev']
        
        if self.debug_mode:
            import matplotlib.pyplot as plt

            plt.imshow(gt_occ_sdf_bev['seg'][0][0].detach().cpu().numpy()); plt.show()
            freespace = pred_occ_sdf_bev['seg'][0][0].sigmoid().detach().cpu().numpy() > 0.5
            plt.imshow(freespace); plt.show()

            plt.imshow(gt_occ_sdf_bev['seg'][0][1].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['seg'][0][1].sigmoid().detach().cpu().numpy() > 0.5); plt.show()

            plt.imshow(gt_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy() > 0); plt.show()

            plt.imshow(gt_occ_sdf_bev['height'][0][0].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['height'][0][0].detach().cpu().numpy()); plt.show()

            # draw_outputs(pred_dict, batched_input_dict)
            # save_outputs(pred_dict, batched_input_dict)
            # TODO: save occ_sdf_bev

        if mode == 'tensor':
            pred_dict['occ_sdf_bev'] = pred_occ_sdf_bev
            return pred_dict
        if mode == 'loss':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            losses['loss'] += losses['occ_sdf_bev_loss']
            return losses
        if mode == 'predict':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            losses['loss'] += losses['occ_sdf_bev_loss']
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
        losses['occ_seg_iou_0_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_dfl_0_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_iou_1_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        losses['occ_seg_dfl_1_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        
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
class ParkingFastRayPlanarSingleFrameModelAPA_Parking_IPM(BaseModel):
    
    def __init__(self,
                 backbone,
                 spatial_transform,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)

        # voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        self.head_occ_sdf_bev = MODELS.build(heads['occ_sdf_bev'])
        # init losses
        self.planar_losses_dict = {}
        if loss_cfg is not None:
            for branch in loss_cfg:
                self.planar_losses_dict[branch] = MODELS.build(loss_cfg[branch])
        self.occ_seg_iou_loss = SegIouLoss(method='linear')
        self.occ_seg_dfl_loss = DualFocalLoss()
        self.occ_sdf_l1_loss = nn.L1Loss(reduction='none')
        self.occ_height_l1_loss = nn.L1Loss(reduction='none')

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
            bev_feats = bev_feats.reshape(N, C*Z, X, Y)
        ## voxel encoder
        bev_feats = self.voxel_encoder(bev_feats)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        out_occ_sdf_bev = self.head_occ_sdf_bev(bev_feats)
        

        # print('bbox_3d_heading=', out_bbox_3d[0][:, 1:14].max(), out_bbox_3d[0][:, 1:14].min())
        # print('bbox_3d_plane_heading=', out_bbox_3d[0][:, 15:26].max(), out_bbox_3d[0][:, 15:26].min())

        pred_dict = dict(
            bbox_3d_heading=dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:14],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_plane_heading=dict(
                cen=out_bbox_3d[0][:, 14:15],
                seg=out_bbox_3d[0][:, 15:26],
                reg=out_bbox_3d[1][:, 20:40]),
            bbox_3d_no_heading=dict(
                cen=out_bbox_3d[0][:, 26:27],
                seg=out_bbox_3d[0][:, 27:34],
                reg=out_bbox_3d[1][:, 40:54]),
            bbox_3d_square=dict(
                cen=out_bbox_3d[0][:, 34:35],
                seg=out_bbox_3d[0][:, 35:40],
                reg=out_bbox_3d[1][:, 54:65]),
            bbox_3d_cylinder=dict(
                cen=out_bbox_3d[0][:, 40:41],
                seg=out_bbox_3d[0][:, 41:51],
                reg=out_bbox_3d[1][:, 65:73]),
            bbox_3d_oriented_cylinder=dict(
                cen=out_bbox_3d[0][:, 51:52],
                seg=out_bbox_3d[0][:, 52:54],
                reg=out_bbox_3d[1][:, 73:86]),
            polyline_3d=dict(
                seg=out_polyline_3d[0][:, 0:9],
                reg=out_polyline_3d[1][:, 0:7]),
            polygon_3d=dict(
                seg=out_polyline_3d[0][:, 9:15],
                reg=out_polyline_3d[1][:, 7:14]),
            parkingslot_3d=dict(
                cen=out_parkingslot_3d[0][:, :1],
                seg=out_parkingslot_3d[0][:, 1:],
                reg=out_parkingslot_3d[1],
                pts=out_parkingslot_3d[2],
                pin=out_parkingslot_3d[3],
            ),
        )
        pred_occ_sdf_bev=dict(
            seg=out_occ_sdf_bev[0],
            sdf=out_occ_sdf_bev[1][:, 0:1],
            height=out_occ_sdf_bev[1][0:, 1:2],
        )


        #generate mask for IPM data
        # 初始化 mask_batch
        mask_batch= torch.ones_like(out_parkingslot_3d[3][:, :1])
        batch_size = bev_feats.shape[0]
        for batch_idx in range(batch_size):
            index_info_str = str(batched_input_dict['index_infos'][batch_idx])
            folder_name = index_info_str.split('/')[0]
            valid_W, valid_H = get_mask_resolution(folder_name)
            # 正确方式：以中心为区域裁剪创建 mask
            # 这里判断一下，full_H和full_W根据数据来传入
            raw_mask = torch.ones((1, 1, 1280, 960), dtype=torch.float32, device=bev_feats.device)
            if any(key in folder_name for key in (mask_1600_folders | mask_960_folders | mask_640_folders)):
                if valid_H == 1600 and valid_W == 1600:
                    # 全图有效区域
                    valid_H= 1000
                    valid_W = 800                    
                    raw_mask = get_centered_mask(
                        full_H=1280, full_W=960,
                        valid_H=valid_H, valid_W=valid_W, margin=16,
                        device=bev_feats.device
                    )
                    margin = 0  # 你想要的margin宽度，可以调整
                    # # 把四周margin部分置为0                    
                    # raw_mask[:, :, :margin, :] = 0         # 顶部margin行
                    # raw_mask[:, :, -margin:, :] = 0        # 底部margin行
                    # raw_mask[:, :, :, :margin] = 0         # 左侧margin列
                    # raw_mask[:, :, :, -margin:] = 0        # 右侧margin列
                elif valid_H == 960 and valid_W == 960:
                    # 中心裁剪区域
                    valid_H= 420
                    valid_W = 640
                    raw_mask = get_centered_mask(
                        full_H=1280, full_W=960,
                        valid_H=valid_H, valid_W=valid_W, margin=10,
                        device=bev_feats.device
                    )
                elif valid_H == 640 and valid_W == 640:
                    # 中心裁剪区域
                    valid_H= 0
                    valid_W = 0
                    raw_mask = get_centered_mask(
                        full_H=1280, full_W=960,
                        valid_H=valid_H, valid_W=valid_W, margin=10,
                        device=bev_feats.device
                    )            
            else:
                # fallback：不包含任何已知关键字，直接全 1
                raw_mask = torch.ones((1, 1, 1280, 960), dtype=torch.float32, device=bev_feats.device)
                margin = 0  # 你想要的margin宽度，可以调整
            seg_shape = out_parkingslot_3d[2].shape[-2:]  # seg 是 head 第三个输出
            mask = F.interpolate(raw_mask, size=seg_shape, mode='nearest')  # (1, 1, 320, 240) for example
            mask_batch[batch_idx] = mask
        pred_dict['mask'] = mask_batch


        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']
            gt_occ_sdf_bev = gt_dict['occ_sdf_bev']
        
        if self.debug_mode:

            # images = draw_results_parking_IPM(pred_dict, batched_input_dict, save_im=True)
            # print(pred_dict)
            images = draw_results_3dboxes(pred_dict, batched_input_dict, save_im=True)
            # images = draw_results_parking(pred_dict, batched_input_dict, save_im=True)

        if mode == 'tensor':
            pred_dict['occ_sdf_bev'] = pred_occ_sdf_bev
            return pred_dict
        if mode == 'loss':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict,batched_input_dict['index_infos']))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev,pred_dict['mask']))
            losses['loss'] += losses['occ_sdf_bev_loss']
            return losses
        if mode == 'predict':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict,batched_input_dict['index_infos']))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev,pred_dict['mask']))
            losses['loss'] += losses['occ_sdf_bev_loss']
            return (
                *[{branch: {t: v.cpu() for t, v in _pred.items()}} for branch, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_planar_losses(self, pred_dict, gt_dict,index_info_str):
        losses = dict(loss=0)
        for branch in self.planar_losses_dict:
            if branch == 'parkingslot_3d':
                losses_branch = self.planar_losses_dict[branch](pred_dict[branch], gt_dict[branch],index_info_str=index_info_str,mask=pred_dict['mask'])
            else:
                losses_branch = self.planar_losses_dict[branch](pred_dict[branch], gt_dict[branch],index_info_str=index_info_str,mask=pred_dict['mask'])
            losses['loss'] += losses_branch[branch + '_loss']
            losses.update(losses_branch)

        return losses
    
    def compute_occ_sdf_losses(self, pred_occ_sdf_bev, gt_occ_sdf_bev,mask):
        # seg_im = np.stack([freespace, occ_edge])
        # sdf_im = np.stack([sdf])
        # height_im = np.stack([height, heigh_mask])

        # 自动获取 batch
        batch = mask.size(0)
        # 判断每个样本 mask 是否全为 1，保持可广播形状
        all_one_flag = (mask.view(batch, -1).min(dim=1).values == 1).view(batch, 1, 1, 1)
        # 根据条件选择, 更新mask
        mask = torch.where(all_one_flag, torch.ones_like(mask), torch.zeros_like(mask))


        mask_seg = mask.expand(-1, pred_occ_sdf_bev['seg'].shape[1], -1, -1)
        mask_seg = F.interpolate(mask_seg, scale_factor=0.5, mode='nearest')
        pred_occ_sdf_bev['seg']  = pred_occ_sdf_bev['seg'] * mask_seg
        gt_occ_sdf_bev['seg'] = gt_occ_sdf_bev['seg']* mask_seg

        mask_sdf = mask.expand(-1, pred_occ_sdf_bev['sdf'].shape[1], -1, -1)
        mask_sdf = F.interpolate(mask_sdf, scale_factor=0.5, mode='nearest')
        pred_occ_sdf_bev['sdf']  = pred_occ_sdf_bev['sdf'] * mask_sdf
        gt_occ_sdf_bev['sdf'] = gt_occ_sdf_bev['sdf']* mask_sdf

        mask_height = mask.expand(-1, pred_occ_sdf_bev['height'].shape[1], -1, -1)
        mask_height = F.interpolate(mask_height, scale_factor=0.5, mode='nearest')
        pred_occ_sdf_bev['height']  = pred_occ_sdf_bev['height'] * mask_height
        gt_occ_sdf_bev['height'] = gt_occ_sdf_bev['height']* mask_height


        losses = {}
        losses['occ_seg_iou_0_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_dfl_0_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_iou_1_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        losses['occ_seg_dfl_1_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        
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
class ParkingFastRayPlanarSingleFrameModelAPA_Parking_only(BaseModel):
    
    def __init__(self,
                 backbone,
                 spatial_transform,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)
        # voxel heads
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        # init losses
        self.planar_losses_dict = {}
        if loss_cfg is not None:
            for branch in loss_cfg:
                self.planar_losses_dict[branch] = MODELS.build(loss_cfg[branch])


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
            bev_feats = bev_feats.reshape(N, C*Z, X, Y)
        ## voxel encoder
        bev_feats = self.voxel_encoder(bev_feats)
        ## heads & outputs
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        pred_dict = dict(

            parkingslot_3d=dict(
                cen=out_parkingslot_3d[0][:, :1],
                seg=out_parkingslot_3d[0][:, 1:],
                reg=out_parkingslot_3d[1],
                pts=out_parkingslot_3d[2],),
        )

        
        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']
            # gt_occ_sdf_bev = gt_dict['occ_sdf_bev']
        
        if self.debug_mode:
            print('skip drawing outputs')
            # images = draw_results_parking_IPM(pred_dict, batched_input_dict, save_im=True)

        if mode == 'tensor':
            pred_dict['occ_sdf_bev'] = pred_occ_sdf_bev
            return pred_dict
        if mode == 'loss':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))

            # losses['loss'] += losses['occ_sdf_bev_loss']
            return losses
        if mode == 'predict':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            losses['loss'] += losses['occ_sdf_bev_loss']
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
        losses['occ_seg_iou_0_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_dfl_0_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_iou_1_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        losses['occ_seg_dfl_1_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        
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
class ParkingFastRayPlanarSingleFrameModelAPA_DP(ParkingFastRayPlanarSingleFrameModelAPA):

    def forward(self, bev_feats):
        ## voxel encoder
        bev_feats = self.voxel_encoder(bev_feats)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        out_occ_sdf_bev = self.head_occ_sdf_bev(bev_feats)

        return out_bbox_3d, out_polyline_3d, out_parkingslot_3d, out_occ_sdf_bev


@MODELS.register_module()
class ParkingFastRayPlanarMultiFrameModelAPA(BaseModel):
    
    def __init__(self,
                 backbone,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 voxel_fusion_before_encoder=False,
                 loss_cfg=None,
                 pre_nframes=1,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        self.voxel_fusion = MODELS.build(voxel_fusion)
        self.voxel_fusion_before_encoder = voxel_fusion_before_encoder
        # voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        self.head_occ_sdf_bev = MODELS.build(heads['occ_sdf_bev'])
        # hidden voxel features for temporal fusion
        self.pre_nframes = pre_nframes
        self.cached_voxel_feats = {}
        self.cached_delta_poses = {}
        # init losses
        self.planar_losses_dict = {}
        for branch in loss_cfg:
            self.planar_losses_dict[branch] = MODELS.build(loss_cfg[branch])
        self.occ_seg_iou_loss = SegIouLoss()
        self.occ_seg_dfl_loss = DualFocalLoss()
        self.occ_sdf_l1_loss = nn.L1Loss(reduction='none')
        self.occ_height_l1_loss = nn.L1Loss(reduction='none')


    def temporal_fusion(self, batched_input_dict, voxel_feats_cur, delta_poses):
        ## temporal transform
        cur_first_index_info = batched_input_dict['index_infos'][0]
        # init tmp pre_0 cache
        self.cached_voxel_feats[f'pre_0'] = voxel_feats_cur
        self.cached_delta_poses[f'pre_0'] = delta_poses
        for pre_i in range(self.pre_nframes):
            index_info_prev_str = 'cur_first_index_info' + ''.join(['.prev'] * (pre_i+1))
            if eval(index_info_prev_str) is None:
                for pre_j in range(pre_i, self.pre_nframes):
                    self.cached_voxel_feats[f'pre_{pre_j+1}'] = self.cached_voxel_feats[f'pre_{pre_j}'].clone().detach()
                    self.cached_delta_poses[f'pre_{pre_j+1}'] = self.cached_delta_poses[f'pre_{pre_j}'].clone().detach()
                break
        # cache delta_poses
        for pre_i in range(self.pre_nframes, 1, -1):
            self.cached_delta_poses[f'pre_{pre_i}'] = (self.cached_delta_poses[f'pre_{pre_i-1}'] @ delta_poses).clone().detach()
        self.cached_delta_poses['pre_1'] = delta_poses.clone().detach()
        # align all history frames to current frame
        aligned_voxel_feats_cat = [voxel_feats_cur]
        for pre_i in range(self.pre_nframes):
            aligned_voxel_feats_cat.append(self.temporal_transform(
                self.cached_voxel_feats[f'pre_{pre_i+1}'], self.cached_delta_poses[f'pre_{pre_i+1}']
            ))
        # if self.debug_mode:
        #     draw_aligned_voxel_feats(aligned_voxel_feats_cat)
        # cache voxel features
        for pre_i in range(self.pre_nframes, 0, -1):
            self.cached_voxel_feats[f'pre_{pre_i}'] = (self.cached_voxel_feats[f'pre_{pre_i-1}']).clone().detach()
        # pop tmp pre_0 cache
        self.cached_voxel_feats.pop('pre_0')
        self.cached_delta_poses.pop('pre_0')
        ## voxel fusion
        voxel_feats_fused = self.voxel_fusion(*aligned_voxel_feats_cat)
        return voxel_feats_fused

    
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
        delta_poses = batched_input_dict['delta_poses']
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            camera_feats_dict[cam_id] = self.backbone(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        bev_feats = self.spatial_transform(camera_feats_dict, camera_lookups)
        if len(bev_feats.shape) == 5:
            N, C, Z, X, Y = bev_feats.shape
            bev_feats = bev_feats.reshape(N, C*Z, X, Y)
        ## voxel encoder and temporal fusion
        if self.voxel_fusion_before_encoder:
            bev_feats = self.temporal_fusion(batched_input_dict, bev_feats, delta_poses)
            bev_feats = self.voxel_encoder(bev_feats)
        else:
            bev_feats = self.voxel_encoder(bev_feats)
            bev_feats = self.temporal_fusion(batched_input_dict, bev_feats, delta_poses)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        out_occ_sdf_bev = self.head_occ_sdf_bev(bev_feats)
        
        pred_dict = dict(
            bbox_3d_heading=dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:14],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_plane_heading=dict(
                cen=out_bbox_3d[0][:, 14:15],
                seg=out_bbox_3d[0][:, 15:26],
                reg=out_bbox_3d[1][:, 20:40]),
            bbox_3d_no_heading=dict(
                cen=out_bbox_3d[0][:, 26:27],
                seg=out_bbox_3d[0][:, 27:34],
                reg=out_bbox_3d[1][:, 40:54]),
            bbox_3d_square=dict(
                cen=out_bbox_3d[0][:, 34:35],
                seg=out_bbox_3d[0][:, 35:40],
                reg=out_bbox_3d[1][:, 54:65]),
            bbox_3d_cylinder=dict(
                cen=out_bbox_3d[0][:, 40:41],
                seg=out_bbox_3d[0][:, 41:51],
                reg=out_bbox_3d[1][:, 65:73]),
            bbox_3d_oriented_cylinder=dict(
                cen=out_bbox_3d[0][:, 51:52],
                seg=out_bbox_3d[0][:, 52:54],
                reg=out_bbox_3d[1][:, 73:86]),
            polyline_3d=dict(
                seg=out_polyline_3d[0][:, 0:9],
                reg=out_polyline_3d[1][:, 0:7]),
            polygon_3d=dict(
                seg=out_polyline_3d[0][:, 9:15],
                reg=out_polyline_3d[1][:, 7:14]),
            parkingslot_3d=dict(
                cen=out_parkingslot_3d[0][:, :1],
                seg=out_parkingslot_3d[0][:, 1:],
                reg=out_parkingslot_3d[1]),
        )
        pred_occ_sdf_bev=dict(
            seg=out_occ_sdf_bev[0],
            sdf=out_occ_sdf_bev[1][:, 0:1],
            height=out_occ_sdf_bev[1][0:, 1:2],
        )
        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']
            gt_occ_sdf_bev = gt_dict['occ_sdf_bev']

        if self.debug_mode:
            import matplotlib.pyplot as plt

            plt.imshow(gt_occ_sdf_bev['seg'][0][0].detach().cpu().numpy()); plt.show()
            freespace = pred_occ_sdf_bev['seg'][0][0].sigmoid().detach().cpu().numpy() > 0.5
            plt.imshow(freespace); plt.show()

            plt.imshow(gt_occ_sdf_bev['seg'][0][1].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['seg'][0][1].sigmoid().detach().cpu().numpy() > 0.5); plt.show()

            plt.imshow(gt_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy() > 0); plt.show()

            plt.imshow(gt_occ_sdf_bev['height'][0][0].detach().cpu().numpy()); plt.show()
            plt.imshow(pred_occ_sdf_bev['height'][0][0].detach().cpu().numpy()); plt.show()

            # draw_outputs(pred_dict, batched_input_dict)
            save_outputs(pred_dict, batched_input_dict)
            # save_pred_outputs(pred_dict)
            # save occ_sdf_bev
            # save_occ_sdf_bev(pred_occ_sdf_bev)
        
        if mode == 'tensor':
            pred_dict['occ_sdf_bev'] = pred_occ_sdf_bev
            return pred_dict
        if mode == 'loss':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            losses['loss'] += losses['occ_sdf_bev_loss']
            return losses
        if mode == 'predict':
            losses = {}
            losses.update(self.compute_planar_losses(pred_dict, gt_dict))
            losses.update(self.compute_occ_sdf_losses(pred_occ_sdf_bev, gt_occ_sdf_bev))
            losses['loss'] += losses['occ_sdf_bev_loss']
            return (
                *[{branch: {k: v.cpu() for k, v in _pred.items()}} for branch, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_planar_losses(self, pred_dict, gt_dict):
        losses = dict(loss=0)
        for branch in self.planar_losses_dict:
            losses_branch = self.planar_losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            # if branch + '_seg_iou_0_loss' in losses_branch:
            #     losses[branch + '_seg_iou_0_loss'] = losses_branch[branch + '_seg_iou_0_loss']
            losses.update(losses_branch)

        return losses
    
    def compute_occ_sdf_losses(self, pred_occ_sdf_bev, gt_occ_sdf_bev):
        # seg_im = np.stack([freespace, occ_edge])
        # sdf_im = np.stack([sdf])
        # height_im = np.stack([height, heigh_mask])
        losses = {}
        losses['occ_seg_iou_0_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_dfl_0_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 0:1], gt_occ_sdf_bev['seg'][:, 0:1])
        losses['occ_seg_iou_1_loss'] = self.occ_seg_iou_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        losses['occ_seg_dfl_1_loss'] = self.occ_seg_dfl_loss(pred_occ_sdf_bev['seg'][:, 1:2], gt_occ_sdf_bev['seg'][:, 1:2])
        
        sdf_mask = gt_occ_sdf_bev['seg'][:, 0:1] + 0.1
        sdf_loss = self.occ_sdf_l1_loss(pred_occ_sdf_bev['sdf'] * sdf_mask, gt_occ_sdf_bev['sdf'] * sdf_mask)
        losses['occ_sdf_loss'] = sdf_loss.sum() / sdf_mask.sum()

        gt_height = gt_occ_sdf_bev['height'][:, 0:1]
        gt_height_mask = gt_occ_sdf_bev['height'][:, 1:2]
        pred_height = pred_occ_sdf_bev['height']
        occ_height_loss = self.occ_height_l1_loss(pred_height * gt_height_mask, gt_height * gt_height_mask)
        losses['occ_height_loss'] = occ_height_loss.sum() / gt_height_mask.sum()

        losses['occ_sdf_bev_loss'] = 5 * losses['occ_seg_iou_0_loss'] + 10 * losses['occ_seg_iou_1_loss'] + \
            10 * losses['occ_seg_dfl_0_loss'] + 20 * losses['occ_seg_dfl_1_loss'] + \
            20 * losses['occ_sdf_loss'] + 20 * losses['occ_height_loss']
        
        return losses


@MODELS.register_module()
class ParkingFastRayPlanarMultiFrameModelAPALidar(BaseModel):

    def __init__(self,
                 backbone,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 voxel_encoder,
                 heads,
                 lidar_voxel_fusion=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 debug_mode=False,
                 voxel_fusion_before_encoder=False,
                 loss_cfg=None,
                 pre_nframes=1,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)

        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.lidar_voxel_fusion=MODELS.build(lidar_voxel_fusion)
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        self.voxel_fusion = MODELS.build(voxel_fusion)
        self.voxel_fusion_before_encoder = voxel_fusion_before_encoder
        # voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        # self.head_occ_sdf_bev = MODELS.build(heads['occ_sdf_bev'])

        # hidden voxel features for temporal fusion
        self.pre_nframes = pre_nframes
        self.cached_voxel_feats = {}
        self.cached_delta_poses = {}
        # init losses
        self.planar_losses_dict = {}
        for branch in loss_cfg:
            self.planar_losses_dict[branch] = MODELS.build(loss_cfg[branch])
        self.occ_seg_iou_loss = SegIouLoss()
        self.occ_seg_dfl_loss = DualFocalLoss()
        self.occ_sdf_l1_loss = nn.L1Loss(reduction='none')
        self.occ_height_l1_loss = nn.L1Loss(reduction='none')

    def temporal_fusion(self, batched_input_dict, voxel_feats_cur, delta_poses):
        ## temporal transform
        cur_first_index_info = batched_input_dict['index_infos'][0]
        # init tmp pre_0 cache
        self.cached_voxel_feats[f'pre_0'] = voxel_feats_cur
        self.cached_delta_poses[f'pre_0'] = delta_poses
        for pre_i in range(self.pre_nframes):
            index_info_prev_str = 'cur_first_index_info' + ''.join(['.prev'] * (pre_i + 1))
            if eval(index_info_prev_str) is None:
                for pre_j in range(pre_i, self.pre_nframes):
                    self.cached_voxel_feats[f'pre_{pre_j + 1}'] = self.cached_voxel_feats[
                        f'pre_{pre_j}'].clone().detach()
                    self.cached_delta_poses[f'pre_{pre_j + 1}'] = self.cached_delta_poses[
                        f'pre_{pre_j}'].clone().detach()
                break
        # cache delta_poses
        for pre_i in range(self.pre_nframes, 1, -1):
            self.cached_delta_poses[f'pre_{pre_i}'] = (
                        self.cached_delta_poses[f'pre_{pre_i - 1}'] @ delta_poses).clone().detach()
        self.cached_delta_poses['pre_1'] = delta_poses.clone().detach()
        # align all history frames to current frame
        aligned_voxel_feats_cat = [voxel_feats_cur]
        for pre_i in range(self.pre_nframes):
            aligned_voxel_feats_cat.append(self.temporal_transform(
                self.cached_voxel_feats[f'pre_{pre_i + 1}'], self.cached_delta_poses[f'pre_{pre_i + 1}']
            ))
        # if self.debug_mode:
        #     draw_aligned_voxel_feats(aligned_voxel_feats_cat)
        # cache voxel features
        for pre_i in range(self.pre_nframes, 0, -1):
            self.cached_voxel_feats[f'pre_{pre_i}'] = (self.cached_voxel_feats[f'pre_{pre_i - 1}']).clone().detach()
        # pop tmp pre_0 cache
        self.cached_voxel_feats.pop('pre_0')
        self.cached_delta_poses.pop('pre_0')
        ## voxel fusion
        voxel_feats_fused = self.voxel_fusion(*aligned_voxel_feats_cat)
        return voxel_feats_fused

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
        # --------------
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        delta_poses = batched_input_dict['delta_poses']
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            camera_feats_dict[cam_id] = self.backbone(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        bev_feats = self.spatial_transform(camera_feats_dict, camera_lookups)
        if len(bev_feats.shape) == 5:
            N, C, Z, X, Y = bev_feats.shape
            bev_feats = bev_feats.reshape(N, C * Z, X, Y)
        ## voxel encoder and temporal fusion
        if self.voxel_fusion_before_encoder:
            bev_feats = self.temporal_fusion(batched_input_dict, bev_feats, delta_poses)
            bev_feats = self.voxel_encoder(bev_feats)
        else:
            bev_feats = self.voxel_encoder(bev_feats)
            bev_feats = self.temporal_fusion(batched_input_dict, bev_feats, delta_poses)
        bev_feats = self.lidar_voxel_fusion(bev_feats, lidar_features)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        out_polyline_3d = self.head_polyline_3d(bev_feats)
        out_parkingslot_3d = self.head_parkingslot_3d(bev_feats)
        # out_occ_sdf_bev = self.head_occ_sdf_bev(bev_feats)

        pred_dict = dict(
            bbox_3d_heading=dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:14],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_plane_heading=dict(
                cen=out_bbox_3d[0][:, 14:15],
                seg=out_bbox_3d[0][:, 15:26],
                reg=out_bbox_3d[1][:, 20:40]),
            bbox_3d_no_heading=dict(
                cen=out_bbox_3d[0][:, 26:27],
                seg=out_bbox_3d[0][:, 27:34],
                reg=out_bbox_3d[1][:, 40:54]),
            bbox_3d_square=dict(
                cen=out_bbox_3d[0][:, 34:35],
                seg=out_bbox_3d[0][:, 35:40],
                reg=out_bbox_3d[1][:, 54:65]),
            bbox_3d_cylinder=dict(
                cen=out_bbox_3d[0][:, 40:41],
                seg=out_bbox_3d[0][:, 41:51],
                reg=out_bbox_3d[1][:, 65:73]),
            bbox_3d_oriented_cylinder=dict(
                cen=out_bbox_3d[0][:, 51:52],
                seg=out_bbox_3d[0][:, 52:54],
                reg=out_bbox_3d[1][:, 73:86]),
            polyline_3d=dict(
                seg=out_polyline_3d[0][:, 0:9],
                reg=out_polyline_3d[1][:, 0:7]),
            polygon_3d=dict(
                seg=out_polyline_3d[0][:, 9:15],
                reg=out_polyline_3d[1][:, 7:14]),
            parkingslot_3d=dict(
                cen=out_parkingslot_3d[0][:, :1],
                seg=out_parkingslot_3d[0][:, 1:],
                reg=out_parkingslot_3d[1]),
        )
        # pred_occ_sdf_bev = dict(
        #     seg=out_occ_sdf_bev[0],
        #     sdf=out_occ_sdf_bev[1][:, 0:1],
        #     height=out_occ_sdf_bev[1][0:, 1:2],
        # )
        if 'annotations' in batched_input_dict:
            gt_dict = batched_input_dict['annotations']
            # gt_occ_sdf_bev = gt_dict['occ_sdf_bev']

        # if self.debug_mode:
        if False:
            import matplotlib.pyplot as plt

            plt.imshow(gt_occ_sdf_bev['seg'][0][0].detach().cpu().numpy());
            plt.show()
            freespace = pred_occ_sdf_bev['seg'][0][0].sigmoid().detach().cpu().numpy() > 0.5
            plt.imshow(freespace);
            plt.show()

            plt.imshow(gt_occ_sdf_bev['seg'][0][1].detach().cpu().numpy());
            plt.show()
            plt.imshow(pred_occ_sdf_bev['seg'][0][1].sigmoid().detach().cpu().numpy() > 0.5);
            plt.show()

            plt.imshow(gt_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy());
            plt.show()
            plt.imshow(pred_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy());
            plt.show()
            plt.imshow(pred_occ_sdf_bev['sdf'][0][0].detach().cpu().numpy() > 0);
            plt.show()

            plt.imshow(gt_occ_sdf_bev['height'][0][0].detach().cpu().numpy());
            plt.show()
            plt.imshow(pred_occ_sdf_bev['height'][0][0].detach().cpu().numpy());
            plt.show()

            draw_outputs(pred_dict, batched_input_dict)
            # save_outputs(pred_dict, batched_input_dict)
            # TODO: save occ_sdf_bev

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
            losses['loss'] += losses['occ_sdf_bev_loss']
            return (
                *[{branch: {t: v.cpu() for t, v in _pred.items()}} for branch, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_planar_losses(self, pred_dict, gt_dict):
        losses = dict(loss=0)
        for branch in self.planar_losses_dict:
            losses_branch = self.planar_losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            # if branch + '_seg_iou_0_loss' in losses_branch:
            #     losses[branch + '_seg_iou_0_loss'] = losses_branch[branch + '_seg_iou_0_loss']
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

        losses['occ_sdf_bev_loss'] = 5 * losses['occ_seg_iou_0_loss'] + 10 * losses['occ_seg_iou_1_loss'] + \
                                     10 * losses['occ_seg_dfl_0_loss'] + 20 * losses['occ_seg_dfl_1_loss'] + \
                                     20 * losses['occ_sdf_loss'] + 20 * losses['occ_height_loss']

        return losses


@MODELS.register_module()
class ParkingFastRayPlanarSingleFrameModelAPALidar(BaseModel):

    def __init__(self,
                 backbone,
                 spatial_transform,
                 voxel_encoder,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs
                 ):
        super().__init__(data_preprocessor, init_cfg)
        self.with_lidar = True
        if self.with_lidar:
            self.pts_middle_encoder = MODELS.build(kwargs['pts_middle_encoder'])
            self.pts_backbone = MODELS.build(kwargs['pts_backbone'])
            self.pts_neck = MODELS.build(kwargs['pts_neck'])
            self.lidar_voxel_fusion = MODELS.build(kwargs['lidar_voxel_fusion'])
        self.debug_mode = debug_mode
        # backbone
        self.backbone = MODELS.build(backbone)
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        # init losses
        self.planar_losses_dict = {}
        if loss_cfg is not None:
            for branch in loss_cfg:
                self.planar_losses_dict[branch] = MODELS.build(loss_cfg[branch])

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


        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)

        pred_dict = dict(
            bbox_3d_heading=dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:16],
                reg=out_bbox_3d[1][:, 0:20])
        )

        if batched_input_dict['annotations']:
            gt_dict = batched_input_dict['annotations']

        if mode == "infer":
            img = draw_results_planar_lidar(pred_dict, batched_input_dict, False)
            return img
        if mode == 'tensor':
            return pred_dict
        if mode == 'loss':
            losses = self.compute_planar_losses(pred_dict, gt_dict)
            return losses
        if mode == 'predict':
            losses = self.compute_planar_losses(pred_dict, gt_dict)
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
