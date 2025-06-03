from contrib.ray_spetr import RaySPETR
from contrib.pretrain import vovnet
from contrib.petr import FocalHead
from contrib.cmt import CmtDetector

import torch
from prefusion.registry import MODELS

def _calc_grid_size(_range, _voxel_size, n_axis=3):
    return [(_range[n_axis+i] - _range[i]) // _voxel_size[i] for i in range(n_axis)]

possible_group_sizes = [20]
voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_range = (point_cloud_range[2::3], point_cloud_range[0::3][::-1], point_cloud_range[1::3][::-1])
grid_size = _calc_grid_size(point_cloud_range, voxel_size)

model = dict(
    type="RaySPETR",
    data_preprocessor=dict(
        type="FrameBatchMerger",
        device="cuda",
    ),
    img_backbone=dict(type='VoVNet',
                out_indices=(0, 1, 3, 5),
                init_cfg=dict(type='Pretrained', checkpoint="./work_dirs/front_pretrian/epoch-13.pth")
                ),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs=False, 
        num_outs=2,
        relu_before_extra_convs=True,
        init_cfg=dict(type='Pretrained', checkpoint="./work_dirs/front_pretrian/epoch-13.pth")),
    # img_neck=dict(type="mmdet3d.CPFPN", in_channels=[1024, 2048], out_channels=256, num_outs=2),
    roi_head=dict(
        type="RayFocalHead",
        num_classes=10,
        loss_cls2d=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0))
        ),
    ),
    box_head=dict(
        type='RaySPETRHead',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10, ##noise groups
        noise_scale = 1.0,
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        LID=True,
        with_position=True,
        code_size=10, # x, y, z, l, w, h, sin(yaw), cos(yaw), Vx, Vy
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        train_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="mmdet.HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(type="IoUCost", weight=0.0 ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        ),
    ),
)

def test_ray_spetr_model():
    model = MODELS.build(model)
    print(model.backbone)
    print(model.img_neck)   