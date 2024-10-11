custom_imports = dict(imports=["prefusion", "contrib.cmt"], allow_failed_imports=False)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

class_names = [
    'class.cycle.bicycle',
    'class.cycle.motorcycle',
    'class.cycle.tricycle',
    'class.parking.charging_infra',
    'class.parking.indoor_column',
    'class.parking.parking_lock',
    'class.parking.text_icon',
    'class.parking.wheel_stopper',
    'class.pedestrian.pedestrian',
    'class.road_marker.arrow',
    'class.road_marker.text',
    'class.sign.traffic_sign.instruction',
    'class.sign.traffic_sign.prohibition_and_limit',
    'class.traffic_facility.bollard',
    'class.traffic_facility.box',
    'class.traffic_facility.cone',
    'class.traffic_facility.gate_barrier',
    'class.traffic_facility.hard_barrier',
    'class.traffic_facility.soft_barrier',
    'class.traffic_facility.speed_bump',
    'class.vehicle.env_protect',
    'class.vehicle.passenger_car',
    'class.vehicle.truck'
]

voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
total_epochs = 30
evaluation = dict(interval=total_epochs)
# dataset_type = 'CustomNuScenesDataset'
dataset_type = 'CustomMTVDataset'
data_root = 'data/mtv/'
data_ori_root = 'data/nuscenes'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

ida_aug_conf = {
    "resize_lim": (0.3, 0.4),
    "final_dim": (352, 576),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 1080,
    "W": 1920,
    "rand_flip": True,
}

train_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweepsPCD',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),

    dict(type='ModalMask3D', mode='train'),
    dict(  # lidar
        type='GlobalRotScaleTransAll',
        # rot_range=[-0.3925 * 8, 0.3925 * 8],  # -45, 45;r
        rot_range=[-0.3925 * 2, 0.3925 * 2],  # -45, 45;r
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(  # flip
        type='CustomRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d',
                    'cam_intrinsic', 'lidar2cam', 'cam_inv_poly'
                    ))
]
test_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweepsPCD',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=[
                     'points', 'img', 'cam_intrinsic', 'lidar2cam', 'cam_inv_poly', # 'gt_labels_3d'
                 ],
                 meta_keys=[
                     'pad_shape', 'lidar2img', 'lidar2cam',
                     'cam_intrinsic', 'lidar2cam', 'cam_inv_poly', 'intrinsic',
                     'box_type_3d', 'filename', 'gt_labels_3d'
                 ])
        ])
]
data = dict(
    samples_per_gpu=2,  # gpu = 4
    workers_per_gpu=2,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'cmt_pkl_lidar_ego.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'cmt_pkl_lidar_ego.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/cmt_pkl_lidar_ego_validate_train.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)
model = dict(
    type='CmtDetector',
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4', 'stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='mmdet3d.HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='mmdet3d.SparseEncoder',
        in_channels=5,
            sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='mmdet3d.SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='mmdet3d.SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmtFisheyeHead',
        # num_query=1500,
        num_query=900,
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=len(class_names), class_names=class_names),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names)),
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
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
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,  # unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))
optimizer = dict(
    type='AdamW',
    lr=0.0002,  # 0.0001， 最大是0.0004，其实感觉用模拟退火可能效果更好。
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(pts_voxel_encoder=False, pts_middle_encoder=False, pts_bbox_head=False))
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(8, 0.0001),
#     cyclic_times=1,
#     step_ratio_up=0.4)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
# load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
# load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
# load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
# load_from = 'work_dirs/cmt_voxel0075_vov_1600x640_cbgs_dbg/epoch_10.pth'
# load_from = 'picked_models/cmt_11cls_9scenes_0712.pth'
# load_from = 'picked_models/epoch_1_one_lidar.pth'
# load_from = 'work_dirs/mtv_voxel0075_vov_640x320_temporal/epoch_16.pth'
load_from = 'work_dirs/cmt_voxel0075_vov_640x320_cbgs_dbg_rotation/epoch_18.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)

#-----------------------------------------------
# dataset
#-----------------------------------------------
IMG_KEYS = [
        'VCAMERA_FISHEYE_FRONT', 'VCAMERA_PERSPECTIVE_FRONT_LEFT', 'VCAMERA_PERSPECTIVE_BACK_LEFT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_PERSPECTIVE_BACK', 'VCAMERA_FISHEYE_BACK',
        'VCAMERA_PERSPECTIVE_FRONT_RIGHT', 'VCAMERA_PERSPECTIVE_BACK_RIGHT', 'VCAMERA_FISHEYE_RIGHT', 'VCAMERA_PERSPECTIVE_FRONT'
        ]
data_root = "data/mv_4d_data/"

W, H = 120, 240
bev_front = 180
bev_left = 60
voxel_size = [0.2, 0.2, 0.5]
downsample_factor=8


fish_img_size = [256, 160]
perspective_img_size = [256, 192]
front_perspective_img_size = [768, 384]
batch_size = 4
group_size = 3
bev_range = [-12, 36, -12, 12, -10, 10]
frame_start_end_ids = [1,0,2]  # group size is 3

voxel_feature_config = dict(
    voxel_shape=(6, H, W),  # Z, X, Y in ego system
    voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
    ego_distance_max=40,
    ego_distance_step=2
)

general_camera_feature_config = dict(
    ray_distance_num_channel=64,
    ray_distance_start=0.25,
    ray_distance_step=0.25,
    feature_downscale=downsample_factor,
)

train_pipeline = [
    dict(type='IntrinsicImage',
        resolutions=dict(
            VCAMERA_PERSPECTIVE_FRONT=front_perspective_img_size,
            VCAMERA_PERSPECTIVE_FRONT_LEFT=perspective_img_size,
            VCAMERA_PERSPECTIVE_BACK_LEFT=perspective_img_size,
            VCAMERA_PERSPECTIVE_BACK=perspective_img_size,
            VCAMERA_PERSPECTIVE_BACK_RIGHT=perspective_img_size,
            VCAMERA_PERSPECTIVE_FRONT_RIGHT=perspective_img_size,
            VCAMERA_FISHEYE_FRONT=fish_img_size,
            VCAMERA_FISHEYE_LEFT=fish_img_size,
            VCAMERA_FISHEYE_BACK=fish_img_size,
            VCAMERA_FISHEYE_RIGHT=fish_img_size,
        ),
        scope='frame'
    ),
    dict(type='RandomExtrinsicImage',
        prob=0.8,
        angles=[0,0,3],
        scope='frame'),
    dict(type='FastRayLookUpTable',
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=dict(
            VCAMERA_FISHEYE_FRONT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_FRONT_LEFT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_BACK_LEFT=general_camera_feature_config,
            VCAMERA_FISHEYE_LEFT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_BACK=general_camera_feature_config,
            VCAMERA_FISHEYE_BACK=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_FRONT_RIGHT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_BACK_RIGHT=general_camera_feature_config,
            VCAMERA_FISHEYE_RIGHT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_FRONT=general_camera_feature_config
            )
        ),
    dict(type='ToTensor', bev_resolution=[H, W], bev_range=bev_range)
]

val_pipeline = [
    dict(type='IntrinsicImage',
        resolutions=dict(
            VCAMERA_PERSPECTIVE_FRONT=front_perspective_img_size,
            VCAMERA_PERSPECTIVE_FRONT_LEFT=perspective_img_size,
            VCAMERA_PERSPECTIVE_BACK_LEFT=perspective_img_size,
            VCAMERA_PERSPECTIVE_BACK=perspective_img_size,
            VCAMERA_PERSPECTIVE_BACK_RIGHT=perspective_img_size,
            VCAMERA_PERSPECTIVE_FRONT_RIGHT=perspective_img_size,
            VCAMERA_FISHEYE_FRONT=fish_img_size,
            VCAMERA_FISHEYE_LEFT=fish_img_size,
            VCAMERA_FISHEYE_BACK=fish_img_size,
            VCAMERA_FISHEYE_RIGHT=fish_img_size,
        ),
        scope='frame'
    ),
    dict(type='ToTensor', bev_resolution=[H, W], bev_range=bev_range)
]

CLASSES = ['class.vehicle.passenger_car', 'class.traffic_facility.box', 'class.traffic_facility.soft_barrier', 'class.traffic_facility.hard_barrier', \
           'class.road_marker.arrow', 'class.traffic_facility.speed_bump', 'class.parking.wheel_stopper',\
           'class.parking.indoor_column']

bbox3d = dict(
    branch_0=dict(classes=['class.vehicle.passenger_car', 'class.traffic_facility.box', 'class.traffic_facility.soft_barrier', 'class.traffic_facility.hard_barrier'], attrs=[['attr.vehicle.is_door_open', 'attr.vehicle.is_trunk_open'], [], [], []])
)

BboxBev = dict(
    branch_0=dict(classes=['class.road_marker.arrow', 'class.traffic_facility.speed_bump', 'class.parking.wheel_stopper'], attrs=[[], [], []]),
)

Cylinder3D = dict(

)

Square3D = dict(
    branch_0=dict(classes=['class.parking.indoor_column'], attrs=[['attr.parking.indoor_column.shape']])
)

collection_info_type = ['camera_images','camera_depths', 'bbox_3d', 'bbox_bev', 'square_3d']

dictionary=dict(
        bbox_3d=bbox3d,
        bbox_bev=BboxBev,
        square_3d=Square3D,
        # cylinder_3d=Cylinder3D
        )


dictionary=dict(
        bbox_3d=bbox3d,
        bbox_bev=BboxBev,
        square_3d=Square3D,
        # cylinder_3d=Cylinder3D
        )

train_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler'),
    collate_fn=dict(type='collate_generator'),
    dataset=dict(
        type='GroupBatchDataset',
        name="mv_4d",
        data_root=data_root,
        info_path=data_root + 'mv_4d_infos.pkl',
        dictionary=dictionary,
        transformable_keys=collection_info_type,
        transforms=train_pipeline,
        phase='train',
        batch_size=batch_size,
        group_size=group_size,
        ),
    )

val_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler'),
    collate_fn=dict(type='collate_generator'),
    dataset=dict(
        type='GroupBatchDataset',
        name="mv_4d",
        data_root=data_root,
        info_path=data_root + 'mv_4d_infos.pkl',
        dictionary=dictionary,
        transformable_keys=collection_info_type,
        transforms=train_pipeline,
        phase='val',
        batch_size=batch_size,
        group_size=group_size,
        ),
    )
model_train_cfg = dict(
    available_elements = ['heatmap', 'anno_boxes', 'gridzs', 'class_maps'],
    to_mv_coord = [36, 12, 3],
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    is_train_depth=True
)

model_test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='rotate',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)
