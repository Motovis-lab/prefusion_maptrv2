default_scope = "prefusion"
experiment_name = "fastbev_single_frame_nusc_r50"

custom_imports = dict(
    imports=["prefusion", "contrib.fastbev_original", "contrib.petr"],
    allow_failed_imports=False)


## camera and voxel feature configs
feature_downscale = 4
default_camera_feature_config = dict(
    ray_distance_num_channel=64,
    ray_distance_start=0.25,
    ray_distance_step=0.25,
    feature_downscale=feature_downscale)

camera_feature_configs = dict(
    CAM_FRONT=default_camera_feature_config,
    CAM_FRONT_RIGHT=default_camera_feature_config,
    CAM_BACK_RIGHT=default_camera_feature_config,
    CAM_BACK=default_camera_feature_config,
    CAM_BACK_LEFT=default_camera_feature_config,
    CAM_FRONT_LEFT=default_camera_feature_config
)

voxel_shape = (6, 256, 256)  # Z, X, Y in ego system
voxel_range = ([-5, 3], [50, -50], [50, -50])
# voxel_range = ([-0.5, 2.5], [30, -12], [12, -12])

voxel_feature_config = dict(
    voxel_shape=voxel_shape,
    voxel_range=voxel_range,
    ego_distance_max=75,  # 50 * sqrt(2)
    ego_distance_step=5)

## dictionaries and mappings for different types of tasks


## camera configs for model inputs

camera_groups = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT',
]

# 1600 x 900, 1408 x 512, 1056 x 384, 704 x 256
resolution_pv = (880, 320)

camera_resolution_configs=dict(
    CAM_FRONT=resolution_pv,
    CAM_FRONT_RIGHT=resolution_pv,
    CAM_BACK_RIGHT=resolution_pv,
    CAM_BACK=resolution_pv,
    CAM_BACK_LEFT=resolution_pv,
    CAM_FRONT_LEFT=resolution_pv)

# camera_intrinsic_configs is calculated by the following code snippet
# H, W = 900, 1600
# new_H, new_W = 320, 880
# for cam_name in NUSC_CAM_NAMES:
#     intr = nusc.get("calibrated_sensor", nusc.get("sample_data", first_sample['data'][cam_name])['calibrated_sensor_token'])['camera_intrinsic']
#     fx, fy, cx, cy = intr[0][0], intr[1][1], intr[0][2], intr[1][2]
#     scale = new_W / W
#     new_fx = fx * scale
#     new_fy = fy * scale
#     new_cx = cx * scale
#     top_to_crop = H * scale - new_H
#     cy_if_no_crop = cy * scale
#     new_cy = cy_if_no_crop - top_to_crop
#     print((f"{cam_name}=" + "{:.3f}, " * 4).format(new_cx, new_cy, new_fx, new_fy))
camera_intrinsic_configs_top_crop = dict(
    CAM_FRONT=[454.623, 83.492, 689.047, 689.047],
    CAM_FRONT_RIGHT=[449.784, 73.575, 691.212, 691.212],
    CAM_BACK_RIGHT=[453.957, 79.401, 687.480, 687.480],
    CAM_BACK=[471.778, 87.287, 438.290, 438.290],
    CAM_BACK_LEFT=[456.267, 81.942, 690.242, 690.242],
    CAM_FRONT_LEFT=[454.983, 73.004, 691.824, 691.824],
)

# new_cy = cy_if_no_crop - to_crop / 2
camera_intrinsic_configs_center_crop = dict(
    CAM_FRONT=[454.623, 170.992, 689.047, 689.047],
    CAM_FRONT_RIGHT=[449.784, 161.075, 691.212, 691.212],
    CAM_BACK_RIGHT=[453.957, 166.901, 687.480, 687.480],
    CAM_BACK=[471.778, 174.787, 438.290, 438.290],
    CAM_BACK_LEFT=[456.267, 169.442, 690.242, 690.242],
    CAM_FRONT_LEFT=[454.983, 160.504, 691.824, 691.824],
)

camera_intrinsic_configs = camera_intrinsic_configs_center_crop

debug_mode = False

if debug_mode:
    batch_size = 1
    num_workers = 0
    persistent_workers = False
    transforms = [
        dict(type='BGR2RGB'),
        dict(type='RenderIntrinsic',
             resolutions=camera_resolution_configs,
             intrinsics=camera_intrinsic_configs)
    ]
    possible_group_sizes = 20
else:
    batch_size = 8
    num_workers = 3
    persistent_workers = True
    transforms = [
        # dict(type='RandomRenderExtrinsic'),
        dict(type='BGR2RGB'),
        dict(type='RenderIntrinsic', resolutions=camera_resolution_configs, intrinsics=camera_intrinsic_configs),
        # dict(type='RandomRotateSpace', angles=[0, 0, 360], prob_inverse_cameras_rotation=0),
        # dict(type='RandomMirrorSpace'),
        # dict(type='RandomImageISP', prob=0.2),
        # dict(type='RandomSetIntrinsicParam', prob=0.2, jitter_ratio=0.01),
        # dict(type='RandomSetExtrinsicParam', prob=0.2, angle=1, translation=0.02)
    ]
    possible_group_sizes = 1

class_mapping = dict(
    bicycle=["vehicle.bicycle"],
    car=["vehicle.car"],
    construction_vehicle=["vehicle.construction"],
    motorcycle=["vehicle.motorcycle"],
    trailer=["vehicle.trailer"],
    truck=["vehicle.truck"],
    bus=["vehicle.bus.bendy", "vehicle.bus.rigid"],
    traffic_cone=["movable_object.trafficcone"],
    pedestrian=["human.pedestrian.adult" ,"human.pedestrian.child" ,"human.pedestrian.construction_worker" ,"human.pedestrian.police_officer"],
    barrier=['movable_object.barrier'],
)

## Transformables
transformables = dict(
    sample_token=dict(type='Variable', loader=dict(type="VariableLoader", variable_key="sample_token")),
    camera_images=dict(
        type='CameraImageSet',
        loader=dict(type="NuscenesCameraImageSetLoader"),
        tensor_smith=dict(type='CameraImageTensor', means=[123.675, 116.28, 103.53], stds=[58.395, 57.12, 57.375]),  # ImageNet mean and std
    ),
    ego_poses=dict(type='EgoPoseSet'),
    bbox_3d=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=class_mapping,
        ),
        tensor_smith=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
    ),
    bbox_3d_basic=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=class_mapping,
        ),
        # tensor_smith=dict(type='XyzLwhYawVeloBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
        tensor_smith=dict(type='Bbox3DBasic', classes=list(class_mapping.keys()), voxel_range=voxel_range)
    ),
)


## GroupBatchDataset configs
train_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='/data/datasets/nuScenes',
    info_path='/data/datasets/nuScenes/nusc_train_info.pkl',
    # info_path='/data/datasets/nuScenes/nusc_scene0103_train_info.pkl',
    # info_path='/data/datasets/nuScenes/nusc_scene0103_87e772078a494d42bd34cd16172808bc_train_info.pkl',
    model_feeder=dict(
        type="FastBEVModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode),
    transformables=transformables,
    transforms=transforms,
    group_sampler=dict(type="IndexGroupSampler",
                       phase="train",
                       possible_group_sizes=1,
                       possible_frame_intervals=1),
    # group_sampler=dict(type="ClassBalancedGroupSampler",
    #                    phase="train",
    #                    possible_group_sizes=1,
    #                    possible_frame_intervals=1,
    #                    transformable_cfg=transformables,
    #                    cbgs_cfg=dict(desired_ratio=0.3)),
    batch_size=batch_size,
)

val_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='/data/datasets/nuScenes',
    # info_path='/data/datasets/nuScenes/nusc_scene0103_val_info.pkl',
    info_path='/data/datasets/nuScenes/nusc_val_info.pkl',
    # info_path='/data/datasets/nuScenes/nusc_scene0103_87e772078a494d42bd34cd16172808bc_val_info.pkl',
    model_feeder=dict(
        type="FastBEVModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode,
    ),
    transformables=transformables,
    transforms=[
        dict(type='BGR2RGB'),
        dict(type='RenderIntrinsic', resolutions=camera_resolution_configs, intrinsics=camera_intrinsic_configs)
    ],
    group_sampler=dict(type="IndexGroupSampler",
                       phase="val",
                       possible_group_sizes=possible_group_sizes,
                       possible_frame_intervals=1),
    batch_size=batch_size,
)

test_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='/data/datasets/nuScenes',
    # info_path='/data/datasets/nuScenes/nusc_scene0103_87e772078a494d42bd34cd16172808bc_val_info.pkl',
    info_path='/data/datasets/nuScenes/nusc_val_info.pkl',
    model_feeder=dict(
        type="FastBEVModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode,
    ),
    transformables=transformables,
    transforms=[
        dict(type='BGR2RGB'),
        dict(type='RenderIntrinsic',
             resolutions=camera_resolution_configs,
             intrinsics=camera_intrinsic_configs)
    ],
    group_sampler=dict(type="SequentialSceneFrameGroupSampler",
                       phase="test_scene_by_scene"),
    batch_size=1,
)

## dataloader configs
train_dataloader = dict(
    num_workers=num_workers,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset,
    persistent_workers=persistent_workers,
    pin_memory=True,
)

val_dataloader = dict(
    num_workers=num_workers,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=val_dataset,
    persistent_workers=persistent_workers,
    pin_memory=True,
)

test_dataloader = dict(
    num_workers=num_workers,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="collate_dict"),
    dataset=test_dataset,
    persistent_workers=persistent_workers,
    pin_memory=True,
)


## model configs
# backbones
backbones = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
)

neck=dict(
    type='mmdet.FPN',
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    # in_channels=[64, 128, 256, 512],
    in_channels=[256, 512, 1024, 2048],
    out_channels=64,
    num_outs=4
)

neck_fuse=dict(in_channels=256, out_channels=64)

neck_3d=dict(
    type='M2BevNeck',
    in_channels=64*voxel_shape[0],
    out_channels=256, # ought to be: 64*voxel_shape[0]//2,
    num_layers=6,
    stride=2,
    is_transpose=False,
    fuse=dict(in_channels=64*voxel_shape[0], out_channels=64*voxel_shape[0]),
    norm_cfg=dict(type='SyncBN', requires_grad=True)
)

# spatial_transform
spatial_transform = dict(
    type='FastRaySpatialTransform',
    voxel_shape=voxel_shape,
    fusion_mode='bilinear_weighted',
    reduce_channels=False,
    bev_mode=False)

# loss configs
bbox_3d_weight_scheme = dict(
    cen=dict(loss_weight=2.5,
             fg_weight=0.5,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5.0,
             dual_focal_loss_weight=10, # 10
             channel_weights=dict(any={"weight": 0.5}, # 0.5
                                  bicycle={"weight": 3}, # 5
                                  car={"weight": 1},
                                  construction_vehicle={"weight": 2}, # 2
                                  motorcycle={"weight": 3}, # 5
                                  trailer={"weight": 1},
                                  truck={"weight": 1},
                                  bus={"weight": 1})),
    reg=dict(loss_weight=10.0,
             partition_weights=dict(center_xy={"weight": 0.3, "slice": (0, 2)},
                                    center_z={"weight": 0.6, "slice": 2},
                                    size={"weight": 0.5, "slice": (3, 6)},
                                    unit_xvec={"weight": 0.5, "slice": (6, 9)},
                                    abs_xvec={"weight": 1.0, "slice": (9, 12)},
                                    xvec_product={"weight": 1.0, "slice": (12, 14)},
                                    abs_roll_angle={"weight": 1.0, "slice": (14, 16)},
                                    roll_angle_product={"weight": 1.0, "slice": 16},
                                    velo={"weight": 0.5, "slice": (17, 20)})))

bbox_3d_cylinder_weight_scheme = dict(
    cen=dict(loss_weight=2.5,
             fg_weight=0.3,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5.0,
             dual_focal_loss_weight=10, # 10
             channel_weights=dict(any={"weight": 1.0},
                                  traffic_cone={"weight": 1.0})),
    reg=dict(loss_weight=10.0,
             partition_weights=dict(center_xy={"weight": 0.6, "slice": (0, 2)},
                                    center_z={"weight": 0.3, "slice": 2},
                                    size={"weight": 0.6, "slice": (3, 5)},
                                    unit_xvec={"weight": 1.0, "slice": (5, 8)})))

bbox_3d_oriented_cylinder_weight_scheme = dict(
    cen=dict(loss_weight=2.5,
             fg_weight=0.3,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5.0,
             dual_focal_loss_weight=10, # 10
             channel_weights=dict(any={"weight": 1.0},
                                  pedestrian={"weight": 1.0})),
    reg=dict(loss_weight=10.0,
             partition_weights=dict(center_xy={"weight": 1.0, "slice": (0, 2)},
                                    center_z={"weight": 0.3, "slice": 2},
                                    size={"weight": 1.0, "slice": (3, 5)},
                                    unit_xvec={"weight": 1.0, "slice": (5, 8)},
                                    zvec_yaw={"weight": 0.5, "slice": (8, 10)},
                                    velo={"weight": 0.5, "slice": (10, 13)})))

bbox_3d_rect_cuboid_weight_scheme = dict(
    cen=dict(loss_weight=2.5,
             fg_weight=0.3,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5.0,
             dual_focal_loss_weight=10, # 10
             channel_weights=dict(any={"weight": 1.0},
                                  barrier={"weight": 1.0})),
    reg=dict(loss_weight=10.0,
             partition_weights=dict(center_xy={"weight": 0.3, "slice": (0, 2)},
                                    center_z={"weight": 0.3, "slice": 2},
                                    size={"weight": 0.5, "slice": (3, 6)},
                                    abs_xvec={"weight": 0.3, "slice": (6, 9)},
                                    xvec_product={"weight": 1.0, "slice": (9, 11)},
                                    abs_roll_angle={"weight": 1.0, "slice": (11, 13)},
                                    roll_angle_product={"weight": 1.0, "slice": 13})))


loss_cfg = dict(
    bbox_3d=dict(
        type='PlanarLoss',
        seg_iou_method='log',
        loss_name_prefix='bbox_3d',
        weight_scheme=bbox_3d_weight_scheme),
    bbox_3d_cylinder=dict(
        type='PlanarLoss',
        seg_iou_method='log',
        loss_name_prefix='bbox_3d_cylinder',
        weight_scheme=bbox_3d_cylinder_weight_scheme),
    bbox_3d_oriented_cylinder=dict(
        type='PlanarLoss',
        seg_iou_method='log',
        loss_name_prefix='bbox_3d_oriented_cylinder',
        weight_scheme=bbox_3d_oriented_cylinder_weight_scheme),
    bbox_3d_rect_cuboid=dict(
        type='PlanarLoss',
        seg_iou_method='log',
        loss_name_prefix='bbox_3d_rect_cuboid',
        weight_scheme=bbox_3d_rect_cuboid_weight_scheme),
)

# metric configs

# integrated model config
model = dict(
    type='FastBEVModel',
    camera_groups=camera_groups,
    backbones=backbones,
    neck=neck,
    neck_fuse=neck_fuse,
    neck_3d=neck_3d,
    spatial_transform=spatial_transform,
    head_bbox_3d=dict(
        type='FreeAnchor3DHead',
        is_transpose=True,
        num_classes=len(class_mapping),
        in_channels=256,
        feat_channels=256,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            # scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        use_scale_nms=False,
        use_tta=False,
        # Normal-NMS
        nms_across_levels=False,
        use_rotate_nms=True,
        nms_thr=0.2,
        # Scale-NMS
        nms_type_list=[
            'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'circle'],
        nms_thr_list=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2],
        nms_radius_thr_list=[4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1],
        nms_rescale_factor=[1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0],
    ),
    loss_cfg=loss_cfg,
    debug_mode=debug_mode,
)

## log_processor
log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(
    timer=dict(type='GroupIterTimerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

custom_hooks = [
    dict(type="DumpDetectionAsNuscenesJsonHook",
         det_anno_transformable_keys=["bbox_3d", "bbox_3d_rect_cuboid", "bbox_3d_cylinder", "bbox_3d_oriented_cylinder"],
         voxel_shape=voxel_shape,
         voxel_range=voxel_range,
         pre_conf_thresh=0.3,
         nms_ratio=1.0,
         area_score_thresh=0.5),
]

## runner loop configs
train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=50, val_interval=-1)
val_cfg = dict(type="GroupBatchValLoop")
test_cfg = dict(type="GroupBatchInferLoop")

## evaluator and metrics
val_evaluator = [
    dict(type="PlanarSegIou"),
    # dict(
    #     type="PlanarBbox3DAveragePrecision",
    #     transformable_name="bbox_3d" ,
    #     tensor_smith_cfg=val_dataset['transformables']['bbox_3d']['tensor_smith'],
    #     dictionary={"classes": ['truck' ,'motorcycle' ,'car' ,'construction' ,'bicycle']},
    #     max_conf_as_pred_class=True,
    # )
]

test_evaluator = [dict(type="PlanarSegIou")]

## optimizer configs
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0004,
        # momentum=0.9,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}
    ),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)

## scheduler configs
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, end_factor=1, by_epoch=False, begin=0, end=1000), # warmup
    dict(type='PolyLR', by_epoch=False, begin=1000, eta_min=0, power=1.0)     # main LR Scheduler
    # dict(type='PolyLR', by_epoch=False, begin=0, eta_min=0, power=1.0)     # main LR Scheduler
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")])

import datetime
today = datetime.datetime.now().strftime("%m%d")

# load_from = "./ckpts/3scenes_singleframe_epoch_50.pth"
# load_from = "./ckpts/single_frame_nusc_1118_epoch_200.pth"
# load_from = "./ckpts/cascade_mask_rcnn_r50_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5400_segm_mAP_0.4300.pth"
load_from = "./ckpts/fastbev_single_frame_nusc_r50_0228_20250228_094338_epoch_50.pth"
# load_from = "./work_dirs/fastbev_single_frame_nusc_r50_0227/epoch_28.pth"
# work_dir = './work_dirs/fastray_planar_single_frame_1104'
# work_dir = './work_dirs/fastray_planar_single_frame_1105_infer'
# work_dir = './work_dirs/fastray_planar_single_frame_1106_sampled'
# work_dir = './work_dirs/fastray_planar_single_frame_1106_sampled_infer'
# work_dir = './work_dirs/fastray_planar_single_frame_1107'
work_dir = f'./work_dirs/{experiment_name}_{today}'

resume = False
