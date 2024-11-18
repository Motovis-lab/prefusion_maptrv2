default_scope = "prefusion"
experiment_name = "fastray_planar_single_frame_nusc"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray_planar"],
    allow_failed_imports=False)


## camera and voxel feature configs

default_camera_feature_config = dict(
    ray_distance_num_channel=64,
    ray_distance_start=0.25,
    ray_distance_step=0.25,
    feature_downscale=8)

camera_feature_configs = dict(
    CAM_FRONT=default_camera_feature_config,
    CAM_FRONT_RIGHT=default_camera_feature_config,
    CAM_BACK_RIGHT=default_camera_feature_config,
    CAM_BACK=default_camera_feature_config,
    CAM_BACK_LEFT=default_camera_feature_config,
    CAM_FRONT_LEFT=default_camera_feature_config
)

voxel_shape = (8, 256, 256)  # Z, X, Y in ego system
voxel_range = ([-3, 5], [50, -50], [50, -50])
# voxel_range = ([-0.5, 2.5], [30, -12], [12, -12])

voxel_feature_config = dict(
    voxel_shape=voxel_shape, 
    voxel_range=voxel_range,
    ego_distance_max=75,  # 50 * sqrt(2)
    ego_distance_step=5)

## dictionaries and mappings for different types of tasks


## camera configs for model inputs

camera_groups = dict(
    pv_sides=[
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ])

# 1600 x 900, 1408 x 512, 1056 x 384, 704 x 256
resolution_pv = (704, 256)

camera_resolution_configs=dict(
    CAM_FRONT=resolution_pv,
    CAM_FRONT_RIGHT=resolution_pv,
    CAM_BACK_RIGHT=resolution_pv,
    CAM_BACK=resolution_pv,
    CAM_BACK_LEFT=resolution_pv,
    CAM_FRONT_LEFT=resolution_pv)

# camera_intrinsic_configs is calculated by the following code snippet
# H, W = 900, 1600
# new_H, new_W = 256, 704
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
camera_intrinsic_configs = dict(
    CAM_FRONT=[359.157, 76.263, 557.224, 557.224],
    CAM_FRONT_RIGHT=[355.506, 77.947, 554.773, 554.773],
    CAM_BACK_RIGHT=[355.191, 80.526, 554.186, 554.186],
    CAM_BACK=[364.857, 71.983, 356.057, 356.057],
    CAM_BACK_LEFT=[348.530, 76.821, 552.966, 552.966],
    CAM_FRONT_LEFT=[363.711, 71.091, 559.943, 559.943],
)

debug_mode = False

if debug_mode:
    batch_size = 1
    num_workers = 0
    persistent_workers = False
    transforms = [
        dict(type='RenderIntrinsic', 
             resolutions=camera_resolution_configs,
             intrinsics=camera_intrinsic_configs)
    ]
else:
    batch_size = 12
    num_workers = 4
    persistent_workers = True
    transforms = [
        dict(type='RandomRenderExtrinsic'),
        dict(type='RenderIntrinsic', resolutions=camera_resolution_configs, intrinsics=camera_intrinsic_configs),
        dict(type='RandomRotateSpace'),
        dict(type='RandomMirrorSpace'),
        dict(type='RandomImageISP', prob=0.2),
        dict(type='RandomSetIntrinsicParam', prob=0.2, jitter_ratio=0.01),
        dict(type='RandomSetExtrinsicParam', prob=0.2, angle=1, translation=0.02)
    ]

## Transformables
transformables = dict(
    camera_images=dict(
        type='CameraImageSet', 
        loader=dict(type="NuscenesCameraImageSetLoader"),
        tensor_smith=dict(type='CameraImageTensor'),
    ),
    ego_poses=dict(type='EgoPoseSet'),
    bbox_3d=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=dict(
                bicycle=["vehicle.bicycle"],
                car=["vehicle.car"],
                construction_vehicle=["vehicle.construction"],
                motorcycle=["vehicle.motorcycle"],
                trailer=["vehicle.trailer"],
                truck=["vehicle.truck"],
                bus=["vehicle.bus.bendy", "vehicle.bus.rigid"],
            ),
        ),
        tensor_smith=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
    ),
    bbox_3d_cylinder=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=dict(
                traffic_cone=["movable_object.trafficcone"]
            ),
        ),
        tensor_smith=dict(type='PlanarCylinder3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
    ),
    bbox_3d_oriented_cylinder=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=dict(
                pedestrian=["human.pedestrian.adult" ,"human.pedestrian.child" ,"human.pedestrian.construction_worker" ,"human.pedestrian.police_officer"],
            ),
        ),
        tensor_smith=dict(type='PlanarOrientedCylinder3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
    ),
    bbox_3d_rect_cuboid=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=dict(
                barrier=['movable_object.barrier']
            ),
        ),
        tensor_smith=dict(type='PlanarRectangularCuboid', voxel_shape=voxel_shape, voxel_range=voxel_range)
    ),
)


## GroupBatchDataset configs
train_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='/data/datasets/nuscenes',
    info_path='/data/datasets/nuscenes/nusc_train_info.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode),
    transformables=transformables,
    transforms=transforms,
    phase="train",
    batch_size=batch_size,
    possible_group_sizes=1,
    possible_frame_intervals=1,
)

val_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='/data/datasets/nuscenes',
    info_path='/data/datasets/nuscenes/nusc_val_info.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
    ),
    transformables=transformables,
    transforms=[dict(type='RenderIntrinsic', resolutions=camera_resolution_configs, intrinsics=camera_intrinsic_configs)],
    phase="val",
    batch_size=batch_size,
    possible_group_sizes=1,
    possible_frame_intervals=1,
)

## dataloader configs
train_dataloader = dict(
    sampler=dict(type='DefaultSampler'),
    num_workers=num_workers,
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset,
    persistent_workers=persistent_workers,
    # pin_memory=True  # better for station or server
)

val_dataloader = dict(
    sampler=dict(type='DefaultSampler'),
    num_workers=0,
    collate_fn=dict(type="collate_dict"),
    dataset=val_dataset,
    persistent_workers=False,
    # pin_memory=True  # better for station or server
)


## model configs
bev_mode = True
# backbones
camera_feat_channels = 128
backbones = dict(
    pv_sides=dict(
        type='VoVNetFPN', 
        out_stride=8, 
        out_channels=camera_feat_channels, 
        init_cfg=dict(type="Pretrained", checkpoint="./ckpts/vovnet_seg_pretrain_backbone_epoch_24.pth")
    )
)
# spatial_transform
spatial_transform = dict(
    type='FastRaySpatialTransform',
    voxel_shape=voxel_shape,
    fusion_mode='bilinear_weighted',
    bev_mode=bev_mode)
# heads
heads = dict(
    voxel_encoder=dict(type='VoVNetEncoder', 
                       in_channels=camera_feat_channels * voxel_shape[0], 
                       mid_channels=128,
                       out_channels=128,
                       repeat=3),
    bbox_3d=dict(type='PlanarHead',
                 in_channels=128,
                 mid_channels=128,
                 cen_seg_channels=sum([
                    # cen: 0
                    1,
                    # seg: slice(1, 9)
                    1 + len(train_dataset["transformables"]["bbox_3d"]["loader"]["class_mapping"]), #
                    # cen: 9
                    1,
                    # seg: slice(10, 12)
                    1 + len(train_dataset["transformables"]["bbox_3d_cylinder"]["loader"]["class_mapping"]),
                    # cen: 12
                    1,
                    # seg: slice(13, 15)
                    1 + len(train_dataset["transformables"]["bbox_3d_oriented_cylinder"]["loader"]["class_mapping"]),
                    # cen: 15
                    1,
                    # seg: slice(16, 18)
                    1 + len(train_dataset["transformables"]["bbox_3d_rect_cuboid"]["loader"]["class_mapping"]),
                 ]),
                 reg_channels=20 + 8 + 13 + 14),
)
# loss configs
bbox_3d_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

bbox_3d_cylinder_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

bbox_3d_oriented_cylinder_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

bbox_3d_rect_cuboid_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

loss_cfg = dict(
    bbox_3d=dict(
        type='PlanarLoss',
        seg_iou_method='linear',
        loss_name_prefix='bbox_3d',
        weight_scheme=bbox_3d_weight_scheme),
    bbox_3d_cylinder=dict(
        type='PlanarLoss',
        seg_iou_method='linear',
        loss_name_prefix='bbox_3d_cylinder',
        weight_scheme=bbox_3d_cylinder_weight_scheme),
    bbox_3d_oriented_cylinder=dict(
        type='PlanarLoss',
        seg_iou_method='linear',
        loss_name_prefix='bbox_3d_oriented_cylinder',
        weight_scheme=bbox_3d_oriented_cylinder_weight_scheme),
    bbox_3d_rect_cuboid=dict(
        type='PlanarLoss',
        seg_iou_method='linear',
        loss_name_prefix='bbox_3d_rect_cuboid',
        weight_scheme=bbox_3d_rect_cuboid_weight_scheme),
)

# metric configs

# integrated model config
model = dict(
    type='NuscenesFastRayPlanarSingleFrameModel',
    camera_groups=camera_groups,
    backbones=backbones,
    spatial_transform=spatial_transform,
    heads=heads,
    loss_cfg=loss_cfg,
    debug_mode=debug_mode,
)

## log_processor
log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(timer=dict(type='GroupIterTimerHook'))

## runner loop configs
train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=12, val_interval=-1)
val_cfg = dict(type="GroupBatchValLoop")

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

## optimizer configs
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=dict(type='SGD', 
                lr=0.01, 
                momentum=0.9,
                weight_decay=0.0001),
    # dtype='bfloat16'
)

## scheduler configs
param_scheduler = dict(type='MultiStepLR', milestones=[5, 8, 10])


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")])

import datetime
today = datetime.datetime.now().strftime("%m%d")

# load_from = "./ckpts/3scenes_singleframe_epoch_50.pth"
# load_from = "./work_dirs/fastray_planar_single_frame_nusc_1111/epoch_99.pth"
# load_from = "./work_dirs/fastray_planar_single_frame_1106_sampled/epoch_50.pth"
# load_from = "./work_dirs/fastray_planar_single_frame_1104/epoch_50.pth"
# work_dir = './work_dirs/fastray_planar_single_frame_1104'
# work_dir = './work_dirs/fastray_planar_single_frame_1105_infer'
# work_dir = './work_dirs/fastray_planar_single_frame_1106_sampled'
# work_dir = './work_dirs/fastray_planar_single_frame_1106_sampled_infer'
# work_dir = './work_dirs/fastray_planar_single_frame_1107'
work_dir = f'./work_dirs/{experiment_name}_{today}'

resume = False
