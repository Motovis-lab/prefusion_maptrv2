default_scope = "prefusion"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray_planar"], 
    allow_failed_imports=False)


## camera and voxel feature configs
feature_downscale = 4
default_camera_feature_config = dict(
    ray_distance_num_channel=64,
    ray_distance_start=0.25,
    ray_distance_step=0.25,
    feature_downscale=feature_downscale)

camera_feature_configs = dict(
    VCAMERA_PERSPECTIVE_FRONT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_FRONT_LEFT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_FRONT_RIGHT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_BACK_LEFT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_BACK_RIGHT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_BACK=default_camera_feature_config,
    VCAMERA_FISHEYE_FRONT=default_camera_feature_config,
    VCAMERA_FISHEYE_LEFT=default_camera_feature_config,
    VCAMERA_FISHEYE_BACK=default_camera_feature_config,
    VCAMERA_FISHEYE_RIGHT=default_camera_feature_config)

voxel_shape = (6, 320, 160)  # Z, X, Y in ego system
voxel_range = ([-0.5, 2.5], [36, -12], [12, -12])

voxel_feature_config = dict(
    voxel_shape=voxel_shape, 
    voxel_range=voxel_range,
    ego_distance_max=40,
    ego_distance_step=5)

## dictionaries and mappings for different types of tasks

mapping_heading_objects = dict(
    # 12 = 11 classes + 1 attr
    class_mapping={
        'passenger_car': ['class.vehicle.passenger_car'],
        'bus': ['class.vehicle.bus', 'class.vehicle.ambulance'],
        'truck': ['class.vehicle.truck', 'class.vehicle.env_protect'],
        'fire_engine': ['class.vehicle.fire_engine'],
        'motorcycle': ['class.cycle.motorcycle'],
        'bicycle': ['class.cycle.bicycle'],
        'tricycle': ['class.cycle.tricycle'],
        'cleaning_cart': ['class.wheeled_push_device.cleaning_cart'],
        'shopping_cart': ['class.wheeled_push_device.shopping_cart'],
        'stroller': ['class.wheeled_push_device.stroller'],
        'scooter': ['class.wheeled_push_device.scooter']
    },
    attr_mapping={
        'cycle_with_rider': ['attr.cycle.is_with_rider.true'],
    }
)

mapping_plane_heading_objects = dict(
    # 10 = 1 class + 9 attrs
    class_mapping={
        'arrow': ['class.road_marker.arrow'],
    },
    attr_mapping={
        'arrow_ahead': ['attr.road_marker.arrow.type.ahead'],
        'arrow_left': ['attr.road_marker.arrow.type.left'],
        'arrow_right': ['attr.road_marker.arrow.type.right'],
        'arrow_uturn': ['attr.road_marker.arrow.type.uturn'],
        'arrow_ahead_left': ['attr.road_marker.arrow.type.ahead_left'],
        'arrow_ahead_right': ['attr.road_marker.arrow.type.ahead_right'],
        'arrow_ahead_left_right': ['attr.road_marker.arrow.type.ahead_left_right'],
        'arrow_left_right': ['attr.road_marker.arrow.type.left_right'],
        'arrow_undefined': ['attr.road_marker.arrow.type.undefined']
    }
)

mapping_no_heading_objects = dict(
    # 6 = 6 classes
    class_mapping={
        'wheel_stopper': ['class.wheeled_push_device.wheel_stopper'],
        'speed_bump': ['class.traffic_facility.speed_bump'],
        'water_filled_barrier': ['class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.water_filled_barrier'],
        'cement_pier': ['class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.cement_isolation_pier'],
        'fire_box': ['class.traffic_facility.box::attr.traffic_facility.box.type.fire_box'],
        'distribution_box': ['class.traffic_facility.box::attr.traffic_facility.box.type.distribution_box'],
    }
)

mapping_square_objects = dict(
    # 4 = 3 classes + 1 attr
    class_mapping={
        'pillar_rectangle': ['class.parking.indoor_column::attr.parking.indoor_column.shape.regtangular'],
        'parking_lock': ['class.parking.parking_lock'],
        'waste_bin': ['class.traffic_facility.box::attr.traffic_facility.box.type.waste_bin']
    },
    attr_mapping={
        'parking_lock_locked': ['attr.parking.parking_lock.is_locked.true'],
    }
)

mapping_cylinder_objects = dict(
    # 9 = 9 classes
    class_mapping={
        'pillar_cylinder': ['class.parking.indoor_column::attr.parking.indoor_column.shape.cylindrical'],
        'cone': ['class.traffic_facility.cone', 
                 'class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.no_parking'],
        'bollard': ['class.traffic_facility.bollard'], 
        'roadblock': ['class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.retractable_roadblock'],
        'stone_ball': ['class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.stone_ball'],
        'crash_barrel': ['class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.crash_barrel'],
        'fire_hydrant': ['class.traffic_facility.box::attr.traffic_facility.box.type.fire_hydrant'],
        'warning_triangle': ['class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.warning_triangle'],
        'charging_infra': ['class.parking.charging_infra'],
    }
)

mapping_oriented_cylinder_objects = dict(
    # 1 = 1 class
    class_mapping={
        'pedestrian': ['class.pedestiran.pedestiran'],
    }
)

dictionary_polylines = dict(
    # 8 = 1 class + 7 attrs
    classes=['class.road_marker.lane_line'],
    attrs=['attr.road_marker.lane_line.style.solid',
           'attr.road_marker.lane_line.style.dashed',
           'attr.road_marker.lane_line.type.regular',
           'attr.road_marker.lane_line.type.wide',
           'attr.road_marker.lane_line.type.stop_line',
           'attr.common.color.single_color.white',
           'attr.common.color.single_color.yellow']
)

dictionary_polygons = dict(
    # 4 = 4 classes
    classes=['class.road_marker.crosswalk',
             'class.road_marker.no_parking_zone',
             'class.road_marker.gore_area',
             'class.parking.access_aisle']
)


## camera configs for model inputs

camera_groups = dict(
    pv_front=['VCAMERA_PERSPECTIVE_FRONT'],
    pv_sides=['VCAMERA_PERSPECTIVE_FRONT_LEFT',
              'VCAMERA_PERSPECTIVE_FRONT_RIGHT',
              'VCAMERA_PERSPECTIVE_BACK_LEFT',
              'VCAMERA_PERSPECTIVE_BACK_RIGHT',
              'VCAMERA_PERSPECTIVE_BACK'],
    fisheyes=['VCAMERA_FISHEYE_FRONT',
              'VCAMERA_FISHEYE_LEFT',
              'VCAMERA_FISHEYE_BACK',
              'VCAMERA_FISHEYE_RIGHT'])

resolution_pv_front = (640, 320)
resolution_pv_sides = (512, 320)
resolution_fisheyes = (512, 320)

camera_resolution_configs = dict(
    VCAMERA_PERSPECTIVE_FRONT=resolution_pv_front,
    VCAMERA_PERSPECTIVE_FRONT_LEFT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_FRONT_RIGHT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_BACK_LEFT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_BACK_RIGHT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_BACK=resolution_pv_sides,
    VCAMERA_FISHEYE_FRONT=resolution_fisheyes,
    VCAMERA_FISHEYE_LEFT=resolution_fisheyes,
    VCAMERA_FISHEYE_BACK=resolution_fisheyes,
    VCAMERA_FISHEYE_RIGHT=resolution_fisheyes)

camera_intrinsic_configs = dict(
    VCAMERA_PERSPECTIVE_FRONT=[319.5, 159.5, 640, 640],
)


debug_mode = False

if debug_mode:
    batch_size = 1
    num_workers = 0
    transforms = [
        dict(type='RenderIntrinsic', 
             resolutions=camera_resolution_configs,
             intrinsics=camera_intrinsic_configs)
    ]
    possible_group_sizes=2,
else:
    batch_size = 4
    num_workers = 2
    transforms = [
        dict(type='RandomRenderExtrinsic'),
        dict(type='RenderIntrinsic', resolutions=camera_resolution_configs, intrinsics=camera_intrinsic_configs),
        dict(type='RandomRotateSpace'),
        dict(type='RandomMirrorSpace'),
        dict(type='RandomImageISP', prob=0.2),
        dict(type='RandomSetIntrinsicParam', prob=0.2, jitter_ratio=0.01),
        dict(type='RandomSetExtrinsicParam', prob=0.2, angle=1, translation=0.02)
    ]
    possible_group_sizes=2,


## GroupBatchDataset configs

# transformables
transformables=dict(
    camera_images=dict(type='CameraImageSet', tensor_smith=dict(type='CameraImageTensor')),
    ego_poses=dict(type='EgoPoseSet'),
    bbox_3d_heading=dict(
        type='Bbox3D', 
        loader=dict(type='AdvancedBbox3DLoader', 
                    class_mapping=mapping_heading_objects['class_mapping'],
                    attr_mapping=mapping_heading_objects['attr_mapping']),
        tensor_smith=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    bbox_3d_plane_heading=dict(
        type='Bbox3D', 
        loader=dict(type='AdvancedBbox3DLoader', 
                    class_mapping=mapping_plane_heading_objects['class_mapping'],
                    attr_mapping=mapping_plane_heading_objects['attr_mapping']),
        tensor_smith=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    bbox_3d_no_heading=dict(
        type='Bbox3D', 
        loader=dict(type='AdvancedBbox3DLoader', 
                    class_mapping=mapping_no_heading_objects['class_mapping'],
                    axis_rearrange_method='longer_edge_as_x'),
        tensor_smith=dict(type='PlanarRectangularCuboid', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    bbox_3d_square=dict(
        type='Bbox3D', 
        loader=dict(type='AdvancedBbox3DLoader', 
                    class_mapping=mapping_square_objects['class_mapping'],
                    attr_mapping=mapping_square_objects['attr_mapping']),
        tensor_smith=dict(type='PlanarSquarePillar', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    bbox_3d_cylinder=dict(
        type='Bbox3D', 
        loader=dict(type='AdvancedBbox3DLoader', class_mapping=mapping_cylinder_objects['class_mapping']),
        tensor_smith=dict(type='PlanarCylinder3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    bbox_3d_oriented_cylinder=dict(
        type='Bbox3D', 
        loader=dict(type='AdvancedBbox3DLoader', class_mapping=mapping_oriented_cylinder_objects['class_mapping']),
        tensor_smith=dict(type='PlanarOrientedCylinder3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    polyline_3d=dict(
        type='Polyline3D', dictionary=dictionary_polylines,
        tensor_smith=dict(type='PlanarPolyline3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    polygon_3d=dict(
        type='Polygon3D', dictionary=dictionary_polygons,
        tensor_smith=dict(type='PlanarPolygon3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    parkingslot_3d=dict(
        type='ParkingSlot3D', dictionary=dict(classes=['class.parking.parking_slot']),
        tensor_smith=dict(type='PlanarParkingSlot3D', voxel_shape=voxel_shape, voxel_range=voxel_range))
)

# datasets
train_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='../MV4D-PARKING',
    info_path='../MV4D-PARKING/mv_4d_infos_train.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode),
    transformables=transformables,
    transforms=transforms,
    phase="train",
    batch_size=batch_size,
    possible_group_sizes=possible_group_sizes,
    possible_frame_intervals=10,
)

val_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='../MV4D-PARKING',
    info_path='../MV4D-PARKING/mv_4d_infos_val.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
    ),
    transformables=transformables,
    transforms=[
        dict(type='RenderIntrinsic', 
             resolutions=camera_resolution_configs,
             intrinsics=camera_intrinsic_configs)
    ],
    phase="train",
    batch_size=batch_size,
    possible_group_sizes=2,
    possible_frame_intervals=10,
)

## dataloader configs
train_dataloader = dict(
    sampler=dict(type='DefaultSampler'),
    num_workers=num_workers,
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset,
    # pin_memory=True  # better for station or server
)

val_dataloader = dict(
    sampler=dict(type='DefaultSampler'),
    num_workers=num_workers,
    collate_fn=dict(type="collate_dict"),
    dataset=val_dataset,
    # pin_memory=True  # better for station or server
)


## model configs
bev_mode = True
# backbones
camera_feat_channels = 128
backbones = dict(
    pv_front=dict(type='VoVNetFPN', out_stride=feature_downscale, out_channels=camera_feat_channels),
    pv_sides=dict(type='VoVNetFPN', out_stride=feature_downscale, out_channels=camera_feat_channels),
    fisheyes=dict(type='VoVNetFPN', out_stride=feature_downscale, out_channels=camera_feat_channels))
# spatial_transform
spatial_transform = dict(
    type='FastRaySpatialTransform',
    voxel_shape=voxel_shape,
    fusion_mode='bilinear_weighted',
    # fusion_mode='weighted',
    bev_mode=bev_mode)
# temporal_transform
temporal_transform = dict(
    type='VoxelTemporalAlign',
    voxel_shape=voxel_shape,
    voxel_range=voxel_range,
    bev_mode=bev_mode,
    interpolation='bilinear')
# voxel fusion
pre_nframes = 1
# voxel_fusion = dict(type='EltwiseAdd')
voxel_fusion = dict(
    type='VoxelConcatFusion',
    in_channels=camera_feat_channels * voxel_shape[0],
    pre_nframes=pre_nframes,
    bev_mode=bev_mode)

# heads
all_bbox_3d_cen_seg_channels = sum([
    2 + 12,  # bbox_3d_heading
    2 + 10,  # bbox_3d_plane_heading
    2 + 6,   # bbox_3d_no_heading
    2 + 4,   # bbox_3d_square
    2 + 9,   # bbox_3d_cylinder
    2 + 1    # bbox_3d_oriented_cylinder
])
all_bbox_3d_reg_channels = sum([
    20,  # bbox_3d_heading
    20,  # bbox_3d_plane_heading
    14,  # bbox_3d_no_heading
    11,  # bbox_3d_square
    8,   # bbox_3d_cylinder
    13   # bbox_3d_oriented_cylinder
])
heads = dict(
    voxel_encoder=dict(type='VoVNetEncoder', 
                       in_channels=camera_feat_channels * voxel_shape[0], 
                       mid_channels=128,
                       out_channels=128,
                       repeat=3),
    bbox_3d=dict(type='PlanarHead',
                 in_channels=128,
                 mid_channels=128,
                 cen_seg_channels=all_bbox_3d_cen_seg_channels,
                 reg_channels=all_bbox_3d_reg_channels),
    polyline_3d=dict(type='PlanarHead',
                     in_channels=128,
                     mid_channels=128,
                     cen_seg_channels=1 + 8 + 2 + 4,
                     reg_channels=7 + 7),
    parkingslot_3d=dict(type='PlanarHead',
                        in_channels=128,
                        mid_channels=128,
                        cen_seg_channels=5,
                        reg_channels=15)
)

# loss configs
bbox_3d_heading_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

bbox_3d_plane_heading_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

bbox_3d_no_heading_weight_scheme = dict(
    cen=dict(loss_weight=0.5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

bbox_3d_square_weight_scheme = dict(
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

polyline_3d_weight_scheme = dict(
    seg=dict(loss_weight=1.0,
             iou_loss_weight=2,
             dual_focal_loss_weight=5),
    reg=dict(loss_weight=1.0))

polygon_3d_weight_scheme = dict(
    seg=dict(loss_weight=1.0,
             iou_loss_weight=2,
             dual_focal_loss_weight=5),
    reg=dict(loss_weight=1.0))

parkingslot_3d_weight_scheme = dict(
    cen=dict(loss_weight=1,
             fg_weight=2,
             bg_weight=0.5),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=1,
             dual_focal_loss_weight=2),
    reg=dict(loss_weight=1.0))

loss_cfg = dict(
    bbox_3d_heading=dict(
        type='PlanarLoss',
        loss_name_prefix='bbox_3d_heading',
        weight_scheme=bbox_3d_heading_weight_scheme),
    bbox_3d_plane_heading=dict(
        type='PlanarLoss',
        loss_name_prefix='bbox_3d_plane_heading',
        weight_scheme=bbox_3d_plane_heading_weight_scheme),
    bbox_3d_no_heading=dict(
        type='PlanarLoss',
        loss_name_prefix='bbox_3d_no_heading',
        weight_scheme=bbox_3d_no_heading_weight_scheme),
    bbox_3d_square=dict(
        type='PlanarLoss',
        loss_name_prefix='bbox_3d_square',
        weight_scheme=bbox_3d_square_weight_scheme),
    bbox_3d_cylinder=dict(
        type='PlanarLoss',
        loss_name_prefix='bbox_3d_cylinder',
        weight_scheme=bbox_3d_cylinder_weight_scheme),
    bbox_3d_oriented_cylinder=dict(
        type='PlanarLoss',
        loss_name_prefix='bbox_3d_oriented_cylinder',
        weight_scheme=bbox_3d_oriented_cylinder_weight_scheme),    
    polyline_3d=dict(
        type='PlanarLoss',
        loss_name_prefix='polyline_3d',
        weight_scheme=polyline_3d_weight_scheme),
    polygon_3d=dict(
        type='PlanarLoss',
        loss_name_prefix='polygon_3d',
        weight_scheme=polygon_3d_weight_scheme),
    parkingslot_3d=dict(
        type='PlanarLoss',
        loss_name_prefix='parkingslot_3d',
        weight_scheme=parkingslot_3d_weight_scheme)
)

# integrated model config
model = dict(
    type='ParkingFastRayPlanarMultiFrameModel',
    camera_groups=camera_groups,
    backbones=backbones,
    spatial_transform=spatial_transform,
    temporal_transform=temporal_transform,
    voxel_fusion=voxel_fusion,
    heads=heads,
    loss_cfg=loss_cfg,
    debug_mode=debug_mode,
    pre_nframes=pre_nframes,
)

## log_processor
log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(timer=dict(type='GroupIterTimerHook'))

## runner loop configs
train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=50, val_interval=-1)
val_cfg = dict(type="GroupBatchValLoop")

## evaluator and metrics
val_evaluator = [dict(type="PlanarSegIou"),]

## optimizer configs
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=dict(type='SGD', 
                lr=0.001, 
                momentum=0.9,
                weight_decay=0.0001)
)

## scheduler configs
param_scheduler = dict(type='MultiStepLR', milestones=[24, 36, 48])


env_cfg = dict(
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_1120"
load_from = "./work_dirs/fastray_planar_single_frame_1119_debug/epoch_3.pth"

resume = False