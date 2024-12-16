default_scope = "prefusion"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray_planar"], 
    allow_failed_imports=False)


## camera and voxel feature configs
feature_downscale = 4
default_camera_feature_config = dict(
    ray_distance_num_channel=32,
    ray_distance_start=0.5,
    ray_distance_step=0.5,
    feature_downscale=feature_downscale)

camera_feature_configs = dict(
    VCAMERA_FISHEYE_FRONT=default_camera_feature_config,
    VCAMERA_FISHEYE_LEFT=default_camera_feature_config,
    VCAMERA_FISHEYE_BACK=default_camera_feature_config,
    VCAMERA_FISHEYE_RIGHT=default_camera_feature_config)

voxel_shape = (8, 160, 120)  # Z, X, Y in ego system
voxel_range = ([-1, 3], [12, -12], [9, -9])

voxel_feature_config = dict(
    voxel_shape=voxel_shape, 
    voxel_range=voxel_range,
    ego_distance_max=16,
    ego_distance_step=2)

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
        'wheel_stopper': ['class.parking.wheel_stopper'],
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
        'pedestrian': ['class.pedestrian.pedestrian'],
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

fisheye_cameras=['VCAMERA_FISHEYE_FRONT',
                 'VCAMERA_FISHEYE_LEFT',
                 'VCAMERA_FISHEYE_BACK',
                 'VCAMERA_FISHEYE_RIGHT']

fisheye_resolution = (640, 384)

camera_resolution_configs = dict(
    VCAMERA_FISHEYE_FRONT=fisheye_resolution,
    VCAMERA_FISHEYE_LEFT=fisheye_resolution,
    VCAMERA_FISHEYE_BACK=fisheye_resolution,
    VCAMERA_FISHEYE_RIGHT=fisheye_resolution)


debug_mode = True

if debug_mode:
    batch_size = 1
    num_workers = 1
    transforms = [dict(type='RenderIntrinsic', resolutions=camera_resolution_configs)]
    possible_group_sizes = 2
else:
    batch_size = 4
    num_workers = 4
    transforms = [
        dict(type='RandomRenderExtrinsic'),
        dict(type='RenderIntrinsic', resolutions=camera_resolution_configs),
        dict(type='RandomRotateSpace', angles=(0, 0, 90), prob_inverse_cameras_rotation=0),
        dict(type='RandomMirrorSpace'),
        dict(type='RandomImageISP', prob=0.2),
        dict(type='RandomSetIntrinsicParam', prob=0.2, jitter_ratio=0.01),
        dict(type='RandomSetExtrinsicParam', prob=0.2, angle=1, translation=0.02)
    ]
    possible_group_sizes = 2


## GroupBatchDataset configs

# transformables
transformables=dict(
    camera_images=dict(
        type='CameraImageSet', 
        loader=dict(type='CameraImageSetLoader',
                    selected_cameras=fisheye_cameras),
        tensor_smith=dict(type='CameraImageTensor')),
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
        tensor_smith=dict(type='PlanarPolyline3D', 
                          voxel_shape=voxel_shape, 
                          voxel_range=voxel_range,
                          reverse_group_dist_thresh=5,
                          reverse_link_max_adist=15)),
    polygon_3d=dict(
        type='Polygon3D', dictionary=dictionary_polygons,
        tensor_smith=dict(type='PlanarPolygon3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    parkingslot_3d=dict(
        type='ParkingSlot3D', dictionary=dict(classes=['class.parking.parking_slot']),
        tensor_smith=dict(type='PlanarParkingSlot3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
    # occ_sdf_bev=dict(
    #     type='OccSdfBev',
    #     src_voxel_range=[[-1, 3], [-12.8, 38.4], [25.6, -25.6]],
    #     tensor_smith=dict(type='PlanarOccSdfBev', voxel_shape=voxel_shape, voxel_range=voxel_range)),
)

# datasets
train_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='../MV4D-PARKING',
    info_path='../MV4D-PARKING/mv_4d_infos_train.pkl',
    # info_path='../MV4D-PARKING/mv_4d_infos_val.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode),
    transformables=transformables,
    transforms=transforms,
    group_sampler=dict(type="IndexGroupSampler",
                       phase="train",
                       possible_group_sizes=possible_group_sizes,
                       possible_frame_intervals=10),
    batch_size=batch_size,
    subepoch_manager=dict(type="SubEpochManager",
                          batch_size=batch_size,
                          num_group_batches_per_subepoch=500,
                          drop_last_group_batch=False,
                          drop_last_subepoch=False,
                          verbose=True,
                          debug_mode=True),
)

val_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='../MV4D-PARKING',
    info_path='../MV4D-PARKING/mv_4d_demo_info.pkl',
    # info_path='../MV4D-PARKING/mv_4d_infos_val.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=debug_mode
    ),
    transformables=transformables,
    transforms=[dict(type='RenderIntrinsic', resolutions=camera_resolution_configs)],
    group_sampler=dict(type="IndexGroupSampler",
                       phase="val",
                       possible_group_sizes=10,
                       possible_frame_intervals=10),
    batch_size=batch_size,
)

## dataloader configs
train_dataloader = dict(
    sampler=dict(type='DefaultSampler'),
    num_workers=num_workers,
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset,
    pin_memory=True  # better for station or server
)

val_dataloader = dict(
    sampler=dict(type='DefaultSampler'),
    num_workers=num_workers,
    # num_workers=0,
    collate_fn=dict(type="collate_dict"),
    dataset=val_dataset,
    # pin_memory=True  # better for station or server
)


## model configs
bev_mode = True
# backbones
camera_feat_channels = 64
backbone = dict(type='VoVNetSlimFPN', out_channels=camera_feat_channels)
# spatial_transform
spatial_transform = dict(
    type='FastRaySpatialTransform',
    voxel_shape=voxel_shape,
    fusion_mode='bilinear_weighted',
    # fusion_mode='weighted',
    bev_mode=bev_mode,
    reduce_channels=True,
    in_channels=camera_feat_channels * voxel_shape[0],
    out_channels=64)
# temporal_transform
temporal_transform = dict(
    type='VoxelTemporalAlign',
    voxel_shape=voxel_shape,
    voxel_range=voxel_range,
    bev_mode=bev_mode,
    interpolation='bilinear')
## voxel encoder
voxel_encoder = dict(
    type='VoxelEncoderFPN', 
    in_channels=64, 
    mid_channels_list=[64, 96, 128],
    out_channels=64,
    repeats=[3, 3, 3])
# temporal fusion
pre_nframes = 1
voxel_fusion = dict(
    type='VoxelConcatFusion',
    in_channels=64,
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
    bbox_3d=dict(type='PlanarHeadSimple',
                 in_channels=64,
                 mid_channels=64,
                 cen_seg_channels=all_bbox_3d_cen_seg_channels,
                 reg_channels=all_bbox_3d_reg_channels),
    polyline_3d=dict(type='PlanarHeadSimple',
                     in_channels=64,
                     mid_channels=64,
                     cen_seg_channels=1 + 8 + 2 + 4,
                     reg_channels=7 + 7),
    parkingslot_3d=dict(type='PlanarHeadSimple',
                        in_channels=64,
                        mid_channels=64,
                        cen_seg_channels=5,
                        reg_channels=15),
    # occ_sdf_bev=dict(type='PlanarHeadSimple',
    #                  in_channels=64,
    #                  mid_channels=64,
    #                  cen_seg_channels=3,
    #                  reg_channels=2)
)

# loss configs
bbox_3d_heading_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "center_xy": {"weight": 2, "slice": (0, 2)},
                "center_z": {"weight": 5, "slice": 2},
                "size": {"weight": 5, "slice": (3, 6)},
                "unit_xvec": {"weight": 10, "slice": (6, 9)},
                "abs_xvec": {"weight": 20, "slice": (9, 14)},
                "abs_roll_angle": {"weight": 5, "slice": (14, 17)},
                "velo": {"weight": 5, "slice": (17, 20)},
            })
)

bbox_3d_plane_heading_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "center_xy": {"weight": 2, "slice": (0, 2)},
                "center_z": {"weight": 5, "slice": 2},
                "size": {"weight": 5, "slice": (3, 6)},
                "unit_xvec": {"weight": 10, "slice": (6, 9)},
                "abs_xvec": {"weight": 20, "slice": (9, 14)},
                "abs_roll_angle": {"weight": 5, "slice": (14, 17)},
                "velo": {"weight": 1, "slice": (17, 20)},
            })
)

bbox_3d_no_heading_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "center_xy": {"weight": 2, "slice": (0, 2)},
                "center_z": {"weight": 5, "slice": 2},
                "size": {"weight": 5, "slice": (3, 6)},
                "abs_xvec": {"weight": 20, "slice": (6, 11)},
                "abs_roll_angle": {"weight": 5, "slice": (11, 14)},
            })
)

bbox_3d_square_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "center_xy": {"weight": 2, "slice": (0, 2)},
                "center_z": {"weight": 5, "slice": 2},
                "size": {"weight": 5, "slice": (3, 6)},
                "unit_zvec": {"weight": 20, "slice": (6, 9)},
                "yaw_angle": {"weight": 5, "slice": (9, 11)},
            })
)

bbox_3d_cylinder_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "center_xy": {"weight": 2, "slice": (0, 2)},
                "center_z": {"weight": 5, "slice": 2},
                "size": {"weight": 5, "slice": (3, 5)},
                "unit_zvec": {"weight": 20, "slice": (5, 8)},
            })
)

bbox_3d_oriented_cylinder_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=1.0,
             bg_weight=1),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "center_xy": {"weight": 2, "slice": (0, 2)},
                "center_z": {"weight": 5, "slice": 2},
                "size": {"weight": 5, "slice": (3, 5)},
                "unit_zvec": {"weight": 20, "slice": (5, 8)},
                "yaw_angle": {"weight": 5, "slice": (8, 10)},
                "velo": {"weight": 5, "slice": (10, 13)},
            })
)

polyline_3d_weight_scheme = dict(
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1.0,
             partition_weights={
                "dist": {"weight": 2, "slice": (0, 1)},
                "vec": {"weight": 2, "slice": (1, 3)},
                "dir": {"weight": 20, "slice": (3, 6)},
                "height": {"weight": 5, "slice": (6, 7)},
            })
)

polygon_3d_weight_scheme = dict(
    seg=dict(loss_weight=2,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=0.5,
             partition_weights={
                "dist": {"weight": 2, "slice": (0, 1)},
                "vec": {"weight": 2, "slice": (1, 3)},
                "dir": {"weight": 20, "slice": (3, 6)},
                "height": {"weight": 5, "slice": (6, 7)},
            })
)

parkingslot_3d_weight_scheme = dict(
    cen=dict(loss_weight=5,
             fg_weight=2,
             bg_weight=0.5),
    seg=dict(loss_weight=1.0,
             iou_loss_weight=5,
             dual_focal_loss_weight=10),
    reg=dict(loss_weight=1,
             partition_weights={
                "dist": {"weight": 5, "slice": (0, 4)},
                "dir": {"weight": 20, "slice": (4, 10)},
                "vec": {"weight": 10, "slice": (10, 14)},
                "height": {"weight": 5, "slice": (14, 15)},
            })
)

# occ_sdf_bev_weight_scheme = dict(
#     seg=dict(loss_weight=1.0,
#              iou_loss_weight=5,
#              dual_focal_loss_weight=10),
#     reg=dict(loss_weight=10,
#              partition_weights={
#                 "sdf": {"weight": 1, "slice": (0, 1)},
#                 "height": {"weight": 1, "slice": (1, 2)},
#             })
# )

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
        weight_scheme=parkingslot_3d_weight_scheme),
    # occ_sdf_bev=dict(
    #     type='PlanarLoss',
    #     loss_name_prefix='occ_sdf_bev',
    #     weight_scheme=occ_sdf_bev_weight_scheme),
)

# integrated model config
model = dict(
    type='ParkingFastRayPlanarMultiFrameModelAPA',
    backbone=backbone,
    spatial_transform=spatial_transform,
    temporal_transform=temporal_transform,
    voxel_fusion=voxel_fusion,
    voxel_encoder=voxel_encoder,
    heads=heads,
    loss_cfg=loss_cfg,
    debug_mode=debug_mode,
    pre_nframes=pre_nframes,
    voxel_fusion_before_encoder=True
)

## log_processor
log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(timer=dict(type='GroupIterTimerHook'))

## runner loop configs
train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=20, val_interval=-1)
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
param_scheduler = dict(type='MultiStepLR', milestones=[10, 15, 18])


env_cfg = dict(
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1127"
# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1128_val"
# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1129_val"
# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1129"
# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1130"
# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1201"
# work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1204"
work_dir = "./work_dirs/fastray_planar_multi_frame_cat_park_apa_1212"
# load_from = "./work_dirs/collected_models/vovnet_fpn_pretrain.pth"
# load_from = "./work_dirs/collected_models/apa_epoch_10.pth"
# load_from = "./work_dirs/collected_models/apa_epoch_20_enhanced.pth"
# load_from = "./work_dirs/collected_models/apa_epoch_20_better.pth"
load_from = "./work_dirs/collected_models/apa_epoch_14_tf.pth"

resume = False