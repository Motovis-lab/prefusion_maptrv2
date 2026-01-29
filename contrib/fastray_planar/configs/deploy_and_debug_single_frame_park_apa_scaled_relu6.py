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
voxel_range = ([-1, 3], [12, -12], [9, -9])  # tensor dimension corresponed physical range, ([Z_min, Z_max], [X_min, X_max], [Y_min, Y_max])

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


dictionary_dict = dict(
    bbox_3d_heading=dict(classes=list(mapping_heading_objects['class_mapping'].keys())),
    bbox_3d_plane_heading=dict(classes=list(mapping_plane_heading_objects['class_mapping'].keys())),
    bbox_3d_no_heading=dict(classes=list(mapping_no_heading_objects['class_mapping'].keys())),
    bbox_3d_square=dict(classes=list(mapping_square_objects['class_mapping'].keys())),
    bbox_3d_cylinder=dict(classes=list(mapping_cylinder_objects['class_mapping'].keys())),
    bbox_3d_oriented_cylinder=dict(classes=list(mapping_oriented_cylinder_objects['class_mapping'].keys())),
    polyline_3d=dictionary_polylines,
    polygon_3d=dictionary_polygons
)

## camera configs for model inputs

# fisheye_camera_mapping = dict(
#     VCAMERA_FISHEYE_FRONT='VCAMERA_FISHEYE_FRONT',
#     VCAMERA_FISHEYE_LEFT='VCAMERA_FISHEYE_LEFT',
#     VCAMERA_FISHEYE_BACK='VCAMERA_FISHEYE_BACK',
#     VCAMERA_FISHEYE_RIGHT='VCAMERA_FISHEYE_RIGHT' 
# )

fisheye_camera_mapping = dict(
    VCAMERA_FISHEYE_FRONT='camera8',
    VCAMERA_FISHEYE_LEFT='camera5',
    VCAMERA_FISHEYE_BACK='camera1',
    VCAMERA_FISHEYE_RIGHT='camera11' 
)

fisheye_resolution = (640, 384)

virtual_camera_settings = dict(
    VCAMERA_FISHEYE_FRONT=dict(cam_type='FisheyeCamera', resolution=fisheye_resolution, euler_angles=[-120, 0, -90], intrinsic='auto'),
    VCAMERA_FISHEYE_LEFT=dict(cam_type='FisheyeCamera', resolution=fisheye_resolution, euler_angles=[-135, 0, 0], intrinsic='auto'),
    VCAMERA_FISHEYE_BACK=dict(cam_type='FisheyeCamera', resolution=fisheye_resolution, euler_angles=[-120, 0, 90], intrinsic='auto'),
    VCAMERA_FISHEYE_RIGHT=dict(cam_type='FisheyeCamera', resolution=fisheye_resolution, euler_angles=[-135, 0, 180], intrinsic='auto'),
)

virtual_camera_transform = dict(type='RenderVirtualCamera', camera_settings=virtual_camera_settings)

transforms = [virtual_camera_transform]

tensor_smith_dict = dict(
    camera_images=dict(type='CameraImageTensor'),
    bbox_3d_heading=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    bbox_3d_plane_heading=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    bbox_3d_no_heading=dict(type='PlanarRectangularCuboid', voxel_shape=voxel_shape, voxel_range=voxel_range),
    bbox_3d_square=dict(type='PlanarSquarePillar', voxel_shape=voxel_shape, voxel_range=voxel_range),
    bbox_3d_cylinder=dict(type='PlanarCylinder3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    bbox_3d_oriented_cylinder=dict(type='PlanarOrientedCylinder3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    polyline_3d=dict(type='PlanarPolyline3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    polygon_3d=dict(type='PlanarPolygon3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    parkingslot_3d=dict(type='PlanarParkingSlot3D', voxel_shape=voxel_shape, voxel_range=voxel_range),
    occ_sdf_bev=dict(type='PlanarOccSdfBev', voxel_shape=voxel_shape, voxel_range=voxel_range)
)


## GroupBatchDataset configs

# transformables
transformables=dict(
    camera_images=dict(
        type='CameraImageSet', 
        loader=dict(type='CameraImageSetLoader', camera_mapping=fisheye_camera_mapping),
        tensor_smith=tensor_smith_dict['camera_images'])
)

# datasets
test_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    # data_root='/data/datasets/MV4D_12V3L',
    # info_path='/data/datasets/MV4D_12V3L/mv_4d_infos_20231029_195612.pkl',
    data_root='../MV4D-PARKING',
    info_path='../MV4D-PARKING/mv_4d_infos_20231031_135230.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        bilinear_interpolation=False,
        debug_mode=False),
    transformables=transformables,
    transforms=transforms,
    group_sampler=dict(type="SequentialSceneFrameGroupSampler",
                       phase="test_scene_by_scene"),
    batch_size=1,
)

## dataloader configs

test_dataloader = dict(
    # sampler=dict(type='DefaultSampler', shuffle=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
    num_workers=0,
    collate_fn=dict(type="collate_dict"),
    dataset=test_dataset,
    pin_memory=True  # better for station or server
)


## model configs
bev_mode = True
relu6 = True
# backbones
camera_feat_channels = 80
backbone = dict(type='VoVNetSlimFPN', out_channels=camera_feat_channels, relu6=relu6)
# spatial_transform
spatial_transform = dict(
    type='FastRaySpatialTransform',
    voxel_shape=voxel_shape,
    fusion_mode='weighted',
    bev_mode=bev_mode,
    reduce_channels=True,
    in_channels=camera_feat_channels * voxel_shape[0],
    out_channels=128)
## voxel encoder
voxel_encoder = dict(
    type='VoxelEncoderFPN', 
    in_channels=128, 
    mid_channels_list=[128, 128, 128],
    out_channels=128,
    repeats=[3, 3, 3],
    relu6=relu6)

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

bbox_3d_heading_reg_scales = [
    0.25, 0.25,  # center_xy
    0.05,   # center_z
    0.1, 0.1, 0.1,  # size
    0.01, 0.01, 0.01,   # unit_xvec
    0.01, 0.01, 0.01, 0.01, 0.01,   # abs_xvec
    0.01, 0.01, 0.01,   # abs_roll_angle
    0.1, 0.1, 0.1,  # velo
]
bbox_3d_plane_heading_reg_scales = [
    0.25, 0.25,  # center_xy
    0.05,   # center_z
    0.1, 0.1, 0.1,  # size
    0.01, 0.01, 0.01,   # unit_xvec
    0.01, 0.01, 0.01, 0.01, 0.01,   # abs_xvec
    0.01, 0.01, 0.01,   # abs_roll_angle
    0.1, 0.1, 0.1,  # velo
]
bbox_3d_no_heading_reg_scales = [
    0.25, 0.25,  # center_xy
    0.05,   # center_z
    0.1, 0.1, 0.1,  # size
    0.01, 0.01, 0.01, 0.01, 0.01,   # abs_xvec
    0.01, 0.01, 0.01,   # abs_roll_angle
]
bbox_3d_square_reg_scales = [
    0.25, 0.25,  # center_xy
    0.05,   # center_z
    0.1, 0.1, 0.1,  # size
    0.01, 0.01, 0.01,   # unit_zvec
    0.01, 0.01,   # yaw_angle
]
bbox_3d_cylinder_reg_scales = [
    0.25, 0.25,  # center_xy
    0.05,   # center_z
    0.1, 0.1,  # size
    0.01, 0.01, 0.01,   # unit_zvec
]
bbox_3d_oriented_cylinder_reg_scales = [
    0.25, 0.25,  # center_xy
    0.05,   # center_z
    0.1, 0.1,  # size
    0.01, 0.01, 0.01,   # unit_zvec
    0.01, 0.01,   # yaw_angle
    0.1, 0.1, 0.1,  # velo
]
bbox_3d_reg_scales = bbox_3d_heading_reg_scales + \
                     bbox_3d_plane_heading_reg_scales + \
                     bbox_3d_no_heading_reg_scales + \
                     bbox_3d_square_reg_scales + \
                     bbox_3d_cylinder_reg_scales + \
                     bbox_3d_oriented_cylinder_reg_scales
assert len(bbox_3d_reg_scales) == all_bbox_3d_reg_channels, f"len(bbox_3d_reg_scales)={len(bbox_3d_reg_scales)} != all_bbox_3d_reg_channels={all_bbox_3d_reg_channels}"

parkingslot_3d_reg_scales = [
    0.5, 0.5, 0.5, 0.5,  # dist
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # dir
    0.5, 0.5, 0.5, 0.5,  # vec
    0.05,  # height
]

heads = dict(
    bbox_3d=dict(type='PlanarHeadSimple',
                 in_channels=128,
                 mid_channels=128,
                 cen_seg_channels=all_bbox_3d_cen_seg_channels,
                 reg_channels=all_bbox_3d_reg_channels,
                 reg_scales=bbox_3d_reg_scales,
                 relu6=relu6),
    polyline_3d=dict(type='PlanarHeadSimple',
                     in_channels=128,
                     mid_channels=64,
                     cen_seg_channels=1 + 8 + 2 + 4,
                     reg_channels=7 + 7,
                     relu6=relu6),
    parkingslot_3d=dict(type='PlanarHeadSimple',
                        in_channels=128,
                        mid_channels=64,
                        cen_seg_channels=5,
                        reg_channels=15,
                        reg_scales=parkingslot_3d_reg_scales,
                        relu6=relu6),
    occ_sdf_bev=dict(type='PlanarHeadSimple',
                     in_channels=128,
                     mid_channels=64,
                     cen_seg_channels=2,
                     reg_channels=2, 
                     relu6=relu6)
)


# integrated model config
model = dict(
    type='ParkingFastRayPlanarSingleFrameModelAPA',
    backbone=backbone,
    spatial_transform=spatial_transform,
    voxel_encoder=voxel_encoder,
    heads=heads,
    debug_mode=False
)

# work_dir = "./work_dirs/deploy_and_debug_single_frame_park_apa_scaled_0211"
# work_dir = "./work_dirs/0_deploy_and_quantize_single_frame_park_apa_scaled_relu6_0224_v11"
# work_dir = "./work_dirs/0_deploy_and_quantize_single_frame_park_apa_scaled_relu6_0227_fixed"
work_dir = "./work_dirs/0_deploy_and_quantize_0310"
## log_processor
log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(timer=dict(type='GroupIterTimerHook'))
custom_hooks = [
    dict(type="DeployAndDebugHookAPA", 
         tensor_smith_dict=tensor_smith_dict, 
         dictionary_dict=dictionary_dict,
         save_dir=work_dir,
         HWC=True),
]


## runner loop configs
test_cfg = dict(type="GroupBatchInferLoop")

## evaluator and metrics
test_evaluator = []


env_cfg = dict(
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# load_from = "./work_dirs/collected_models/vovnet_fpn_pretrain.pth"
# load_from = "./work_dirs/collected_models/apa_nearest_scaled_relu6_epoch_43.pth"
load_from = "./work_dirs/collected_models/apa_nearest_scaled_relu6_epoch_12.pth"

# resume = False