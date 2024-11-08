default_scope = "prefusion"
experiment_name = "fastray_planar_demo"

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

mapping_heading_objects = {
    'passenger_car': [],
    'bus': [],
    'truck': [],
    'ambulance': [],
    'fire_engine': [],
    'env_protect': [],
    'tricycle':[],
    'motorcycle':[],
    'bicycle':[],
    'cleaning_cart':[],
    'shopping_cart':[],
    'stroller':[]    
}

dictionary_heading_objects = dict(
    classes=['passenger_car', 
             'bus', 
             'truck', 
             'ambulance', 
             'fire_engine', 
             'env_protect',
             'tricycle',
             'motorcycle',
             'bicycle',
             'cleaning_cart',
             'shopping_cart',
             'stroller'],
    attrs=['cycle.is_with_rider.true']
)

dictionary_plane_objects = dict(
    classes=[],
    attrs=[]
)

mapping_no_heading_objects = {
    'speed_bump': []
}

dictionary_no_heading_objects = dict(
    classes=['speed_bump'],
    attrs=[]
)

dictionary_square_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_cylinder_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_oriented_cylinder_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_polylines = dict(
    classes=['class.road_marker.lane_line'],
    attrs=['attr.road_marker.lane_line.type.regular',
           'attr.road_marker.lane_line.type.wide',
           'attr.road_marker.lane_line.type.stop_line',
           'attr.common.color.single_color.yellow',
           'attr.common.color.single_color.white',
           'attr.road_marker.lane_line.style.dashed',
           'attr.road_marker.lane_line.style.solid']
)

dictionary_polygons = dict(
    classes=['class.road_marker.no_parking_zone',
             'class.road_marker.crosswalk',
             'class.road_marker.gore_area'],
    attrs=[]
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
# resolution_pv_front = (512, 320)
resolution_pv_sides = (512, 320)
resolution_fisheyes = (512, 320)

camera_resolution_configs=dict(
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
    # VCAMERA_PERSPECTIVE_FRONT='default',
)

debug_mode = True

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
    num_workers = 8
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

## GroupBatchDataset configs
train_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='../MV4D-PARKING',
    info_path='../MV4D-PARKING/mv_4d_infos_20231028_150815.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
    ),
    transformables=dict(
        camera_images=dict(type='CameraImageSet', tensor_smith=dict(type='CameraImageTensor')),
        ego_poses=dict(type='EgoPoseSet'),
        bbox_3d=dict(
            type='Bbox3D', 
            dictionary=dict(classes=['class.vehicle.passenger_car']),
            tensor_smith=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
        polyline_3d=dict(
            type='Polyline3D',
            dictionary=dict(classes=['class.road_marker.lane_line']),
            tensor_smith=dict(type='PlanarPolyline3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
        parkingslot_3d=dict(
            type='ParkingSlot3D',
            dictionary=dict(classes=['class.parking.parking_slot']),
            tensor_smith=dict(type='PlanarParkingSlot3D', voxel_shape=voxel_shape, voxel_range=voxel_range))
    ),
    transforms=transforms,
    phase="train",
    batch_size=batch_size,
    possible_group_sizes=1,
    possible_frame_intervals=1,
)

## dataloader configs
train_dataloader = dict(
    num_workers=num_workers,
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset,
    persistent_workers=persistent_workers,
)

# val_dataloader = train_dataloader


## model configs
bev_mode = True
# backbones
camera_feat_channels = 128
backbones = dict(
    pv_front=dict(type='VoVNetFPN', out_stride=8, out_channels=camera_feat_channels),
    pv_sides=dict(type='VoVNetFPN', out_stride=8, out_channels=camera_feat_channels),
    fisheyes=dict(type='VoVNetFPN', out_stride=8, out_channels=camera_feat_channels))
# spatial_transform
spatial_transform = dict(
    type='FastRaySpatialTransform',
    voxel_shape=voxel_shape,
    fusion_mode='weighted',
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
                 cen_seg_channels=3,
                 reg_channels=20),
    polyline_3d=dict(type='PlanarHead',
                     in_channels=128,
                     mid_channels=128,
                     cen_seg_channels=2,
                     reg_channels=7),
    parkingslot_3d=dict(type='PlanarHead',
                        in_channels=128,
                        mid_channels=128,
                        cen_seg_channels=5,
                        reg_channels=15)
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

polyline_3d_weight_scheme = dict(
    seg=dict(loss_weight=1.0,
             iou_loss_weight=2,
             dual_focal_loss_weight=5),
    reg=dict(loss_weight=1.0))

parkingslot_3d_weight_scheme = dict(
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
        loss_name_prefix='bbox_3d',
        weight_scheme=bbox_3d_weight_scheme),
    polyline_3d=dict(
        type='PlanarLoss',
        loss_name_prefix='polyline_3d',
        weight_scheme=polyline_3d_weight_scheme),
    parkingslot_3d=dict(
        type='PlanarLoss',
        loss_name_prefix='parkingslot_3d',
        weight_scheme=parkingslot_3d_weight_scheme)
)

# metric configs

# integrated model config
model = dict(
    type='FastRayPlanarSingleFrameModel',
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
train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=50, val_interval=-1)

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
param_scheduler = dict(type='MultiStepLR', milestones=[24, 36, 48])

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


load_from = "./work_dirs/fastray_planar_single_frame_1107/epoch_50.pth"
# load_from = "./work_dirs/fastray_planar_single_frame_1106_sampled/epoch_50.pth"
# load_from = "./work_dirs/fastray_planar_single_frame_1104/epoch_50.pth"
# work_dir = './work_dirs/fastray_planar_single_frame_1104'
# work_dir = './work_dirs/fastray_planar_single_frame_1105_infer'
# work_dir = './work_dirs/fastray_planar_single_frame_1106_sampled'
# work_dir = './work_dirs/fastray_planar_single_frame_1106_sampled_infer'
# work_dir = './work_dirs/fastray_planar_single_frame_1107'
work_dir = './work_dirs/fastray_planar_single_frame_1107_infer'

# resume = True