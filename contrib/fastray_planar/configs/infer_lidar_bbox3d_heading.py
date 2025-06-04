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
# voxel_shape = (8, 320, 240)  # Z, X, Y in ego system
voxel_range = ([-1, 3], [12, -12], [9, -9])  # tensor dimension corresponed physical range, ([Z_min, Z_max], [X_min, X_max], [Y_min, Y_max])
lidar_voxel_size = [(voxel_range[1][0] - voxel_range[1][1]) / voxel_shape[1],
                    (voxel_range[2][0] - voxel_range[2][1]) / voxel_shape[2],
                    0.2]
lidar_voxel_shape = [41, voxel_shape[1] * 8, voxel_shape[2] * 8]

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
        'bus': ['class.vehicle.bus'],
        'truck': ['class.vehicle.truck'],
        'ambulance': [ 'class.vehicle.ambulance'],
        'env_protect':['class.vehicle.env_protect'],
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

dictionary_dict = dict(bbox_3d_heading=dict(classes=list(mapping_heading_objects['class_mapping'].keys())))


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

# fisheye_resolution = (640, 384)
fisheye_resolution = (800, 480)

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
    lidar_sweeps=dict(type="PointsToVoxelsTensor", voxel_size=lidar_voxel_size,
                      max_point_per_voxel=10, max_voxels=120000, max_input_points=1200000,
                      point_cloud_range=[voxel_range[1][1], voxel_range[2][1], 
                                         voxel_range[0][0], voxel_range[1][0],
                                         voxel_range[2][0], voxel_range[0][1]]),
    bbox_3d_heading=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
)


## GroupBatchDataset configs

# transformables
transformables=dict(
    lidar_sweeps=dict(
        type="LidarPoints",
        loader=dict(type="LidarSweepsLoader", sweep_info_length=0),
        tensor_smith=dict(type="PointsToVoxelsTensor", voxel_size=lidar_voxel_size,
                          max_point_per_voxel=10, max_voxels=120000,
                          max_input_points=1200000,
                          point_cloud_range=[voxel_range[1][1], voxel_range[2][1], 
                                             voxel_range[0][0], voxel_range[1][0],
                                             voxel_range[2][0], voxel_range[0][1]]),
    ),
    camera_images=dict(
        type='CameraImageSet',
        loader=dict(type='CameraImageSetLoader', camera_mapping=fisheye_camera_mapping),
        tensor_smith=dict(type='CameraImageTensor')),
    bbox_3d_heading=dict(
        type='Bbox3D',
        loader=dict(type='AdvancedBbox3DLoader',
                    class_mapping=mapping_heading_objects['class_mapping'],
                    attr_mapping=mapping_heading_objects['attr_mapping']),
        tensor_smith=dict(type='PlanarBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)),
)

# datasets
test_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='../MV4D-PARKING',
    # info_path='../MV4D-PARKING/planar_lidar_nocamerapose_20230901_152553.pkl',
    info_path='../MV4D-PARKING/mv_4d_infos_20230901_152553_fix.pkl',
    model_feeder=dict(
        type="FastRayLidarPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
        debug_mode=False),
    transformables=transformables,
    transforms=transforms,
    group_sampler=dict(type="SequentialSceneFrameGroupSampler",
                       phase="test_scene_by_scene"),
    batch_size=1,
)

## dataloader configs

test_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    num_workers=0,
    collate_fn=dict(type="collate_dict"),
    dataset=test_dataset,
    pin_memory=True  # better for station or server
)

## model configs
bev_mode = True
# backbones
camera_feat_channels = 80
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
    out_channels=128)
# lidar fusion
lidar_voxel_fusion = dict(
    type='FeatureConcatFusion',
    in_channels=128 + 512,
    out_channels=128,
    bev_mode=True,
    dilation=3)
# voxel encoder
voxel_encoder = dict(
    type='VoxelEncoderFPN',
    in_channels=128,
    mid_channels_list=[128, 128, 128],
    out_channels=128,
    repeats=[3, 3, 3])

# heads
all_bbox_3d_cen_seg_channels = sum([
    2 + 14,  # bbox_3d_heading
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
                 in_channels=128,
                 mid_channels=128,
                 cen_seg_channels=all_bbox_3d_cen_seg_channels,
                 reg_channels=all_bbox_3d_reg_channels),
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

loss_cfg = dict(
    bbox_3d_heading=dict(type='PlanarLoss',
                         seg_iou_method='linear',
                         loss_name_prefix='bbox_3d_heading',
                         weight_scheme=bbox_3d_heading_weight_scheme)
)

# integrated model config
model = dict(
    type='ParkingFastRayPlanarSingleFrameModelAPALidar',
    lidar_voxel_fusion=lidar_voxel_fusion,
    pts_middle_encoder=dict(
        type='mmdet3d.SparseEncoder',
        in_channels=5,
        sparse_shape=lidar_voxel_shape,
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

    backbone=backbone,
    spatial_transform=spatial_transform,
    voxel_encoder=voxel_encoder,
    heads=heads,
    loss_cfg=loss_cfg,
    debug_mode=False
)

## log_processor
work_dir = "./work_dirs/planar_lidar_dumps_0307"
log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(timer=dict(type='GroupIterTimerHook'))
custom_hooks = [
    dict(type="DumpPlanarPredResultsHookAPA", 
         tensor_smith_dict=tensor_smith_dict, 
         dictionary_dict=dictionary_dict,
         save_dir=work_dir),
]

## runner loop configs
test_cfg = dict(type="GroupBatchInferLoop")

## evaluator and metrics
test_evaluator = []


env_cfg = dict(
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

load_from = "./work_dirs/collected_models/lidar_epoch_92.pth"
# load_from = "./ckpts/single_frame_epoch_14.pth"

resume = False