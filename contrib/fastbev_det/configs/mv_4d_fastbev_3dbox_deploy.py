__base__ = '../../configs/default_runtime.py'
default_scope = "prefusion"
custom_imports = dict(
    imports=['prefusion', 'contrib.fastbev_det'],
    allow_failed_imports=False
)

backend_args = None

IMG_KEYS = [
        'VCAMERA_FISHEYE_FRONT', 'VCAMERA_PERSPECTIVE_FRONT_LEFT', 'VCAMERA_PERSPECTIVE_BACK_LEFT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_PERSPECTIVE_BACK', 'VCAMERA_FISHEYE_BACK', 
        'VCAMERA_PERSPECTIVE_FRONT_RIGHT', 'VCAMERA_PERSPECTIVE_BACK_RIGHT', 'VCAMERA_FISHEYE_RIGHT', 'VCAMERA_PERSPECTIVE_FRONT'
        ]
data_root = "data/mv_4d_data/"
data_root = "data/146_data/"
W, H = 120, 240
bev_front = 180 
bev_left = 60
voxel_size = [0.2, 0.2, 0.5]
downsample_factor=4

img_scale = 2
fish_img_size = [256 * img_scale, 160 * img_scale]
perspective_img_size = [256 * img_scale, 192 * img_scale]
front_perspective_img_size = [768, 384]
batch_size = 1
group_size = 1
bev_range = [-12, 36, -12, 12, -0.5, 2.5]

voxel_feature_config = dict(
    voxel_shape=[6, H, W],  # Z, X, Y in ego system
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

base_resolutions = dict(
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
    )
resolutions = {
    f"{cam_id}_{x-1}":base_resolutions[cam_id] for cam_id in base_resolutions for x in range(0,3)
}
resolutions.update(base_resolutions)

train_pipeline = [
    dict(type='RenderIntrinsic',
        resolutions=resolutions,
        scope='frame'
    ),
    dict(type='RandomRenderExtrinsic',
        prob=0., 
        angles=[0,0,0],
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
        )
]

val_pipeline = [
    dict(type='RenderIntrinsic',
        resolutions=resolutions,
        scope='frame'
    ),
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
        )
]

CLASSES = [
    'class.vehicle.passenger_car', 'class.vehicle.bus', 'class.vehicle.truck', 'class.vehicle.ambulance', 'class.vehicle.fire_engine', 'class.cycle.tricycle', \
    'class.cycle.motorcycle', 'class.cycle.bicycle', 'class.wheeled_push_device.cleaning_cart', 'class.wheeled_push_device.stroller', 'class.wheeled_push_device.shopping_cart', \
    'class.wheeled_push_device.scooter', 'class.animal.animal', 'class.traffic_facility.box', 'class.traffic_facility.cone',  'class.traffic_facility.soft_barrier',\
    'class.traffic_facility.hard_barrier', 'class.traffic_facility.speed_bump', 'class.traffic_facility.gate_barrier', 'class.sign.traffic_sign', 'class.parking.indoor_column',\
    'class.parking.parking_guide', 'class.parking.charging_infra', 'class.parking.parking_lock', 'class.parking.wheel_stopper',\
    
    'class.road_marker.arrow', 'class.parking.access_aisle', 'class.road_marker.text', 'class.parking.text_icon',\
    
    'class.pedestiran.pedestiran',\
    
    'class.traffic_facility.bollard'
]

Bbox3d = dict(
    classes=[
    'class.vehicle.passenger_car', 'class.vehicle.bus', 'class.vehicle.truck', 'class.vehicle.ambulance', 'class.vehicle.fire_engine', 'class.cycle.tricycle', \
    'class.cycle.motorcycle', 'class.cycle.bicycle', 'class.wheeled_push_device.cleaning_cart', 'class.wheeled_push_device.stroller', 'class.wheeled_push_device.shopping_cart', \
    'class.wheeled_push_device.scooter', 'class.animal.animal', 'class.traffic_facility.box', 'class.traffic_facility.cone',  'class.traffic_facility.soft_barrier',\
    'class.traffic_facility.hard_barrier', 'class.traffic_facility.speed_bump', 'class.traffic_facility.gate_barrier', 'class.sign.traffic_sign', 'class.parking.indoor_column',\
    'class.parking.parking_guide', 'class.parking.charging_infra', 'class.parking.parking_lock', 'class.parking.wheel_stopper'], 
    attrs=[['attr.vehicle.is_door_open', 'attr.vehicle.is_trunk_open'], [], [], []]
)

BboxBev = dict(classes=['class.road_marker.arrow', 'class.parking.access_aisle', 'class.road_marker.text', 'class.parking.text_icon'], attrs=[[], [], [], []])

Cylinder3D = dict(classes=['class.pedestiran.pedestiran'], attrs=[[]])

Square3D = dict(classes=['class.traffic_facility.bollard'], attrs=[[]])

transformable_name = ['camera_images','camera_depths', 'bbox_3d', 'bbox_bev', 'cylinder3d', 'square_3d']

transformables={
    "camera_images": dict(type="CameraImageSet"),
    "camera_depths": dict(type="CameraDepthSet"),
    "bbox_3d": dict(type="Bbox3D", dictionary=Bbox3d),
    "bbox_bev": dict(type="Bbox3D", dictionary=BboxBev),
    "cylinder3d": dict(type="Bbox3D", dictionary=Cylinder3D),
    "square_3d": dict(type="Bbox3D", dictionary=Square3D)
    }

dictionary=dict(
        bbox_3d=Bbox3d,
        bbox_bev=BboxBev,
        cylinder_3d=Cylinder3D,
        square_3d=Square3D
    )

train_dataloader = dict(
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='collate_dict'),
    pin_memory=False,
    dataset=dict(
        type='GroupBatchDataset',
        name="mv_4d",
        data_root=data_root,
        info_path=data_root + 'mv_4d_infos_val.pkl',
        transformables=transformables,
        transforms=train_pipeline,
        phase='train',
        batch_size=batch_size, 
        possible_group_sizes=[1],
        possible_frame_intervals=[1]
        ),
    )

val_dataloader = dict(
    num_workers=6,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='collate_dict'),
    pin_memory=False,
    dataset=dict(
        type='GroupBatchDataset',
        name="mv_4d",
        data_root=data_root,
        info_path=data_root + 'mv_4d_infos_val.pkl',
        transformables=transformables,
        transforms=val_pipeline,
        phase='val',
        batch_size=batch_size, 
        possible_group_sizes=[1],
        possible_frame_intervals=[1]
        ),
    )

model_train_cfg = dict(
    available_elements = ['heatmap', 'anno_boxes', 'gridzs', 'class_maps'],
    to_mv_coord = [36, 12, 3],
    bev_range = [-36, -12, -0.5, 12, 12, 2.5],
    grid_size=[120, 240, 1],
    voxel_size=[0.2, 0.2, 0.5],
    out_size_factor=1,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
    is_train_depth=True
)

model_test_cfg = dict(
    post_center_limit_range=[-100, -100, -10, 100, 100, 10],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 0.5],
    nms_type='rotate',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

img_backbone_conf=dict(
        type='VoVNet',
        # model_type="vovnet57",
        # out_indices=[4, 8],
        model_type="vovnet39",
        out_indices=[2, 3, 4, 5],
        base_channels=32,
        # init_cfg=dict(type='Pretrained', checkpoint="./work_dirs/backbone_checkpoint/vovnet57_match.pth")
    )
img_neck_conf=dict(
    type='SECONDFPN',
    in_channels=[128, 128, 128, 256],
    upsample_strides=[1, 1, 1, 2],
    out_channels=[64, 64, 64, 64],
)

model = dict(
    type='FastBEV_Det_DP',
    data_preprocessor=dict(
        type='GroupDataPreprocess',
        mean=[128, 128, 128],
        std=[255, 255, 255],
        IMG_KEYS=IMG_KEYS, 
        label_type=transformables,
        predict_elements=['heatmap', 'anno_boxes', 'gridzs', 'class_maps'],
        batch_size=batch_size,
        group_size=group_size,
        label_start_idx=2, # process labels info start index of collection_info_type
    ),
    backbone_conf=dict(
        type='FastRay_DP_v2',
        x_bound=[36, -12, 0.2],  # BEV grids bounds and size (m)
        y_bound=[12, -12, 0.2],  # BEV grids bounds and size (m)
        z_bound=[-0.5, 2.5, 0.5],  # BEV grids bounds and size (m)
        # final_dim=[fish_img_size[1], fish_img_size[0]],  # img size for model input (pix)
        output_channels=80,  # BEV feature channels
        downsample_factor=downsample_factor,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)  # noqa
        fish_img_backbone_neck_conf=dict(type='ImgBackboneNeck',
                                         img_backbone_conf=img_backbone_conf,
                                         img_neck_conf=img_neck_conf),
        pv_img_backbone_neck_conf=dict(type='ImgBackboneNeck',
                                         img_backbone_conf=img_backbone_conf,
                                         img_neck_conf=img_neck_conf),
        front_img_backbone_neck_conf=dict(type='ImgBackboneNeck',
                                         img_backbone_conf=img_backbone_conf,
                                         img_neck_conf=img_neck_conf),                                                                                 
        depth_net_fish_conf=dict(type='DepthNet', 
                            in_channels=256, 
                            mid_channels=256, 
                            context_channels=80, 
                            d_bound=[0.1, 5.1, 0.2],  # Categorical Depth bounds and division (m)
                            ),
        depth_net_pv_conf=dict(type='DepthNet', 
                            in_channels=256, 
                            mid_channels=256, 
                            context_channels=80, 
                            d_bound=[0.1, 12.1, 0.2],   # Categorical Depth bounds and division (m)
                            ),
        depth_net_front_conf=dict(type='DepthNet', 
                            in_channels=256, 
                            mid_channels=256, 
                            context_channels=80, 
                            d_bound=[0.1, 36.1, 0.2],  # Categorical Depth bounds and division (m)
                            ),
        bev_feature_reducer_conf=dict(type='BEV_Feat_Reducer', in_channels=(256+80)*voxel_feature_config['voxel_shape'][0]),
        voxel_shape=voxel_feature_config['voxel_shape'] + [1]
        # depth_reducer_conf=dict(type='DepthReducer', img_channels=80, mid_channels=80),
        # horiconv_conf=dict(type='HoriConv', in_channels=80, mid_channels=128, out_channels=80),
    ),
    # mono_depth = dict(type='Mono_Depth', 
    #                   fish_img_size=fish_img_size, 
    #                   pv_img_size=perspective_img_size, 
    #                   front_img_size=front_perspective_img_size, 
    #                   downsample_factor=downsample_factor, 
    #                   batch_size=batch_size, 
    #                   avg_reprojection=True,
    #                   disparity_smoothness=0.001,
    #                   fish_unproject_cfg=dict(type='Fish_BackprojectDepth', 
    #                                           batch_size=batch_size, 
    #                                           height=fish_img_size[1], 
    #                                           width=fish_img_size[0],
    #                                           intrinsic=((fish_img_size[0]-1)/2, (fish_img_size[1]-1)/2, fish_img_size[0]/4, fish_img_size[0]/4, 0.1, 0,0,0)),
    #                   fish_project3d_cfg=dict(type='Fish_Project3D', 
    #                                           batch_size=batch_size, 
    #                                           height=fish_img_size[1], 
    #                                           width=fish_img_size[0],
    #                                           intrinsic=((fish_img_size[0]-1)/2, (fish_img_size[1]-1)/2, fish_img_size[0]/4, fish_img_size[0]/4, 0.1, 0,0,0)),                      
    #                   mono_depth_net_cfg=dict(type='Mono_DepthReducer', img_channels=256, mid_channels=128)
    # ),
    head_conf=dict(
        type='BEVDepthHeadV1',
        bev_backbone_conf=dict(
            type='ResNet',
            in_channels=80,
            depth=18,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=[0, 1, 2],
            norm_eval=False,
            base_channels=160,
        ),
        bev_neck_conf=dict(type='SECONDFPN',
            in_channels=[80, 160, 320, 640],
            upsample_strides=[1, 2, 4, 8],
            out_channels=[64, 64, 64, 64]),
        tasks=[
            dict(label_type='bbox_3d', num_class=len(Bbox3d['classes']), class_names=Bbox3d['classes']),
            dict(label_type='bbox_bev', num_class=len(BboxBev['classes']), class_names=BboxBev['classes']),
            dict(label_type='square_3d', num_class=len(Square3D['classes']), class_names=Square3D['classes']),
            ],
        common_heads=dict(reg=(2, 2),
                          height=(1, 2),
                          dim=(3, 2),
                          rot=(2, 2),
                          vel=(2, 2),
                          fb_z=(2, 2)),
        bbox_coder=dict(
            type='mmdet3d.CenterPointBBoxCoder',
            post_center_range=[-100, -100, -10, 100, 100, 10],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=[0.2, 0.2, 3],
            pc_range=[-36, -12, -0.5, 12, 12,  2.5],
            code_size=9,
            ),
        separate_head=dict(type='mmdet3d.SeparateHead',
                           init_bias=-2.19,
                           final_kernel=3),
        train_cfg=model_train_cfg,
        test_cfg=model_test_cfg,
        in_channels=256,
        loss_cls=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    ),
    is_train_depth=True,
    depth_weight=0.25,
    eval_only=False
)

val_evaluator = [
    dict(
        type='Box3DMetric',  # only for det3d of head BEVDepthHeadV1
        available_range=bev_range,
        available_class=CLASSES,
        available_branch=dictionary,
        collect_device='cpu',
        jsonfile_prefix='./work_dirs/',
    )
]


train_cfg = dict(type='GroupBatchTrainLoop', max_epochs=24, val_interval=1)  # -1 note don't eval
val_cfg = dict(type='GroupBatchValLoop')

test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_cfg = dict(type='GroupBatchInferLoop')

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
find_unused_parameters = True

runner_type = 'GroupBatchRunner'

lr = 0.01  # total lr per gpu lr is lr/n 
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=2)
        }),
    # dtype="float16"  # it works only for arg --amp
    )
param_scheduler = dict(type='MultiStepLR', milestones=[16, 20])

auto_scale_lr = dict(enable=False, batch_size=32)

log_processor = dict(type='GroupAwareLogProcessor')
default_hooks = dict(
    timer=dict(type='GroupIterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='ExperimentWiseCheckpointHook', interval=5))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='MomentumAnnealingEMA',
        momentum=0.0001,
        update_buffers=True,
    ),
    ]

vis_backends = [dict(type='LocalVisBackend')]

load_from = None
resume=False