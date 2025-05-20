import datetime

experiment_name = "stream_petr_nusc_r50"

_base_ = "../../../configs/default_runtime.py"

custom_imports = dict(imports=["prefusion", "contrib.petr", "mmdet"], allow_failed_imports=False)

backend_args = None

def _calc_grid_size(_range, _voxel_size, n_axis=3):
    return [(_range[n_axis+i] - _range[i]) // _voxel_size[i] for i in range(n_axis)]

batch_size = 2
num_epochs = 500
voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_range = (point_cloud_range[2::3], point_cloud_range[0::3][::-1], point_cloud_range[1::3][::-1])
grid_size = _calc_grid_size(point_cloud_range, voxel_size)

# 1600 x 900, 1408 x 512, 1056 x 384, 704 x 256
# resolution_pv = (880, 320)

# camera_resolution_configs=dict(
#     CAM_FRONT=resolution_pv,
#     CAM_FRONT_RIGHT=resolution_pv,
#     CAM_BACK_RIGHT=resolution_pv,
#     CAM_BACK=resolution_pv,
#     CAM_BACK_LEFT=resolution_pv,
#     CAM_FRONT_LEFT=resolution_pv)

# camera_intrinsic_configs_top_crop = dict(
#     CAM_FRONT=[454.623, 83.492, 689.047, 689.047],
#     CAM_FRONT_RIGHT=[449.784, 73.575, 691.212, 691.212],
#     CAM_BACK_RIGHT=[453.957, 79.401, 687.480, 687.480],
#     CAM_BACK=[471.778, 87.287, 438.290, 438.290],
#     CAM_BACK_LEFT=[456.267, 81.942, 690.242, 690.242],
#     CAM_FRONT_LEFT=[454.983, 73.004, 691.824, 691.824],
# )

# # new_cy = cy_if_no_crop - to_crop / 2
# camera_intrinsic_configs_center_crop = dict(
#     CAM_FRONT=[454.623, 170.992, 689.047, 689.047],
#     CAM_FRONT_RIGHT=[449.784, 161.075, 691.212, 691.212],
#     CAM_BACK_RIGHT=[453.957, 166.901, 687.480, 687.480],
#     CAM_BACK=[471.778, 174.787, 438.290, 438.290],
#     CAM_BACK_LEFT=[456.267, 169.442, 690.242, 690.242],
#     CAM_FRONT_LEFT=[454.983, 160.504, 691.824, 691.824],
# )

# camera_intrinsic_configs = camera_intrinsic_configs_center_crop

class_mapping = dict(
    car=["vehicle.car"],
    truck=["vehicle.truck"],
    construction_vehicle=["vehicle.construction"],
    bus=["vehicle.bus.bendy", "vehicle.bus.rigid"],
    trailer=["vehicle.trailer"],
    barrier=['movable_object.barrier'],
    motorcycle=["vehicle.motorcycle"],
    bicycle=["vehicle.bicycle"],
    pedestrian=["human.pedestrian.adult" ,"human.pedestrian.child" ,"human.pedestrian.construction_worker" ,"human.pedestrian.police_officer"],
    traffic_cone=["movable_object.trafficcone"],
)

transformables = dict(
    sample_token=dict(type='Variable', loader=dict(type="VariableLoader", variable_key="sample_token")),
    camera_images=dict(
        type="CameraImageSet",
        loader=dict(type="NuscenesCameraImageSetLoader"),
        tensor_smith=dict(
            type="DivisibleCameraImageTensor",
            means=[123.675, 116.280, 103.530],
            stds=[58.395, 57.120, 57.375],
            image_size_divisor=32,
            image_pad_value=0.0)),
    ego_poses=dict(type='EgoPoseSet'),
    bbox_3d=dict(
        type='Bbox3D',
        loader=dict(
            type="AdvancedBbox3DLoader",
            class_mapping=class_mapping,
        ),
        # tensor_smith=dict(type='XyzLwhYawVeloBbox3D', voxel_shape=voxel_shape, voxel_range=voxel_range)
        tensor_smith=dict(type='Bbox3DBasic', classes=list(class_mapping.keys()), voxel_range=voxel_range)
    ),
)

train_dataset = dict(
    type="GroupBatchDataset",
    name="MvParkingTest",
    data_root="/ssd4/datasets/nuScenes",
    info_path="/ssd4/datasets/nuScenes/nusc_scene0001_train_info_separated.pkl",
    model_feeder=dict(
        type="StreamPETRModelFeeder",
        visible_range=point_cloud_range,
        bbox_3d_pos_repr="bottom_center",
        lidar_extrinsics=[
            [ 0.00203327,  0.99970406,  0.02424172,  0.943713  ],
            [-0.9999805 ,  0.00217566, -0.00584864,  0.        ],
            [-0.00589965, -0.02422936,  0.99968904,  1.84023   ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ],
    ),
    transformables=transformables,
    transforms=[
        dict(type='BGR2RGB'),
    ],
    group_sampler=dict(type="IndexGroupSampler",
                        phase="val",
                    #    indices_path="/ssd4/datasets/nuScenes/nusc_scene0001_train_info_separated_indices.txt",
                        possible_group_sizes=[20],
                        possible_frame_intervals=[1]),
    batch_size=batch_size,
)

val_dataset = dict(
    type="GroupBatchDataset",
    name="MvParkingTest",
    data_root="/ssd4/datasets/nuScenes",
    info_path="/ssd4/datasets/nuScenes/nusc_scene0001_train_info_separated.pkl",
    model_feeder=dict(
        type="StreamPETRModelFeeder",
        visible_range=point_cloud_range,
        bbox_3d_pos_repr="bottom_center",
        lidar_extrinsics=[
            [ 0.00203327,  0.99970406,  0.02424172,  0.943713  ],
            [-0.9999805 ,  0.00217566, -0.00584864,  0.        ],
            [-0.00589965, -0.02422936,  0.99968904,  1.84023   ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ],
    ),
    transformables=transformables,
    transforms=[
        dict(type='BGR2RGB'),
    ],
    group_sampler=dict(type="IndexGroupSampler",
                        phase="val",
                    #    indices_path="/ssd4/datasets/nuScenes/nusc_scene0001_train_info_separated_indices.txt",
                        possible_group_sizes=[20],
                        possible_frame_intervals=[1]),
    batch_size=batch_size,
)

test_dataset = dict(
    type="GroupBatchDataset",
    name="MvParkingTest",
    data_root="/ssd4/datasets/nuScenes",
    info_path="/ssd4/datasets/nuScenes/nusc_scene0001_train_info_separated.pkl",
    model_feeder=dict(
        type="StreamPETRModelFeeder",
        visible_range=point_cloud_range,
        bbox_3d_pos_repr="bottom_center",
        lidar_extrinsics=[
            [ 0.00203327,  0.99970406,  0.02424172,  0.943713  ],
            [-0.9999805 ,  0.00217566, -0.00584864,  0.        ],
            [-0.00589965, -0.02422936,  0.99968904,  1.84023   ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ],
    ),
    transformables=transformables,
    transforms=[
        dict(type='BGR2RGB'),
    ],
    group_sampler=dict(type="IndexGroupSampler",
                        phase="val",
                    #    indices_path="/ssd4/datasets/nuScenes/nusc_scene0001_train_info_separated_indices.txt",
                        possible_group_sizes=[20],
                        possible_frame_intervals=[1]),
    batch_size=1,
)

train_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset,
)

val_dataloader = dict(
    num_workers=0,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=val_dataset,
    persistent_workers=False,
    pin_memory=True,
)

test_dataloader = dict(
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="collate_dict"),
    dataset=test_dataset,
    persistent_workers=False,
    pin_memory=True,
)

train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=num_epochs, val_interval=1)  # -1 note don't eval
val_cfg = dict(type="GroupBatchValLoop")
test_cfg = dict(type="GroupBatchInferLoop")

model = dict(
    type="StreamPETR",
    data_preprocessor=dict(
        type="FrameBatchMerger",
        device="cuda",
    ),
    img_backbone=dict(
        pretrained="torchvision://resnet50",
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style="pytorch",
    ),
    img_neck=dict(type="mmdet3d.CPFPN", in_channels=[1024, 2048], out_channels=256, num_outs=2),
    # roi_head=dict(
    #     type="FocalHead",
    #     num_classes=len(class_mapping),
    #     loss_cls2d=dict(
    #         type='mmdet.QualityFocalLoss',
    #         use_sigmoid=True,
    #         beta=2.0,
    #         loss_weight=2.0),
    #     loss_centerness=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    #     loss_bbox2d=dict(type='mmdet.L1Loss', loss_weight=5.0),
    #     loss_iou2d=dict(type='mmdet.GIoULoss', loss_weight=2.0),
    #     loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
    #     train_cfg=dict(
    #         assigner2d=dict(
    #             type='mmdet.HungarianAssigner2D',
    #             cls_cost=dict(type='FocalLossCost', weight=2.),
    #             reg_cost=dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0),
    #             centers2d_cost=dict(type='mmdet.BBox3DL1Cost', weight=10.0))
    #     ),
    # ),
    box_head=dict(
        type='StreamPETRHead',
        num_classes=len(class_mapping),
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
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_mapping)),
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
                reg_cost=dict(type="mmdet.BBox3DL1Cost", weight=0.25),
                iou_cost=dict( type="mmdet.IoUCost", weight=0.0 ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        ),
    ),
)

val_evaluator = dict(type="AccuracyPetr")
test_evaluator = dict(type="AccuracyPetr")


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=6e-5, # total lr per gpu lr is lr/n
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.25),  # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

## scheduler configs
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, end_factor=1, by_epoch=False, begin=0, end=500), # warmup
    dict(type='CosineAnnealingLR', by_epoch=False, begin=500, eta_min=1e-5)     # main LR Scheduler
    # dict(type='PolyLR', by_epoch=False, begin=0, eta_min=0, power=1.0)     # main LR Scheduler
]


visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")])


log_processor = dict(type='GroupAwareLogProcessor')

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10, save_best="accuracy", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

custom_hooks = [
    dict(type="DumpPETRDetectionAsNuscenesJsonHook",
         det_anno_transformable_keys=["bbox_3d"],
         pre_conf_thresh=0.3),
]


today = datetime.datetime.now().strftime("%m%d")

work_dir = f'./work_dirs/{experiment_name}_{today}'
# load_from = "work_dirs/stream_petr_nusc_r50_0513/stream_petr_nusc_r50_0513_epoch_500.pth"

resume = False
