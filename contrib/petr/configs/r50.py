experiment_name = "stream_petr_r50_demo"

_base_ = "../../../configs/default_runtime.py"

# custom_imports = dict(
#     imports=['models', 'datasets', 'hooks', 'runner', 'utils', 'evaluator', 'losses'],
#     allow_failed_imports=False
# )
custom_imports = dict(imports=["prefusion", "contrib.petr"], allow_failed_imports=False)
# custom_imports = dict(
#     imports=['prefusion', 'contrib'],
#     allow_failed_imports=False
# )

backend_args = None

det_classes = [
    "class.vehicle.passenger_car",
    "class.traffic_facility.box",
    "class.road_marker.arrow",
    "class.parking.text_icon",
    "class.cycle.motorcycle",
]

voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

train_dataloader = dict(
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=dict(
        type="GroupBatchDataset",
        name="MvParkingTest",
        data_root="/ssd2/datasets/mv4d",
        info_path="/ssd2/datasets/mv4d/mv4d_infos.pkl",
        dictionaries={
            "camera_images": {},
            "bbox_3d": {"det": {"classes": det_classes}},
        },
        tensor_smiths=dict(
            camera_images=dict(
                type="CameraImageTensor",
                means=[123.675, 116.280, 103.530],
                stds=[58.395, 57.120, 57.375],
            ),
            bbox_3d=dict(type="Bbox3D_XYZ_LWH_Yaw_VxVy", classes=det_classes),
        ),
        model_feeder=dict(type="StreamPETRModelFeeder"),
        transformable_keys=["camera_images", "bbox_3d", "ego_poses"],
        transforms=[
            dict(type="RandomMirrorSpace", prob=0.5, scope="group"),
            dict(
                type="RandomImageISP",
                prob=0.5,
            ),
        ],
        phase="train",
        batch_size=3,
        possible_group_sizes=[3, 4, 5],
        possible_frame_intervals=[1, 2],
    ),
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader

train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=24, val_interval=-1)  # -1 note don't eval
val_cfg = dict(type="GroupValLoop")
test_cfg = dict(type="GroupTestLoop")

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
    roi_head=dict(
        type="FocalHead",
        num_classes=len(det_classes),
        loss_cls2d=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='mmdet.HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='mmdet.BBox3DL1Cost', weight=10.0))
        ),
    ),
    box_head=dict(
        type='StreamPETRHead',
        num_classes=len(det_classes),
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
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        train_cfg=dict(
            grid_size=[512, 512, 1],
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

val_evaluator = dict(type="Accuracy")
test_evaluator = dict(type="Accuracy")


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

lr = 0.006  # total lr per gpu lr is lr/n

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = dict(type="MultiStepLR", milestones=[16, 20])

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="precision", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")])


# load_from = "work_dirs/mv_4d_fastbev_t/20240903_023804/epoch_24.pth"
resume = False
