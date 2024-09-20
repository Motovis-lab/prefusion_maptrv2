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

train_dataloader = dict(
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=dict(
        type="GroupBatchDataset",
        name="MvParkingTest",
        data_root="/Users/rlan/work/dataset/motovis/mv4d",
        info_path="/Users/rlan/work/dataset/motovis/mv4d/mv4d_infos.pkl",
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
            bbox_3d=dict(type="Bbox3DCorners"),
        ),
        model_feeder=dict(type="StreamPETRModelFeeder"),
        transformable_keys=["camera_images", "bbox_3d"],
        transforms=[
            dict(type="RandomMirrorSpace", prob=0.5, scope="group"),
            dict(
                type="RandomImageISP",
                prob=0.5,
            ),
        ],
        phase="train",
        batch_size=2,
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
        device="mps",
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
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.),
                reg_cost=dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='mmdet.BBox3DL1Cost', weight=10.0))
        ),
    ),
    box_head=dict(type="StreamPETRHead"),
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
