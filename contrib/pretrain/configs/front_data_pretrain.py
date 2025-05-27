__base__ = '../../configs/default_runtime.py'
default_scope = "prefusion"
custom_imports = dict(
    imports=['prefusion', 'contrib.pretrain'],
    allow_failed_imports=False
)
dataset_front_type = 'PretrainDataset_FrontData'
data_root = 'data/voc_bm_with_attrs_resized/'
# data_root = "/home/wuhan/prefusion"

crop_size = (384, 768)
train_pipeline = [
    dict(type='FusionLoadImageFromFile'),
    dict(type='DetLoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        # ratio_range=(1.0,1.0),
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=crop_size),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='FusionLoadImageFromFile'),
    dict(type='DetLoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        scale=crop_size,
        keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]
batch_size = 4

dataset_front = dict(
    type=dataset_front_type,
    data_root=data_root,
    ann_file="front_data_val_index.txt",
    # ann_file="tests/contrib/pretrain/index.txt",
    pipeline=train_pipeline,
    reduce_zero_label=False,
    lazy_init=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type="SameSourceBatchSampler", drop_last=True),
    # dataset=dict(
    #     type=dataset_type_wrapper,
    #     datasets=[dataset_avp, dataset_4d]
    # )
    dataset=dataset_front
)


val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_front_type,
        data_root=data_root,
        ann_file="front_data_val_index.txt",
        # ann_file="tests/contrib/pretrain/index.txt",
        pipeline=val_pipeline,
        reduce_zero_label=False
    )
)


image_size = crop_size
batch_augments = [dict(type='mmdet.BatchFixedSizePad', size=image_size)]

data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[128, 128, 128],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments
        )

model = dict(
    type='ADAS_Det',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='VoVNet',
                  out_indices=(0, 1, 3, 5),
                  ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs=False, 
        num_outs=4,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FusionFCOSHead',
        num_classes=8,
        regress_ranges=((-1, 64), (64, 128), (128, 256),
                        (256, 100000000)),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
        bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.5,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

max_epochs = 24
train_cfg = dict(type="mmengine.EpochBasedTrainLoop",max_epochs=max_epochs, val_interval=2)  # -1 note don't eval
val_cfg = dict(type="mmengine.ValLoop")

backend_args = None

val_evaluator = dict(type='FusionDetMetric', metric='mAP', eval_mode='11points', random_show=0.001)

debug_mode = False

if debug_mode:
    val_evaluator = None
    val_dataloader = None
    val_cfg = None

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
find_unused_parameters = True

runner_type = 'GroupBatchRunner'

lr = 0.02  # total lr per gpu lr is lr/n 
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'backbone': dict(lr_mult=1)
    #     }),
    # dtype="float16"  # it works only for arg --amp
    )
param_scheduler = dict(type='MultiStepLR', milestones=[16, 20])

auto_scale_lr = dict(enable=False, batch_size=32)

log_processor = dict(type='mmengine.LogProcessor')
default_hooks = dict(
    timer=dict(type='mmengine.IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='ExperimentWiseCheckpointHook', interval=max_epochs))

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