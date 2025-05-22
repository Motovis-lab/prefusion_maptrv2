default_scope = 'mmdet'
custom_imports = dict(
    imports=['prefusion', 'contrib.pretrain'],
    allow_failed_imports=False
)
dataset_front_type = 'prefusion.PretrainDataset_FrontData'
data_root = 'data/voc_bm_with_attrs_resized'
# data_root = "/home/wuhan/prefusion"
# ori_shape=(640, 1024)
crop_size = (384, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='prefusion.DetLoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=crop_size,
        # ratio_range=(0.5, 2.0),
        ratio_range=(1.0,1.0),
        keep_ratio=True),
    # dict(type='mmdet.RandomCrop', crop_size=crop_size),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='prefusion.DetLoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        scale=crop_size,
        keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]
batch_size = 8

dataset_front = dict(
    type=dataset_front_type,
    data_root=data_root,
    ann_file="front_data_val_index.txt",
    # ann_file="tests/contrib/pretrain/index.txt",
    pipeline=train_pipeline,
    reduce_zero_label=False
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
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
    num_workers=12,
    persistent_workers=True,
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
test_dataloader = val_dataloader


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False


# model settings
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[8, 18],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator