__base__ = '../../configs/default_runtime.py'
default_scope = "prefusion"
custom_imports = dict(
    imports=['prefusion', 'contrib.fastbev_det', 'contrib.pretrain'],
    allow_failed_imports=False
)
dataset_type_wrapper = "ConcatDataset"
dataset_type_4d = 'PretrainDataset'
dataset_type_avp = "PretrainDataset_AVP"
data_root = 'data/pretrain_data/'
avp_data_root = 'data/avp/'
# ori_shape=(640, 1024)
crop_size = (640, 1024)
train_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='LoadAnnotationsPretrain', with_depth=True, with_seg_mask=True, reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='mmseg.RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='mmseg.LoadImageFromFile'),
    dict(type='LoadAnnotationsPretrain', with_depth=True, with_seg_mask=True, reduce_zero_label=True),
    dict(
        type='Resize',
        scale=crop_size,
        keep_ratio=True),
    dict(type='PackSegInputs')
]
batch_size = 10

dataset_4d = dict(
    type=dataset_type_4d,
    data_root=data_root,
    ann_file="mv_4d_infos_pretrain_train.pkl",
    pipeline=train_pipeline,
    camera_types=["VCAMERA_FISHEYE_BACK", "VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT", "VCAMERA_FISHEYE_RIGHT"]
)

dataset_avp = dict(
        type=dataset_type_avp,
        data_root=avp_data_root,
        ann_file="avp_train.pkl",
        pipeline=train_pipeline,
        camera_types=["VCAMERA_FISHEYE_FRONT"]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_wrapper,
        datasets=[dataset_4d, dataset_avp]
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_4d,
        data_root=data_root,
        ann_file="mv_4d_infos_pretrain_val.pkl",
        pipeline=val_pipeline,
        camera_types=["VCAMERA_FISHEYE_BACK", "VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT", "VCAMERA_FISHEYE_RIGHT"])
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

data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[128, 128, 128],
    std=[255, 255, 255],
    size_divisor=32,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='mmseg.EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='ImgBackboneNeck',
                  img_backbone_conf=img_backbone_conf,
                  img_neck_conf=img_neck_conf),
    decode_head=dict(
        type='mmseg.DepthwiseSeparableASPPHead',
        in_channels=256,
        in_index=0,
        channels=128,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=26,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=26,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_cfg = dict(type="mmengine.EpochBasedTrainLoop",max_epochs=24, val_interval=2)  # -1 note don't eval
val_cfg = dict(type="mmengine.ValLoop")

backend_args = None

IMG_KEYS = [
        'VCAMERA_FISHEYE_FRONT', 'VCAMERA_PERSPECTIVE_FRONT_LEFT', 'VCAMERA_PERSPECTIVE_BACK_LEFT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_PERSPECTIVE_BACK', 'VCAMERA_FISHEYE_BACK', 
        'VCAMERA_PERSPECTIVE_FRONT_RIGHT', 'VCAMERA_PERSPECTIVE_BACK_RIGHT', 'VCAMERA_FISHEYE_RIGHT', 'VCAMERA_PERSPECTIVE_FRONT'
        ]

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU'])

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
find_unused_parameters = True

runner_type = 'GroupRunner'

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

log_processor = dict(type='mmengine.LogProcessor')
default_hooks = dict(
    timer=dict(type='mmengine.IterTimerHook'),
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