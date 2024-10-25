__base__ = '../../configs/default_runtime.py'
default_scope = "prefusion"
custom_imports = dict(
    imports=['prefusion', 'contrib.fastbev_det', 'contrib.mono_depth'],
    allow_failed_imports=False
)

backend_args = None

IMG_KEYS = [
        'VCAMERA_FISHEYE_FRONT', 'VCAMERA_PERSPECTIVE_FRONT_LEFT', 'VCAMERA_PERSPECTIVE_BACK_LEFT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_PERSPECTIVE_BACK', 'VCAMERA_FISHEYE_BACK', 
        'VCAMERA_PERSPECTIVE_FRONT_RIGHT', 'VCAMERA_PERSPECTIVE_BACK_RIGHT', 'VCAMERA_FISHEYE_RIGHT', 'VCAMERA_PERSPECTIVE_FRONT'
        ]
data_root = "data/mv_4d_data/"
data_root = "data/146_data/"

downsample_factor=4

img_scale = 2
fish_img_size = [256 * img_scale, 160 * img_scale]
perspective_img_size = [256 * img_scale, 192 * img_scale]
front_perspective_img_size = [768, 384]
batch_size = 3
group_size = 1

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
]

val_pipeline = [
    dict(type='RenderIntrinsic',
        resolutions=resolutions,
        scope='frame'
    )
]


transformable_name = ['camera_images','camera_depths', 'mono_depth']

transformables={
    "camera_images": dict(type="CameraImageSet"),
    "camera_depths": dict(type="CameraDepthSet"),
    "mono_depth" :dict(type="MonoDepthSet")
}


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
    type='MonoDepth',
    data_preprocessor=dict(
        type='GroupDataPreprocess',
        mean=[128, 128, 128],
        std=[255, 255, 255],
        IMG_KEYS=IMG_KEYS, 
        label_type=transformable_name,
        predict_elements=['heatmap', 'anno_boxes', 'gridzs', 'class_maps'],
        batch_size=batch_size,
        group_size=group_size,
        label_start_idx=2, # process labels info start index of transformable_name
    ),
    backbone=dict(type='ImgBackboneNeck',
                  img_backbone_conf=img_backbone_conf,
                  img_neck_conf=img_neck_conf),
    mono_depth = dict(type='Mono_Depth_Head', 
                      fish_img_size=fish_img_size, 
                      pv_img_size=perspective_img_size, 
                      front_img_size=front_perspective_img_size, 
                      downsample_factor=downsample_factor, 
                      batch_size=batch_size, 
                      avg_reprojection=True,
                      disparity_smoothness=0.001,
                      fish_unproject_cfg=dict(type='Fish_BackprojectDepth', 
                                              batch_size=batch_size, 
                                              height=fish_img_size[1], 
                                              width=fish_img_size[0],
                                              intrinsic=((fish_img_size[0]-1)/2, (fish_img_size[1]-1)/2, fish_img_size[0]/4, fish_img_size[0]/4, 0.1, 0,0,0)),
                      fish_project3d_cfg=dict(type='Fish_Project3D', 
                                              batch_size=batch_size, 
                                              height=fish_img_size[1], 
                                              width=fish_img_size[0],
                                              intrinsic=((fish_img_size[0]-1)/2, (fish_img_size[1]-1)/2, fish_img_size[0]/4, fish_img_size[0]/4, 0.1, 0,0,0)),                      
                      mono_depth_net_cfg=dict(type='Mono_DepthReducer', img_channels=256, mid_channels=128)
    )
)    

val_evaluator = None


train_cfg = dict(max_epochs=24, val_interval=2)  # -1 note don't eval


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