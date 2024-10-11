experiment_name = 'cmt_train_demo'
default_scope = "prefusion"
custom_imports = dict(imports=["prefusion", "contrib.cmt"], allow_failed_imports=False)

work_dir = 'cmt_demo'
backend_args = None
class_names = det_classes = [
    "class.vehicle.passenger_car",
    "class.traffic_facility.box",
    "class.road_marker.arrow",
    "class.parking.text_icon",
    "class.cycle.motorcycle",
]
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
resolutions = {
    'camera1': [640, 320],
    'camera5': [640, 320],
    'camera8': [640, 320],
    'camera11': [640, 320],
}
# out_size_factor = 8
# total_epochs = 30
# evaluation = dict(interval=total_epochs)
# dataset_type = 'CustomNuScenesDataset'
# dataset_type = 'CustomMTVDataset'
# data_root = 'data/mtv/'
# data_ori_root = 'data/nuscenes'
# input_modality = dict(
#     use_lidar=True,
#     use_camera=True,
#     use_radar=False,
#     use_map=False,
#     use_external=False)
train_dataloader = dict(
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=dict(
        type="GroupBatchDataset",
        name="MvParkingTest",
        data_root="/ssd1/data/4d",
        info_path="/ssd1/data/4d/mv4d_infos_mini_lidar.pkl",
        dictionaries={
            "camera_images": {},
            "bbox_3d": {"det": {"classes": det_classes}},
            "lidar_points": {},
        },
        tensor_smiths=dict(
            camera_images=dict(
                type="CameraImageTensor",
                means=[123.675, 116.280, 103.530],
                stds=[58.395, 57.120, 57.375],
            ),
            bbox_3d=dict(type="Bbox3D_XYZ_LWH_Yaw_VxVy", classes=det_classes),
            lidar_points=dict(type="PointsToVoxelsTensor", voxel_size=voxel_size,
                              max_point_per_voxel=10, max_voxels=120000,
                              max_input_points=1200000,
                              point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
        ), # TODO different num in train/val phase in tensor smith
        model_feeder=dict(type="CMTModelFeeder"),
        transformable_keys=["camera_images", "bbox_3d", "ego_poses", 'lidar_points'],
        transforms=[
            dict(type="RenderIntrinsic", resolutions=resolutions),
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
# --
train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=24, val_interval=-1)  # -1 note don't eval
val_cfg = dict(type="GroupValLoop")
test_cfg = dict(type="GroupTestLoop")
# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
#
# ida_aug_conf = {
#     "resize_lim": (0.3, 0.4),
#     "final_dim": (352, 576),
#     "bot_pct_lim": (0.0, 0.0),
#     "rot_lim": (0.0, 0.0),
#     "H": 1080,
#     "W": 1920,
#     "rand_flip": True,
# }

# train_pipeline = [
#     dict(
#         type='LoadPointsFromPCD',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#     ),
#     dict(
#         type='LoadPointsFromMultiSweepsPCD',
#         sweeps_num=10,
#         use_dim=[0, 1, 2, 3, 4],
#     ),
#     dict(type='LoadMultiViewImageFromFiles'),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#
#     dict(type='ModalMask3D', mode='train'),
#     dict(  # lidar
#         type='GlobalRotScaleTransAll',
#         # rot_range=[-0.3925 * 8, 0.3925 * 8],  # -45, 45;r
#         rot_range=[-0.3925 * 2, 0.3925 * 2],  # -45, 45;r
#         scale_ratio_range=[0.9, 1.1],
#         translation_std=[0.5, 0.5, 0.5]),
#     dict(  # flip
#         type='CustomRandomFlip3D',
#         sync_2d=False,
#         flip_ratio_bev_horizontal=0.5,
#         flip_ratio_bev_vertical=0.5),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectNameFilter', classes=class_names),
#     dict(type='PointShuffle'),
#     dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
#     dict(type='NormalizeMultiviewImage', **img_norm_cfg),
#     dict(type='PadMultiViewImage', size_divisor=32),
#     dict(type='DefaultFormatBundle3D', class_names=class_names),
#     dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
#          meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
#                     'depth2img', 'cam2img', 'pad_shape',
#                     'scale_factor', 'flip', 'pcd_horizontal_flip',
#                     'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
#                     'img_norm_cfg', 'pcd_trans', 'sample_idx',
#                     'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
#                     'transformation_3d_flow', 'rot_degree',
#                     'gt_bboxes_3d', 'gt_labels_3d',
#                     'cam_intrinsic', 'lidar2cam', 'cam_inv_poly'
#                     ))
# ]
# test_pipeline = [
#     dict(
#         type='LoadPointsFromPCD',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#     ),
#     dict(
#         type='LoadPointsFromMultiSweepsPCD',
#         sweeps_num=10,
#         use_dim=[0, 1, 2, 3, 4],
#     ),
#     dict(type='LoadMultiViewImageFromFiles'),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1.0, 1.0],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
#             dict(type='NormalizeMultiviewImage', **img_norm_cfg),
#             dict(type='PadMultiViewImage', size_divisor=32),
#             dict(
#                 type='DefaultFormatBundle3D',
#                 class_names=class_names,
#                 with_label=False),
#             dict(type='Collect3D',
#                  keys=[
#                      'points', 'img', 'cam_intrinsic', 'lidar2cam', 'cam_inv_poly', # 'gt_labels_3d'
#                  ],
#                  meta_keys=[
#                      'pad_shape', 'lidar2img', 'lidar2cam',
#                      'cam_intrinsic', 'lidar2cam', 'cam_inv_poly', 'intrinsic',
#                      'box_type_3d', 'filename', 'gt_labels_3d'
#                  ])
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,  # gpu = 4
#     workers_per_gpu=2,
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file=data_root + 'cmt_pkl_lidar_ego.pkl',
#             load_interval=1,
#             pipeline=train_pipeline,
#             classes=class_names,
#             modality=input_modality,
#             test_mode=False,
#             box_type_3d='LiDAR')),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'cmt_pkl_lidar_ego.pkl',
#         load_interval=1,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + '/cmt_pkl_lidar_ego_validate_train.pkl',
#         load_interval=1,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'),
#     shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
#     nonshuffler_sampler=dict(type='DistributedSampler'),
# )
out_size_factor = 8
model = dict(
    type='CmtDetector',
    data_preprocessor=dict(
        type="FrameBatchMerger",
        device="cuda",
    ),
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4', 'stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=120000,  # train & val 160000
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='mmdet3d.HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='mmdet3d.SparseEncoder',
        in_channels=5,
            sparse_shape=[41, 1440, 1440],
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
    pts_bbox_head=dict(
        type='CmtFisheyeHead',
        # num_query=1500,
        num_query=900,
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=len(class_names), class_names=class_names),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names)),
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
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
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,  # unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))

val_evaluator = dict(type="Accuracy")
test_evaluator = dict(type="Accuracy")
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
lr = 2e-4  # total lr per gpu lr is lr/n
num_epochs = 3

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=lr,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),  # origin none
)

# optimizer = dict(
#     type='AdamW',
#     lr=0.0002,  # 0.0001， 最大是0.0004，其实感觉用模拟退火可能效果更好。
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.01, decay_mult=5),
#             'img_neck': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
# optimizer_config = dict(
#     type='CustomFp16OptimizerHook',
#     loss_scale='dynamic',
#     grad_clip=dict(max_norm=35, norm_type=2),
#     custom_fp16=dict(pts_voxel_encoder=False, pts_middle_encoder=False, pts_bbox_head=False))
# # lr_config = dict(
# #     policy='cyclic',
# #     target_ratio=(8, 0.0001),
# #     cyclic_times=1,
# #     step_ratio_up=0.4)
param_scheduler = [
    dict(type='CosineAnnealingLR',
         eta_min=0.005,
         begin=num_epochs * 0.75,
         end=num_epochs,
         T_max=num_epochs * 0.25,
         by_epoch=True,
         )
]
log_processor = dict(type='GroupAwareLogProcessor')
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
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=1.0 / 3,
#     min_lr_ratio=1e-3)
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.8947368421052632, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# checkpoint_config = dict(interval=1)
# log_config = dict(
#     interval=10,
#     hooks=[dict(type='TextLoggerHook'),
#            dict(type='TensorboardLoggerHook')])
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = None
# # load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
# # load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
# # load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
# # load_from = 'work_dirs/cmt_voxel0075_vov_1600x640_cbgs_dbg/epoch_10.pth'
# # load_from = 'picked_models/cmt_11cls_9scenes_0712.pth'
# # load_from = 'picked_models/epoch_1_one_lidar.pth'
# # load_from = 'work_dirs/mtv_voxel0075_vov_640x320_temporal/epoch_16.pth'
# load_from = 'work_dirs/cmt_voxel0075_vov_640x320_cbgs_dbg_rotation/epoch_18.pth'
# resume_from = None
# workflow = [('train', 1)]
# gpu_ids = range(0, 8)
#