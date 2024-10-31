experiment_name = "stream_petr_r50_demo_grpsize1_bs4_lr6e-5"

_base_ = "../../../configs/default_runtime.py"

custom_imports = dict(imports=["prefusion", "contrib.petr"], allow_failed_imports=False)

backend_args = None

det_classes = [
    'motorcycle',
    'text_icon',
    'pedestrian',
    'arrow',
    'box',
    'speed_bump',
    'passenger_car',
]

def _calc_grid_size(_range, _voxel_size, n_axis=3):
    return [(_range[n_axis+i] - _range[i]) // _voxel_size[i] for i in range(n_axis)]

batch_size = 4
num_epochs = 500
lr = 6e-5  # total lr per gpu lr is lr/n
voxel_size = [0.1, 0.1, 3]
point_cloud_range = [-12.8, -12.8, -1.0, 12.8, 12.8, 2.0]
grid_size = _calc_grid_size(point_cloud_range, voxel_size)

train_dataloader = dict(
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler"),
    collate_fn=dict(type="collate_dict"),
    dataset=dict(
        type="GroupBatchDataset",
        name="MvParkingTest",
        data_root="/data/datasets/mv4d",
        info_path="/data/datasets/mv4d/mv4d_infos_dbg_246_noalign.pkl",
        model_feeder=dict(
            type="StreamPETRModelFeeder",
            visible_range=point_cloud_range,
        ),
        transformables=dict(
            camera_images=dict(
                type="CameraImageSet",
                loader=dict(type="CameraImageSetLoader"),
                tensor_smith=dict(
                    type="CameraImageTensor",
                    means=[123.675, 116.280, 103.530],
                    stds=[58.395, 57.120, 57.375])),
            bbox_3d=dict(  # bbox_directional
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        motorcycle=['class.cycle.motorcycle'],
                        arrow=['class.road_marker.arrow'],
                        passenger_car=['class.vehicle.passenger_car'],
                    ),
                    attr_mapping=dict(
                        is_door_open=["attr.vehicle.is_door_open.true"],
                        is_trunk_open=["attr.vehicle.is_trunk_open.true"],
                    )
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_rect_cuboid=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        text_icon=['class.parking.text_icon'],
                        box=['class.traffic_facility.box'],
                        speed_bump=['class.traffic_facility.speed_bump'],
                    ),
                    axis_rearrange_method="longer_edge_as_y",
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_cylinder_directional=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(pedestrian=['class.pedestrian.pedestrian']),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            ego_poses=dict(
                type="EgoPoseSet",
                loader=dict(type="EgoPoseSetLoader")
            )
        ),
        transforms=[
            # dict(type="RandomMirrorSpace", prob=0.5, scope="group"),
            dict(
                type="RandomImageISP",
                prob=0.0001,
            ),
        ],
        phase="train",
        batch_size=batch_size,
        possible_group_sizes=[1],
        possible_frame_intervals=[1],
    ),
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader

train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=num_epochs, val_interval=-1)  # -1 note don't eval
val_cfg = dict(type="GroupBatchValLoop")
test_cfg = dict(type="GroupBatchTestLoop")

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
    #     num_classes=len(det_classes),
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
        code_size=10, # x, y, z, l, w, h, sin(yaw), cos(yaw), Vx, Vy
        position_range=[-13.0, -13.0, -3.0, 13.0, 13.0, 3.0],
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
            post_center_range=[-13.0, -13.0, -3.0, 13.0, 13.0, 3.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(det_classes)),
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

val_evaluator = dict(type="Accuracy")
test_evaluator = dict(type="Accuracy")


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=lr,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.25),  # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# param_scheduler = dict(type='MultiStepLR', milestones=[12, 20])

log_processor = dict(type='GroupAwareLogProcessor')

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="precision", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

visualizer = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")])

# load_from = "work_dirs/r50/epoch_5.pth"
# resume = True
