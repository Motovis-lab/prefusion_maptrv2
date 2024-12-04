experiment_name = "stream_petr_r50_demo_grpsize1_bs4_lr6e-5"

_base_ = "../../../configs/default_runtime.py"

custom_imports = dict(imports=["prefusion", "contrib.petr"], allow_failed_imports=False)

backend_args = None

det_classes = [
    'class.cycle.motorcycle',
    'class.parking.text_icon',
    'class.pedestrian.pedestrian',
    'class.road_marker.arrow',
    'class.traffic_facility.box',
    'class.traffic_facility.speed_bump',
    'class.vehicle.passenger_car'
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
            bbox_3d_directional=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        passenger_car=["class.vehicle.passenger_car"],
                        bus=["class.vehicle.bus"],
                        truck=["class.vehicle.truck"],
                        ambulance=["class.vehicle.ambulance"],
                        fire_engine=["class.vehicle.fire_engine"],
                        trailer=["class.vehicle.trailer"],
                        construction=["class.vehicle.construction"],
                        env_protect=["class.vehicle.env_protect"],
                        toy_car=["class.wheeled_push_device.toy_car"],
                        street_vendor=["class.wheeled_push_device.street_vendor"],
                        tricycle=["class.cycle.tricycle"],
                        motorcycle=["class.cycle.motorcycle"],
                        bicycle=["class.cycle.bicycle"],
                        cleaning_cart=["class.wheeled_push_device.cleaning_cart"],
                        stroller=["class.wheeled_push_device.stroller"],
                        shopping_cart=["class.wheeled_push_device.shopping_cart"],
                        scooter=["class.wheeled_push_device.scooter"],
                        animal=["class.animal.animal"],
                    ),
                    attr_mapping=dict(
                        is_trunk_open=["attr.vehicle.is_trunk_open.true"],
                        is_door_open=["attr.vehicle.is_door_open.true"],
                        is_with_rider=["attr.cycle.is_with_rider.true"],
                    ),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_3d_directional_ground=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        arrow_ahead=["class.road_marker.arrow::attr.road_marker.arrow.type.ahead"],
                        arrow_left=["class.road_marker.arrow::attr.road_marker.arrow.type.left"],
                        arrow_right=["class.road_marker.arrow::attr.road_marker.arrow.type.right"],
                        arrow_uturn=["class.road_marker.arrow::attr.road_marker.arrow.type.uturn"],
                        arrow_ahead_left=["class.road_marker.arrow::attr.road_marker.arrow.type.ahead_left"],
                        arrow_ahead_right=["class.road_marker.arrow::attr.road_marker.arrow.type.ahead_right"],
                        arrow_ahead_left_right=["class.road_marker.arrow::attr.road_marker.arrow.type.ahead_left_right"],
                        arrow_left_right=["class.road_marker.arrow::attr.road_marker.arrow.type.left_right"],
                    ),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_3d_directional_cylinder=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        pedestrian=["class.pedestrian.pedestrian"],
                    ),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_3d_cylinder=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        bollard=["class.traffic_facility.bollard"],
                        fire_hydrant=["class.traffic_facility.box::attr.traffic_facility.box.type.fire_hydrant"],
                        cone=["class.traffic_facility.cone"],
                        crash_barrel=["class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.crash_barrel"],
                        stone_ball=["class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.stone_ball"],
                        retractable_roadblock=["class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.retractable_roadblock"],
                        cylindrical_pillar=["class.parking.indoor_column::attr.parking.indoor_column.shape.cylindrical"],
                        undefined_pillar=["class.parking.indoor_column::attr.parking.indoor_column.shape.undefined"],
                    ),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_3d_rect_cuboid=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    axis_rearrange_method="longer_edge_as_y",
                    class_mapping=dict(
                        fire_box=["class.traffic_facility.box::attr.traffic_facility.box.type.fire_box"],
                        distribution_box=["class.traffic_facility.box::attr.traffic_facility.box.type.distribution_box"],
                        waste_bin=["class.traffic_facility.box::attr.traffic_facility.box.type.waste_bin"],
                        water_filled_barrier=["class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.water_filled_barrier"],
                        cement_isolation_pier=["class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.cement_isolation_pier"],
                        speed_bump=["class.traffic_facility.speed_bump"],
                        charging_infra=["class.parking.charging_infra"],
                        text_road_marker=["class.road_marker.text"],
                        parking_lock=["class.parking.parking_lock"],
                        parking_slot_text_icon=["class.parking.text_icon"],
                        wheel_stopper=["class.parking.wheel_stopper"],
                        undefined_box=["class.traffic_facility.box::attr.traffic_facility.box.type.undefined"],
                        undefined_barrier=[
                            "class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.undefined", 
                            "class.traffic_facility.hard_barrier::attr.traffic_facility.hard_barrier.type.undefined"
                        ],
                    ),
                    attr_mapping=dict(
                        unlocked=["class.parking.parking_lock::attr.parking.parking_lock.state.unlocked"],
                        unlocked=["class.parking.parking_lock::attr.parking.parking_lock.state.locked"],
                        text_icon_number=["class.parking.text_icon::attr.parking.text_icon.type.number"],
                        text_icon_time=["class.parking.text_icon::attr.parking.text_icon.type.time"],
                        text_icon_accessible=["class.parking.text_icon::attr.parking.text_icon.type.accessible"],
                        text_icon_undefined=["class.parking.text_icon::attr.parking.text_icon.type.undefined"],
                        is_wheel_stopper_separated=["class.parking.wheel_stopper::attr.parking.wheel_stopper.is_separated.true"],
                    ),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            bbox_3d_square_pillar=dict(
                type="Bbox3D", 
                loader=dict(
                    type="AdvancedBbox3DLoader",
                    class_mapping=dict(
                        no_parking_barrier=["class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.no_parking"],
                        regtangular_pillar=["class.parking.indoor_column::attr.parking.indoor_column.shape.regtangular"],
                    ),
                ),
                tensor_smith=dict(type="Bbox3DBasic", classes=det_classes)),
            polyline_3d=dict(
                type="Polyline3D",
                loader=dict(
                    type="Polyline3DClassMappingLoader",
                    class_mapping=dict(
                        speed_down_marker=["class.road_marker.speed_down"],
                        lane_line=["class.road_marker.lane_line"],
                        no_parking_zone=["class.road_marker.no_parking_zone"],
                        crosswalk=["class.road_marker.crosswalk"],
                        gore_area=["class.road_marker.gore_area"],
                        access_aisle=["class.parking.access_aisle"],
                    ),
                    attr_mapping=dict(
                        yellow=["attr.common.color.single_color.yellow"],
                        white=["attr.common.color.single_color.white"],
                        undefined_color=["attr.common.color.single_color.undefined"],
                        dashed=["attr.road_marker.lane_line.style.dashed"],
                        solid=["attr.road_marker.lane_line.style.solid"],
                    )
                ),
                tensor_smith=dict(type="Polyline3D")
            ),
            parking_slot=dict(
                type="ParkingSlot3D",
                loader=dict(
                    type="ParkingSlot3DClassMappingLoader",
                    class_mapping=dict(
                        parking_slot=["class.parking.parking_slot"],
                    ),
                    attr_mapping=dict(
                        is_mechanical=["attr.parking.parking_slot.is_mechanical.true"],
                    )
                ),
                tensor_smith=dict(type="ParkingSlot3DBasic")
            ),
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
        group_sampler=dict(type="IndexGroupSampler",
                           phase="train",
                           possible_group_sizes=[1],
                           possible_frame_intervals=[1]),
        batch_size=batch_size,
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
