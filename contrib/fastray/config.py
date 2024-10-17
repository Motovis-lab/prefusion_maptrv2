experiment_name = "fastray_planar_demo"

_base_ = "../../../configs/default_runtime.py"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray"], 
    allow_failed_imports=False
)


voxel_feature_config=dict(
    voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
    voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
    ego_distance_max=40,
    ego_distance_step=5
)
default_camera_feature_config = dict(
    ray_distance_num_channel=64,
    ray_distance_start=0.25,
    ray_distance_step=0.25,
    feature_downscale=8,
)
camera_feature_configs=dict(
    VCAMERA_PERSPECTIVE_FRONT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_FRONT_LEFT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_FRONT_RIGHT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_BACK_LEFT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_BACK_RIGHT=default_camera_feature_config,
    VCAMERA_PERSPECTIVE_BACK=default_camera_feature_config,
    VCAMERA_FISHEYE_FRONT=default_camera_feature_config,
    VCAMERA_FISHEYE_LEFT=default_camera_feature_config,
    VCAMERA_FISHEYE_BACK=default_camera_feature_config,
    VCAMERA_FISHEYE_RIGHT=default_camera_feature_config,
)

# TODO:
# classes and attrs needs to be verified in info.pkl
# or redefine classes in tensor_smith
dictionary_heading_objects = dict(
    classes=['class.vehicle.passenger_car', 
             'class.vehicle.bus', 
             'class.vehicle.truck', 
             'class.vehicle.ambulance', 
             'class.vehicle.fire_engine', 
             'class.vehicle.env_protect',
             'class.cycle.tricycle',
             'class.cycle.motorcycle',
             'class.cycle.bicycle',
             'class.wheeled_push_device.cleaning_cart',
             'class.wheeled_push_device.shopping_cart',
             'class.wheeled_push_device.stroller'],
    attrs=['attr.cycle.is_with_rider.true']
)

dictionary_plane_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_square_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_cylinder_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_oriented_cylinder_objects = dict(
    classes=[],
    attrs=[]
)

dictionary_polylines = dict(
    classes=['class.road_marker.lane_line'],
    attrs=['attr.road_marker.lane_line.type.regular',
           'attr.road_marker.lane_line.type.wide',
           'attr.road_marker.lane_line.type.stop_line',
           'attr.common.color.single_color.yellow',
           'attr.common.color.single_color.white',
           'attr.road_marker.lane_line.style.dashed',
           'attr.road_marker.lane_line.style.solid']
)

dictionary_polygons = dict(
    classes=['class.road_marker.no_parking_zone',
             'class.road_marker.crosswalk',
             'class.road_marker.gore_area'],
    attrs=[]
)


train_dataset = dict(
    type='GroupBatchDataset',
    name="demo_parking",
    data_root='/home/alpha/Projects/MV4D-PARKING/20231028_150815',
    info_path='/home/alpha/Projects/MV4D-PARKING/info_pkls/mv_4d_infos_20231028_150815.pkl',
    model_feeder=dict(
        type="FastRayModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
    ),
    transformables=dict(
        camera_images=dict(type='camera_images', tensor_smith=dict(type='CameraImageTensor')),
        ego_poses=dict(type='ego_poses'),
        bbox_3d=dict(type='bbox_3d', dictionary=dictionary_heading_objects),
    )
)




# train_dataloader = dict(
#     num_workers=1,
#     persistent_workers=True,
#     collate_fn=dict(type="collate_dict"),
#     dataset=dict(
#         type="GroupBatchDataset",
#         name="ParkingDataset",
#         data_root="/data/datasets/mv4d",
#         info_path="/data/datasets/mv4d/mv4d_infos_dbg_246_noalign.pkl",
#         model_feeder=dict(
#             type="StreamPETRModelFeeder",
#             visible_range=point_cloud_range,
#         ),
#         transformables=dict(
#             camera_images=dict(
#                 type='camera_images', 
#                 tensor_smith=dict(
#                     type='CameraImageTensor',
#                     means=[123.675, 116.280, 103.530],
#                     stds=[58.395, 57.120, 57.375],
#                 )
#             ),
#             bbox_3d_0=dict(type='bbox_3d', dictionary={'classes': det_classes}),
#             bbox_3d_1=dict(type='bbox_3d', dictionary={'classes': det_classes})
#         )
#         transforms=[
#             # dict(type="RandomMirrorSpace", prob=0.5, scope="group"),
#             dict(
#                 type="RandomImageISP",
#                 prob=0.0001,
#             ),
#         ],
#         phase="train",
#         batch_size=batch_size,
#         possible_group_sizes=[1],
#         possible_frame_intervals=[1],
#     ),
# )

val_dataloader = train_dataloader
test_dataloader = train_dataloader