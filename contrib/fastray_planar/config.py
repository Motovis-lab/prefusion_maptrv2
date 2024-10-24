experiment_name = "fastray_planar_demo"

_base_ = "../../../configs/default_runtime.py"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray_planar"], 
    allow_failed_imports=False
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
voxel_feature_config=dict(
    voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
    voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
    ego_distance_max=40,
    ego_distance_step=5
)

# TODO:
# classes and attrs needs to be redefined in info.pkl
# or in transformable_loader or in tensor_smith
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

dictionary_no_heading_objects = dict(
    classes=['speed_bump'],
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
    data_root='../MV4D-PARKING/20231028_150815',
    info_path='../MV4D-PARKING/info_pkls/mv_4d_infos_20231028_150815.pkl',
    model_feeder=dict(
        type="FastRayPlanarModelFeeder",
        voxel_feature_config=voxel_feature_config,
        camera_feature_configs=camera_feature_configs,
    ),
    transformables=dict(
        camera_images=dict(type='CameraImageSet', tensor_smith=dict(type='CameraImageTensor')),
        ego_poses=dict(type='ego_poses'),
        bbox_3d=dict(
            type='Bbox3D', 
            dictionary=dictionary_heading_objects,
            tensor_smith=dict(type='PlanarBbox3D', 
                              voxel_shape=voxel_feature_config['voxel_shape'],
                              voxel_range=voxel_feature_config['voxel_range'])
        ),
        polyline_3d=dict(
            type='Polyline3D',
            dictionary=dictionary_polylines,
            tensor_smith=dict(type='PlanarPolyline3D', 
                              voxel_shape=voxel_feature_config['voxel_shape'],
                              voxel_range=voxel_feature_config['voxel_range'])
        ),
        parkingslot_3d=dict(
            type='ParkingSlot3D',
            tensor_smith=dict(type='PlanarParkingSlot3D', 
                              voxel_shape=voxel_feature_config['voxel_shape'],
                              voxel_range=voxel_feature_config['voxel_range'])
        )
    ),
    phase="train",
    batch_size=4,
    possible_group_sizes=4,
    possible_frame_intervals=5,
)


train_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader