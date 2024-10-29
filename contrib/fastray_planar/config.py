default_scope = "prefusion"
experiment_name = "fastray_planar_demo"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray_planar"], 
    allow_failed_imports=False
)


## camera and voxel feature configurations

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

## dictionaries and mappings for different types of tasks

mapping_heading_objects = {
    'passenger_car': [],
    'bus': [],
    'truck': [],
    'ambulance': [],
    'fire_engine': [],
    'env_protect': [],
    'tricycle':[],
    'motorcycle':[],
    'bicycle':[],
    'cleaning_cart':[],
    'shopping_cart':[],
    'stroller':[]    
}

dictionary_heading_objects = dict(
    classes=['passenger_car', 
             'bus', 
             'truck', 
             'ambulance', 
             'fire_engine', 
             'env_protect',
             'tricycle',
             'motorcycle',
             'bicycle',
             'cleaning_cart',
             'shopping_cart',
             'stroller'],
    attrs=['cycle.is_with_rider.true']
)

dictionary_plane_objects = dict(
    classes=[],
    attrs=[]
)

mapping_no_heading_objects = {
    'speed_bump': []
}

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


## camera configurations for model inputs

camera_groups = dict(
    pv_front=['VCAMERA_PERSPECTIVE_FRONT'],
    pv_sides=['VCAMERA_PERSPECTIVE_FRONT_LEFT',
              'VCAMERA_PERSPECTIVE_FRONT_RIGHT',
              'VCAMERA_PERSPECTIVE_BACK_LEFT',
              'VCAMERA_PERSPECTIVE_BACK_RIGHT',
              'VCAMERA_PERSPECTIVE_BACK'],
    fisheyes=['VCAMERA_FISHEYE_FRONT',
              'VCAMERA_FISHEYE_LEFT',
              'VCAMERA_FISHEYE_BACK',
              'VCAMERA_FISHEYE_RIGHT']
)

resolution_pv_front = (640, 320)
resolution_pv_sides = (512, 320)
resolution_fisheyes = (512, 320)

camera_resolution_configs=dict(
    VCAMERA_PERSPECTIVE_FRONT=resolution_pv_front,
    VCAMERA_PERSPECTIVE_FRONT_LEFT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_FRONT_RIGHT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_BACK_LEFT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_BACK_RIGHT=resolution_pv_sides,
    VCAMERA_PERSPECTIVE_BACK=resolution_pv_sides,
    VCAMERA_FISHEYE_FRONT=resolution_fisheyes,
    VCAMERA_FISHEYE_LEFT=resolution_fisheyes,
    VCAMERA_FISHEYE_BACK=resolution_fisheyes,
    VCAMERA_FISHEYE_RIGHT=resolution_fisheyes,
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
            dictionary=dict(classes=['class.vehicle.passenger_car'], attrs=[]),
            tensor_smith=dict(type='PlanarBbox3D', 
                              voxel_shape=voxel_feature_config['voxel_shape'],
                              voxel_range=voxel_feature_config['voxel_range'])
        ),
        polyline_3d=dict(
            type='Polyline3D',
            dictionary=dict(classes=['class.road_marker.lane_line'], attrs=[]),
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
    transforms=[
        dict(type='RandomRenderExtrinsic'),
        dict(type='RandomRotateSpace'),
        dict(type='RenderIntrinsic', resolutions=camera_resolution_configs),
        dict(type='RandomMirrorSpace'),
        dict(type='RandomImageISP'),
        dict(type='RandomSetIntrinsicParam', prob=0.2, jitter_ratio=0.01),
        dict(type='RandomSetExtrinsicParam', prob=0.2, angle=1, translation=0.02)
    ],
    phase="train",
    batch_size=4,
    possible_group_sizes=4,
    possible_frame_intervals=5,
)


train_dataloader = dict(
    num_workers=1,
    persistent_workers=True,
    collate_fn=dict(type="collate_dict"),
    dataset=train_dataset
)

val_dataloader = train_dataloader

train_cfg = dict(type="GroupBatchTrainLoop", max_epochs=1, val_interval=-1)
val_cfg = dict(type="GroupBatchValLoop")

