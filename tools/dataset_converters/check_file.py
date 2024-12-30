import mmengine
import os
from pathlib import Path as P
from rich.progress import track

data_root = P("./MV4D_12V3L/")

camera_check_elements = ['VCAMERA_FISHEYE_FRONT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_FISHEYE_BACK', 'VCAMERA_FISHEYE_RIGHT']
lidar_check_elements = ['lidar1']
sdf_check_elements = ['occ_2d', 'ground', 'sdf', 'bev_height_map', 'bev_lidar_mask', 'occ_edge_height_map', 'occ_edge_lidar_mask']

data = mmengine.load(data_root / "mv_4d_infos_train.pkl")
for scene_name in track(data, "process check info"):
    f = open(data_root / f"{scene_name}_check_info.txt", "a")
    for frame_id in data[scene_name]['frame_info']:
        frame_info = data[scene_name]['frame_info'][frame_id]
        for ele in camera_check_elements:
            if not os.path.isfile(data_root / P(frame_info['camera_image'][ele])):
                f.write(f"{frame_info['camera_image'][ele]}\n")
        for ele in lidar_check_elements:
            if not os.path.isfile(data_root / P(frame_info['lidar_points'][ele])):
                f.write(f"{frame_info['lidar_points'][ele]}\n")
        for ele in sdf_check_elements:
            if not os.path.isfile(data_root / P(frame_info['occ_sdf'][ele])):
                f.write(f"{frame_info['occ_sdf'][ele]}\n")
    f.close()