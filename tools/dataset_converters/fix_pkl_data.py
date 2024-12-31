import mmengine
from pathlib import Path as P
from mmengine import Config
from scipy.spatial.transform import Rotation
import numpy as np

file_names = P("/mnt/ssd1/wuhan/prefusion/data/MV4D_12V3L/").glob("mv_4d_infos_2023*.pkl")

fish_filenames = ["VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT", "VCAMERA_FISHEYE_BACK", "VCAMERA_FISHEYE_RIGHT"]
fishcamera_map = {"VCAMERA_FISHEYE_FRONT": "camera8",
                  "VCAMERA_FISHEYE_LEFT": "camera5",
                  "VCAMERA_FISHEYE_BACK": "camera1",
                  "VCAMERA_FISHEYE_RIGHT": "camera11"}
parm_cameras_v = {
    "VCAMERA_FISHEYE_FRONT":(-120, 0, -90),
    "VCAMERA_FISHEYE_LEFT":(-135, 0, 0),
    "VCAMERA_FISHEYE_RIGHT":(-135, 0, -180),
    "VCAMERA_FISHEYE_BACK":(-120, 0, 90)
}
calib = Config.fromfile("/mnt/ssd1/wuhan/prefusion/data/MV4D_12V3L/20230822_110856/calibration_center.yml")
R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()
W = 768
H = 512
cx = (W - 1) / 2
cy = (H - 1) / 2
fx = fy = W / 4
v_camera_intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)

for file in file_names:
    data = mmengine.load(file)
    scene_name = file.stem[12:]
    for key in data[scene_name]['scene_info']['calibration']:
        if key in fish_filenames:
            t = np.array((R_nus.T @ np.array(calib['rig'][fishcamera_map[key]]['extrinsic'][:3]).reshape(3)).tolist()).reshape(3)
            R = Rotation.from_euler('xyz', angles=parm_cameras_v[key], degrees=True).as_matrix()
            data[scene_name]['scene_info']['calibration'][key] = {"extrinsic":(R, t), "intrinsic": v_camera_intrinsic, 'camera_type': 'FisheyeCamera'}

    mmengine.dump(data, file)