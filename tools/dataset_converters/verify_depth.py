from functools import partial
import torch
import numpy as np
import open3d as o3d
import mmengine
import random
import sys
sys.path.append(".")
from contrib.fastbev_det.models.utils.virtual_camera import get_unproj_func

cameras_tmp = {
    "VCAMERA_PERSPECTIVE_FRONT":(45, 'b', 'b'),
    "VCAMERA_PERSPECTIVE_FRONT_LEFT":(45, 'b', 'b'),
    "VCAMERA_PERSPECTIVE_FRONT_RIGHT":(-45, 's', 'b'),
    "VCAMERA_PERSPECTIVE_BACK":(45, 'b', 'b'),
    "VCAMERA_PERSPECTIVE_BACK_LEFT":(-45, 's', 'b'),
    "VCAMERA_PERSPECTIVE_BACK_RIGHT":(45, 'b', 'b'),
    "VCAMERA_FISHEYE_FRONT":None,
    "VCAMERA_FISHEYE_LEFT":None,
    "VCAMERA_FISHEYE_RIGHT":None,
    "VCAMERA_FISHEYE_BACK":None
}

def unproject_points_from_image_to_camera_fisheye(resolution, intrinsic, fov):
    W, H = resolution
    cx, cy, fx, fy, p0, p1, p2, p3 = intrinsic
    unproj_func = get_unproj_func(p0, p1, p2, p3, fov=fov)
    
    uu, vv = np.meshgrid(
        np.linspace(0, W - 1, W), 
        np.linspace(0, H - 1, H)
    )
    x_distorted = (uu - cx) / fx
    y_distorted = (vv - cy) / fy
    
    # r_distorted = theta_distorted
    r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
    r_distorted[r_distorted < 1e-5] = 1e-5
    theta = unproj_func(r_distorted)
    # theta = np.clip(theta, - 0.5 * self.fov * np.pi / 180, 0.5 * self.fov * np.pi / 180)
    camera_mask = np.float32(np.abs(theta * 180 / np.pi) < fov / 2)

    # get camera coords by ray intersecting with a sphere in image-style (x-y-z right-down-forward)
    r_distorted[r_distorted < 1e-5] = 1e-5
    dd = np.sin(theta)
    xx = x_distorted * dd / r_distorted
    yy = y_distorted * dd / r_distorted
    zz = np.cos(theta)
    
    camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)

    return camera_points

def unproject_points_from_image_to_camera_perspective(resolution, intrinsic, depth_mode='z'):
    W, H = resolution
    cx, cy, fx, fy = intrinsic
    assert depth_mode in ['z', 'd']
    
    uu, vv = np.meshgrid(
        np.linspace(0, W - 1, W), 
        np.linspace(0, H - 1, H)
    )
    # get camera coords by ray intersecting with a z-plane in image-style (x-y-z right-down-forward)
    xx = (uu - cx) / fx
    yy = (vv - cy) / fy
    zz = np.ones_like(uu)

    camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)
    if depth_mode in ['d']:
        # 正常情况下到平面的距离z（对应单位向量为1）,到相机光心的距离为d,如果同样情况下到平面的距离应该为1/d
        # z = 1*z z = x*d  z/d = x  1/(norm(d)) = x  对应到光心d时的缩放量
        camera_points /= np.linalg.norm(camera_points, axis=0)

    return camera_points


data_root = "/ssd/home/wuhan/prefusion/data/mv_4d_data/"
data = mmengine.load(f"{data_root}mv_4d_infos.pkl")

scene_ids = random.choices(list(data.keys()), k=1)

for scene_id in scene_ids:
    frame_ids = random.choices(list(data[scene_id]['frame_info'].keys()), k=3)
    for frame_id in frame_ids:
        for cam in list(cameras_tmp.keys()):
            calibration = data[scene_id]['scene_info']['calibration'][cam]
            depth_path = data[scene_id]['frame_info'][frame_id]['camera_image_depth'][cam]

            depth = np.load(f"{data_root}{scene_id}/depth/{cam}/{frame_id}.npz")['depth']
            depth[depth==-1] = 0 
            resolution = depth.shape[::-1]
            if "FISH" in cam:
                # depth = torch.max_pool2d(torch.from_numpy(depth.astype(np.float32)[None, ...]), kernel_size=(4,4), stride=(4,4)).numpy()[0]
                # depth = mmcv.imresize(depth.astype(np.float32), (256, 160), interpolation="nearest")
                # depth = mmcv.imresize(depth.astype(np.float32), (1024, 640), interpolation="nearest")
                camera_points = unproject_points_from_image_to_camera_fisheye(resolution, calibration['intrinsic'], 180)
            else:
                camera_points = unproject_points_from_image_to_camera_perspective(resolution, calibration['intrinsic'], depth_mode='d')
            points = depth.reshape(1, -1) * camera_points
            # ground = points[1] > 0
            # points = points[:, ground]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.T)
            mmengine.mkdir_or_exist(f"work_dirs/mv_4d_matrix/verify_depth/{cam}/")
            o3d.io.write_point_cloud(f'work_dirs/mv_4d_matrix/verify_depth/{cam}/{frame_id}.pcd', pcd)
