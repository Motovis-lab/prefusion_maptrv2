import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import pdb
import open3d as o3d
import mmcv
import mmengine
import pickle
from rich.progress import track
import os.path as op
from collections import defaultdict
from copy import deepcopy
import numpy as np
import json
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from pathlib import Path as P
from typing import List, Tuple, Union
import torch
from mmengine import Config
from contrib.fastbev_det.models.utils.virtual_camera import render_image, PerspectiveCamera, FisheyeCamera, create_virtual_perspective_camera, \
    pcd_lidar_point, read_pcd_lidar, load_point_cloud, render_image_with_src_camera_points
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil



def convert_virtual_camera(src_camear_root, save_img_root, save_mask_root, real_cam_model, v_cam_paramter, calib_back, W=1280, H=960):
    src_image = mmcv.imread(src_camear_root, channel_order="bgr")
    R = Rotation.from_euler('xyz', angles=v_cam_paramter, degrees=True).as_matrix()
    # R = Quaternion(v_cam_paramter).rotation_matrix
    t = [0,0,0]
    v_cam_rmatrix = R
    v_cam_t = np.array(calib_back['extrinsic'][:3]).reshape(3)
    if 'FISHEYE' not in save_img_root:
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
        vcamera = PerspectiveCamera((W,H), (R, t), intrinsic)
    else:
        W = 1024
        H = 640
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 4
        intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)
        vcamera = FisheyeCamera((W,H), (R, t), intrinsic, fov=225)
    dst_image, dst_mask = render_image(src_image, real_cam_model, vcamera)
    mmcv.imwrite(dst_image, save_img_root)
    # mmcv.imwrite(dst_mask, save_mask_root)

    return v_cam_rmatrix, v_cam_t, intrinsic, src_image, real_cam_model, vcamera

def lidar_point2depth(v_cam_rmatrix, v_cam_t, src_image, real_cam_model, vcamera, lidar_point_path, dst_lidar_hpr, lidar_val_mask, v_camera_depth_root):
    cam_m = np.eye(4)
    cam_m[:3, :3] = v_cam_rmatrix.T
    cam_m[:3, 3] = -(v_cam_rmatrix.T) @ v_cam_t.reshape(3)

    hpr_points = o3d.io.read_point_cloud(str(lidar_point_path))
    _, pt_map = hpr_points.hidden_point_removal(camera_location=v_cam_t, radius=2000)
    hpr_lidar_point = np.asarray(hpr_points.select_by_index(pt_map).points).T
    hpr_lidar_point = np.concatenate([hpr_lidar_point, np.ones((1, hpr_lidar_point.shape[1]))], axis=0)
    hpr_lidar_point = cam_m @ deepcopy(hpr_lidar_point)
    # o3d.io.write_point_cloud("/ssd4/home/wuhan/MatrixVT_tda4/tmp/hpr_lidar_point.pcd", hpr_lidar_point)
    
    mask_matrix = np.eye(4)
    if dst_lidar_hpr:
        mask_matrix[:3, :3] = Quaternion(axis=[0,1,0], angle=(dst_lidar_hpr[0])*np.pi/180).rotation_matrix
        hpr_lidar_point = mask_matrix @ hpr_lidar_point
        if dst_lidar_hpr[1]=='b':
            x = hpr_lidar_point[0, ...] > 0
        elif dst_lidar_hpr[1]=='s':
            x = hpr_lidar_point[0, ...] < 0
        if dst_lidar_hpr[2]=='b':
            z = hpr_lidar_point[2, ...] > 0
        elif dst_lidar_hpr[2]=='s':
            z = hpr_lidar_point[2, ...] < 0
        mask_cam = np.logical_and(x, z)
        hpr_lidar_point = hpr_lidar_point[..., mask_cam]
        hpr_lidar_point = mask_matrix.T @ hpr_lidar_point

    dst_image, dst_img_w_point, dst_mask, depth = render_image_with_src_camera_points(src_image, real_cam_model, vcamera, hpr_lidar_point, return_depth=True)
    # plt.imshow(dst_img_w_point[..., ::-1])
    # plt.show()
    sensor_root_tmp = "./data/pretrain_data/"
    lidar_mask = mmcv.imread(sensor_root_tmp + lidar_val_mask)[..., 0]
    depth[lidar_mask==0] = -1
    np.savez_compressed(v_camera_depth_root, depth=depth.astype(np.float16))


def single_lidar_process(lidar1_filename, save_root):
    cams_lidar_point = np.asarray(o3d.io.read_point_cloud(str(lidar1_filename)).points).T
    cams_lidar_point_ = np.ones((4, cams_lidar_point.shape[1]))
    cams_lidar_point[1, :] += (3.816862089990445 - 2.436862089990445)
    cams_lidar_point_[:3, :] = cams_lidar_point
    R_nus = np.eye(4)
    R_nus[:3, :3] = Rotation.from_euler("XYZ", angles=(0,0,90), degrees=True).as_matrix()
    lidar_point = R_nus.T @ deepcopy(cams_lidar_point_)
    
    pcd_lidar_point(save_root, lidar_point.T)


if __name__ == "__main__":
    # ---args
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('scene_name', help='train config file path')
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    scene_infos = defaultdict()
    sphere_flip_radius = 1000
    max_sweeps = 20
    
    cameras_v = {
        "VCAMERA_PERSPECTIVE_FRONT":(-90, 0, -90),
        "VCAMERA_PERSPECTIVE_FRONT_LEFT":(-90, 0, -45),
        "VCAMERA_PERSPECTIVE_FRONT_RIGHT":(-90, 0, -135),
        "VCAMERA_PERSPECTIVE_BACK":(-90, 0, 90),
        "VCAMERA_PERSPECTIVE_BACK_LEFT":(-90, 0, 45),
        "VCAMERA_PERSPECTIVE_BACK_RIGHT":(-90, 0, 135),
        "VCAMERA_FISHEYE_FRONT":(-120, 0, -90),
        "VCAMERA_FISHEYE_LEFT":(-135, 0, 0),
        "VCAMERA_FISHEYE_RIGHT":(-135, 0, -180),
        "VCAMERA_FISHEYE_BACK":(-120, 0, 90)
    }
    cameras_hpr = {
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
    
    align_real_v = {"camera8":["VCAMERA_FISHEYE_FRONT", "VCAMERA_PERSPECTIVE_FRONT"],
                    "camera5":["VCAMERA_PERSPECTIVE_FRONT_LEFT", "VCAMERA_PERSPECTIVE_BACK_LEFT", "VCAMERA_FISHEYE_LEFT"],
                    "camera1":["VCAMERA_PERSPECTIVE_BACK", "VCAMERA_FISHEYE_BACK"],
                    'camera11':["VCAMERA_PERSPECTIVE_FRONT_RIGHT", "VCAMERA_PERSPECTIVE_BACK_RIGHT", "VCAMERA_FISHEYE_RIGHT"],
    }

    scene_root = "./data/pretrain_data"
    # scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("2023*") if p.is_dir()]
    scene_names = [args.scene_name]
    print("=="*60)
    print(scene_names)
    print("=="*60)
    for scene_name in scene_names:
        dump_root = scene_root
        calib_back = Config.fromfile(f"{scene_root}/calibration_back.yml")
        scene_root = os.path.join(scene_root,  scene_name)

        R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()
        calib_back['rig']['camera8']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera8']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera8']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera5']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera5']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera5']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera1']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera1']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera1']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera11']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera11']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera11']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        lidar1_cali_r = R_nus.T @ Rotation.from_quat(calib_back['rig']['lidar1']['extrinsic'][3:]).as_matrix()
        lidar1_cali_t = R_nus.T @ np.array(calib_back['rig']['lidar1']['extrinsic'][:3]).reshape(3)
        cameras_real = {
            "camera8":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera8']),
            "camera5":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera5']),
            "camera1":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera1']),
            "camera11":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera11'])
        }

        lidar_timestamps = list(sorted((P(scene_root) / P("undistort_static_merged_lidar1")).glob("*.pcd")))
        cam1_timestamps = list(sorted((P(scene_root) / P("camera/camera1")).glob("*.jpg")))
        cam5_timestamps = list(sorted((P(scene_root) / P("camera/camera5")).glob("*.jpg")))
        cam8_timestamps = list(sorted((P(scene_root) / P("camera/camera8")).glob("*.jpg")))
        cam11_timestamps = list(sorted((P(scene_root) / P("camera/camera11")).glob("*.jpg")))
        all_timestamps = sorted(list(set([x.stem for x in lidar_timestamps]) & set([x.stem for x in cam1_timestamps]) 
                              & set([x.stem for x in cam5_timestamps]) & set([x.stem for x in cam8_timestamps]) & set([x.stem for x in cam11_timestamps])))
        # all_timestamps = ['1693298281964']
        scene_info = {}
        scene_info["scene_info"] = {}
        scene_info["scene_info"]['camera_mask'] = {
            "VCAMERA_FISHEYE_BACK": "fisheye_ego_mask/VCAMERA_FISHEYE_BACK_MASK/1692759619664_ego_mask.jpg",
            "VCAMERA_FISHEYE_FRONT": "fisheye_ego_mask/VCAMERA_FISHEYE_FRONT_MASK/1692759619664_ego_mask.jpg",
            "VCAMERA_FISHEYE_LEFT": "fisheye_ego_mask/VCAMERA_FISHEYE_LEFT_MASK/1692759619664_ego_mask.jpg",
            "VCAMERA_FISHEYE_RIGHT": "fisheye_ego_mask/VCAMERA_FISHEYE_RIGHT_MASK/1692759619664_ego_mask.jpg",
            "VCAMERA_PERSPECTIVE_BACK_LEFT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_BACK_LEFT_MASK/1692759619664.jpg",
            "VCAMERA_PERSPECTIVE_BACK": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_BACK_MASK/1692759619664.jpg",
            "VCAMERA_PERSPECTIVE_BACK_RIGHT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_BACK_RIGHT_MASK/1692759619664.jpg",
            "VCAMERA_PERSPECTIVE_FRONT_LEFT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_FRONT_LEFT_MASK/1692759619664.jpg",
            "VCAMERA_PERSPECTIVE_FRONT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_FRONT_MASK/1692759619664.jpg",  # 从鱼眼前视虚拟出来的front，不是原始的camera6
            "VCAMERA_PERSPECTIVE_FRONT_RIGHT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_FRONT_RIGHT_MASK/1692759619664.jpg",
        }
        scene_info["scene_info"]['calibration'] = {}

        frame_info = {}
        timestamp_window = defaultdict(list)
        for timestamp in track(all_timestamps, "Process timestamp ... "):
            times_id = timestamp
            frame_info[times_id] = {}
            # convert real_cam to v_cam
            camera_image = {}
            camera_img_seg = {}
            camera_image_depth = {}
            lidar_point = {}
            src_lidar_root = P(scene_root) / P(f"undistort_static_merged_lidar1/{times_id}.pcd")
            save_lidar_point = f"{scene_root}/lidar/undistort_static_lidar1_model/{times_id}.pcd"
            P(save_lidar_point).parent.mkdir(parents=True, exist_ok=True)
            single_lidar_process(src_lidar_root, save_lidar_point)

            for real_cam_name, camera_model in cameras_real.items():
                camera_filename = f"{times_id}.jpg"                
                src_camera_root = P(scene_root) / P(f"camera/{real_cam_name}/{camera_filename}")
                src_seg_root = P(scene_root) / P(f"seg/fisheye_semantic_segmentation/{real_cam_name}/{times_id}.png")    
                
                for V_CAM in align_real_v[real_cam_name]:
                    v_camera_root = f"{scene_root}/camera/{V_CAM}/{camera_filename}"
                    # v_camera_mask_root = f"{scene_root}/camera/{V_CAM}_MASK/{camera_filename}"
                    P(v_camera_root).parent.mkdir(parents=True, exist_ok=True)
                    # P(v_camera_mask_root).parent.mkdir(parents=True, exist_ok=True)
                    v_camera_rmatrix, v_camera_t, v_camera_intrinsic, d_src_image, d_real_cam_model, d_vcamera = convert_virtual_camera(src_camera_root, v_camera_root, None, camera_model, cameras_v[V_CAM], calib_back.rig[real_cam_name])
                    scene_info["scene_info"]['calibration'][V_CAM] = {"extrinsic":(v_camera_rmatrix, v_camera_t), "intrinsic": v_camera_intrinsic}
                    camera_image[V_CAM] = f"{scene_name}/camera/{V_CAM}/{camera_filename}"
                    
                    v_camera_seg_root = f"{scene_root}/seg/fisheye_semantic_segmentation/{V_CAM}/{times_id}.png"
                    P(v_camera_seg_root).parent.mkdir(parents=True, exist_ok=True)
                    v_camera_rmatrix, v_camera_t, v_camera_intrinsic, d_src_image, d_real_cam_model, d_vcamera = convert_virtual_camera(src_seg_root, v_camera_seg_root, None, camera_model, cameras_v[V_CAM], calib_back.rig[real_cam_name])
                    camera_img_seg[V_CAM] = f"{scene_name}/seg/fisheye_semantic_segmentation/{V_CAM}/{times_id}.png"
                    
                    v_camera_depth_root = f"{scene_root}/depth/{V_CAM}/{times_id}.npz"
                    P(v_camera_depth_root).parent.mkdir(parents=True, exist_ok=True)
                    dst_lidar_hpr = cameras_hpr[V_CAM]
                    lidar_val_mask = scene_info["scene_info"]['camera_mask'][V_CAM]
                    lidar_point2depth(v_camera_rmatrix, v_camera_t, d_src_image, d_real_cam_model, d_vcamera, save_lidar_point, dst_lidar_hpr, lidar_val_mask, v_camera_depth_root)
                    camera_image_depth[V_CAM] = f"{scene_name}/depth/{V_CAM}/{times_id}.npz"

            R_t = np.eye(4)
            R_t[:3, :3] = R_nus
            frame_info[times_id] = {
                "camera_image": camera_image,
                "camera_image_seg": camera_img_seg, 
                "camera_image_depth": camera_image_depth
            }
            
        scene_info.update({'frame_info': frame_info})
    scene_infos[scene_name] = scene_info
    mmengine.dump(scene_infos, f"{dump_root}/mv_4d_infos_{scene_names[0]}.pkl")
    shutil.rmtree(scene_root / P("lidar"))
    shutil.rmtree(scene_root / P("undistort_static_merged_lidar1"))