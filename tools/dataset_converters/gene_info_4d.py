# s3://mv-4d-annotation/AutoCollaborator/manual_annotation/baidu500/20230823_110018/whole_scene/
# s3://mv-4d-annotation/data/multimodel_data_baidu/20230823_110018/3d_box/3d_box_sync

# manual_map_box
# manual_frame_box
# infer_box
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import pdb
sys.path.append(".")
sys.path.append("/home/wuhan/mtv4d/scripts/")
sys.path.append("/home/wuhan/mtv4d/")
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

from mtv4d.annos_4d.helper import (  # type: ignore
    check_whether_points_behind_camera,
    CameraParam,
    torch_pool,
    transform_points_ego_to_lidar,
    write_json_from_list,
    read_json_to_list,
    anno_box_to_7_values_box,
    get_times,
    get_sync_filename,
    read_cal_data,
    generate_lidar_mask,
)

from mtv4d.annos_4d.misc import read_ego_paths  # type: ignore
from scripts.generate_4d_frame_clean import generate_DS4d_from_4dMapJson, solve_ds_occlusion_sub_id, generate_4d_frame_json_data, read_ts_json # type: ignore


label_to_class = {
    "101": "Box_truck",
    "102": "Truck",
    "103": "Car",
    "104": "Van",
    "105": "Bus",
    "106": "Engineering_vehicle",
    "201": "Pedestrian",
    "202": "Cyclist",
    "301": "Bicycle",
    "100": "DontCare",
}


def generate_labels_scene_from_4dMapjson(scene_root, Twes):
    (
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    ) = generate_DS4d_from_4dMapJson(op.join(scene_root, "4d_anno_infos/annos.json"), Twes)  # load进来转成DS4D
    print('generate finish')
    (
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    ) = solve_ds_occlusion_sub_id(   
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    )

    # generate frame info txt
    output_json_frame_dlist = generate_4d_frame_json_data(dn_boxes_vis_dict_ts2id, map_boxes_vis_dict_ts2id, map_polylines_vis_dict_ts2id)
    # for ts, frames_labels in tqdm(output_json_frame_dlist.items(), desc='generating frame json'):
    #     write_json_from_list(
    #         frames_labels, op.join(scene_root, f"4d_anno_infos/4d_anno_infos_frame/frames_labels_all/{int(ts)}.json"), format_float=True, indent=4
    #     )
    return output_json_frame_dlist

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
        vcamera = FisheyeCamera((W,H), (R, t), intrinsic, fov=180)
    dst_image, dst_mask = render_image(src_image, real_cam_model, vcamera)
    mmcv.imwrite(dst_image, save_img_root)
    # mmcv.imwrite(dst_mask, save_mask_root)

    return v_cam_rmatrix, v_cam_t, intrinsic, src_image, real_cam_model, vcamera

def lidar_point2depth(v_cam_rmatrix, v_cam_t, src_image, real_cam_model, vcamera, lidar_point_path, dst_lidar_hpr, lidar_val_mask, v_camera_depth_root):
    cam_m = np.eye(4)
    cam_m[:3, :3] = v_cam_rmatrix.T
    cam_m[:3, 3] = -(v_cam_rmatrix.T) @ v_cam_t.reshape(3)

    hpr_points = o3d.io.read_point_cloud(lidar_point_path)
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
    sensor_root_tmp = "/home/wuhan/mv_4d_data/"
    lidar_mask = mmcv.imread(sensor_root_tmp + lidar_val_mask)[..., 0]
    depth[lidar_mask==0] = -1
    np.savez_compressed(v_camera_depth_root, depth=depth.astype(np.float16))


def dummy_lidar_process(calib_back):
    cam1_lidar = load_point_cloud(P("/ssd3/data/mv4d_sample/20230823_110018/lidar/undistort_static_lidar1/000007_1692759619664.bin"), discard_intensity=False)
    # cam1_lidar = load_point_cloud(P("/ssd4/home/wuhan/MatrixVT_tda4/test/show_data/lidar_data/cam5/002395_1698826056864.bin"), discard_intensity=False)
    cams_lidar_point = np.concatenate([cam1_lidar], 0).T

    lidar1_calibration = calib_back['rig']['lidar1']['extrinsic']
    lidar1_R_T = np.eye(4)  
    lidar1_R_T[:3, :3] = Quaternion([lidar1_calibration[-1], lidar1_calibration[3], lidar1_calibration[4], lidar1_calibration[5]]).rotation_matrix
    lidar1_R_T[:3, 3]  = lidar1_calibration[:3]
    intensity = deepcopy(cams_lidar_point[3, :])
    cams_lidar_point[3, :] = 1
    # cams_lidar_point = lidar1_R_T @ cams_lidar_point
    R_nus = np.eye(4)
    R_nus[:3, :3] = Rotation.from_euler("XYZ", angles=(0,0,90), degrees=True).as_matrix()
    lidar_point = R_nus.T @ (lidar1_R_T @ cams_lidar_point)
    lidar_point[3, :] = intensity
    # pcd_lidar_point("/ssd3/data/mv4d_sample/20230823_110018/lidar/undistort_lidar1/1692759619664.pcd", cams_lidar_point.T)
    pcd_lidar_point("./test.pcd", lidar_point.T)

def single_lidar_process(lidar1_filename, lidar2_filename, lidar3_filename, save_root):
    try:
        cams_lidar1_point = read_pcd_lidar(lidar1_filename).T
    except:
        cams_lidar1_point = np.zeros((4, 0))
    try:
        cams_lidar2_point = read_pcd_lidar(lidar2_filename).T
    except:
        cams_lidar2_point = np.zeros((4, 0))
    try:
        cams_lidar3_point = read_pcd_lidar(lidar3_filename).T
    except:
        cams_lidar3_point = np.zeros((4, 0))
    cams_lidar_point = np.concatenate([cams_lidar1_point, cams_lidar2_point, cams_lidar3_point], axis=-1)
    cams_lidar_point[1, :] += (3.816862089990445 - 2.436862089990445)
    intensity = deepcopy(cams_lidar_point[3, :])
    cams_lidar_point[3, :] = 1
    R_nus = np.eye(4)
    R_nus[:3, :3] = Rotation.from_euler("XYZ", angles=(0,0,90), degrees=True).as_matrix()
    lidar_point = R_nus.T @ deepcopy(cams_lidar_point)
    lidar_point[3, :] = intensity
    pcd_lidar_point(save_root, lidar_point.T)

def create_moving_object(all_frames_infos, timestamps):
    R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()
    moving_objects_track_id_trajectory = {} # defaultdict(list)
    for timestamps in timestamps:
        for ann in all_frames_infos[timestamps]:
            if ann['geometry_type'] == 'box3d' and int(ann['obj_track_id'].split('_')[0]) < 10000:
                pose_xyz = R_nus.T @ np.array(ann['geometry_w']["pos_xyz"]).reshape(3, 1)
                rot_xyz = R_nus.T @ Rotation.from_euler("XYZ", angles=ann['geometry_w']["rot_xyz"], degrees=False).as_matrix()
                if ann['obj_track_id'] in moving_objects_track_id_trajectory:
                    moving_objects_track_id_trajectory[ann['obj_track_id']]['timestamps'].append(int(ann['timestamp']))
                    moving_objects_track_id_trajectory[ann['obj_track_id']]['attr'].append(ann['obj_attr'])
                    moving_objects_track_id_trajectory[ann['obj_track_id']]['pose'].append((pose_xyz, rot_xyz))
                else:
                    moving_objects_track_id_trajectory[ann['obj_track_id']] = {
                        'class': ann['obj_type'],
                        'timestamps':[int(ann['timestamp'])],
                        'attr':[ann['obj_attr']],
                        'pose':[(pose_xyz, rot_xyz)]
                    }
    return moving_objects_track_id_trajectory


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
    
    align_real_v = {"camera8":["VCAMERA_FISHEYE_FRONT"],
                    "camera5":["VCAMERA_PERSPECTIVE_FRONT_LEFT", "VCAMERA_PERSPECTIVE_BACK_LEFT", "VCAMERA_FISHEYE_LEFT"],
                    "camera1":["VCAMERA_PERSPECTIVE_BACK", "VCAMERA_FISHEYE_BACK"],
                    'camera11':["VCAMERA_PERSPECTIVE_FRONT_RIGHT", "VCAMERA_PERSPECTIVE_BACK_RIGHT", "VCAMERA_FISHEYE_RIGHT"],
                    'camera6':["VCAMERA_PERSPECTIVE_FRONT"]
    }

    scene_root = "/home/wuhan/mv_4d_data/"
    # scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("2023*") if p.is_dir()]
    scene_names = [args.scene_name]
    print("=="*60)
    print(scene_names)
    print("=="*60)
    for scene_name in scene_names:
        dump_root = scene_root
        scene_root = os.path.join(scene_root,  scene_name)

        calib_back = Config.fromfile(f"{scene_root}/calibration_back.yml")
        R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()
        calib_back['rig']['camera8']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera8']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera8']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera5']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera5']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera5']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera1']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera1']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera1']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera11']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera11']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera11']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_back['rig']['camera6']['extrinsic'] = [*((R_nus.T @ np.array(calib_back['rig']['camera6']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_back['rig']['camera6']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        lidar1_cali_r = R_nus.T @ Rotation.from_quat(calib_back['rig']['lidar1']['extrinsic'][3:]).as_matrix()
        lidar1_cali_t = R_nus.T @ np.array(calib_back['rig']['lidar1']['extrinsic'][:3]).reshape(3)
        cameras_real = {
            "camera8":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera8']),
            "camera5":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera5']),
            "camera1":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera1']),
            "camera11":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera11']),
            "camera6":FisheyeCamera.init_from_motovis_cfg(calib_back.rig['camera6'])
        }

        ts_4d_rel_path = "4d_anno_infos/ts.json"
        trajectory_prefix = "trajectory"
        calib = read_cal_data(op.join(scene_root, "calibration_center.yml"))
        traj_p = op.join(scene_root, f"{trajectory_prefix}.txt")
        Twes, _ = read_ego_paths(traj_p)  
        timestamps, tmp_sensors_timestamps = read_ts_json(op.join(scene_root, "4d_anno_infos/ts_full.json"))
        sensors_timestamps = deepcopy(tmp_sensors_timestamps)
        for key in tmp_sensors_timestamps:
            if not all(tmp_sensors_timestamps[key].values()):
                sensors_timestamps.pop(key)
                timestamps.remove(key)
        assert all([i in Twes.keys() for i in timestamps])  # TODO: 确定是否内含这个assert
        all_frames_infos = generate_labels_scene_from_4dMapjson(scene_root, Twes)
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
            "VCAMERA_PERSPECTIVE_FRONT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_FRONT_MASK/1692759619664_ego_mask.jpg",
            "VCAMERA_PERSPECTIVE_FRONT_RIGHT": "fisheye_ego_mask/VCAMERA_PERSPECTIVE_FRONT_RIGHT_MASK/1692759619664.jpg",
        }
        scene_info["scene_info"]['calibration'] = {}
        scene_info["scene_info"]['moving_objects_track_id_trajectory'] = create_moving_object(all_frames_infos, timestamps)
        # meta_info 
        scene_info['meta_info'] = {
            "description": "some information with this scene",
            "space_range":{
                "map": [36, -12, -12, 12, 10, -10],
                "det": [36, -12, -12, 12, 10, -10],
                "occ": [36, -12, -12, 12, 10, -10]
            },
            "time_range": 2
        }
        # dummy_lidar_process(calib_back)
        frame_info = {}
        timestamp_window = defaultdict(list)
        for timestamp in track(timestamps, "Process timestamp ... "):
            times_id = str(int(timestamp))
            frame_info[times_id] = {}
            # convert real_cam to v_cam
            camera_image = {}
            camera_image_depth = {}
            boxes_3d = []
            polyline_3d = []
            lidar_point = {}
            
            scene_info["scene_info"]['calibration'].update({'lidar1':(lidar1_cali_r, lidar1_cali_t)})
            # process lidar
            src_lidar1_point = f"{scene_root}/lidar/undistort_static_lidar1/{times_id}.pcd"
            src_lidar2_point = f"{scene_root}/lidar/undistort_static_lidar2/{times_id}.pcd"
            src_lidar3_point = f"{scene_root}/lidar/undistort_static_lidar3/{times_id}.pcd"
            save_lidar_point = f"{scene_root}/lidar/undistort_static_lidar1_model/{times_id}.pcd"
            P(save_lidar_point).parent.mkdir(parents=True, exist_ok=True)
            single_lidar_process(src_lidar1_point, src_lidar2_point, src_lidar3_point, save_lidar_point)
            lidar_point['lidar1'] = f"{scene_name}/lidar/undistort_static_lidar1_model/{times_id}.pcd"
            if "lidar1" in timestamp_window:
                if len(timestamp_window['lidar1']) <= max_sweeps:
                    timestamp_window['lidar1'].append(f"{scene_name}/lidar/undistort_static_lidar1_model/{times_id}.pcd")
                else:
                    timestamp_window['lidar1'].pop(0)
                    timestamp_window['lidar1'].append(f"{scene_name}/lidar/undistort_static_lidar1_model/{times_id}.pcd")
            else:
                timestamp_window['lidar1'] = [f"{scene_name}/lidar/undistort_static_lidar1_model/{times_id}.pcd"]

            for real_cam_name, camera_model in cameras_real.items():
                camera_filename = sensors_timestamps[timestamp][real_cam_name]
                src_camera_root = f"{scene_root}/camera/{real_cam_name}/{camera_filename}"
                for V_CAM in align_real_v[real_cam_name]:
                    v_camera_root = f"{scene_root}/camera/{V_CAM}/{camera_filename}"
                    v_camera_mask_root = f"{scene_root}/camera/{V_CAM}_MASK/{camera_filename}"
                    P(v_camera_root).parent.mkdir(parents=True, exist_ok=True)
                    P(v_camera_mask_root).parent.mkdir(parents=True, exist_ok=True)
                    v_camera_rmatrix, v_camera_t, v_camera_intrinsic, d_src_image, d_real_cam_model, d_vcamera = convert_virtual_camera(src_camera_root, v_camera_root, v_camera_mask_root, camera_model, cameras_v[V_CAM], calib_back.rig[real_cam_name])
                    scene_info["scene_info"]['calibration'][V_CAM] = {"extrinsic":(v_camera_rmatrix, v_camera_t), "intrinsic": v_camera_intrinsic}
                    # if "FISHEYE" in V_CAM:
                    #     scene_info["scene_info"]['camera_mask'][V_CAM] = f"{scene_name}/fisheye_ego_mask/{V_CAM}_MASK/{timestamp}_ego_mask.jpg"
                    camera_image[V_CAM] = f"{scene_name}/camera/{V_CAM}/{camera_filename}"
                    if V_CAM in timestamp_window:
                        if len(timestamp_window[V_CAM])<=max_sweeps:
                            timestamp_window[V_CAM].append(f"{scene_name}/camera/{V_CAM}/{camera_filename}")
                        else:
                            timestamp_window[V_CAM].pop(0)
                            timestamp_window[V_CAM].append(f"{scene_name}/camera/{V_CAM}/{camera_filename}")
                    else:
                        timestamp_window[V_CAM] = [f"{scene_name}/camera/{V_CAM}/{camera_filename}"]
                    
                    v_camera_depth_root = f"{scene_root}/depth/{V_CAM}/{times_id}.npz"
                    P(v_camera_depth_root).parent.mkdir(parents=True, exist_ok=True)
                    dst_lidar_hpr = cameras_hpr[V_CAM]
                    lidar_val_mask = scene_info["scene_info"]['camera_mask'][V_CAM]
                    lidar_point2depth(v_camera_rmatrix, v_camera_t, d_src_image, d_real_cam_model, d_vcamera, save_lidar_point, dst_lidar_hpr, lidar_val_mask, v_camera_depth_root)
                    camera_image_depth[V_CAM] = f"{scene_name}/depth/{V_CAM}/{times_id}.npz"

            for ann in all_frames_infos[timestamp]:
                if ann['geometry_type'] == 'box3d':
                    for sensor_name, lidar_point_num in ann['visibility'].items():
                        if "camera" in sensor_name and lidar_point_num > 0:
                            pos_xyz = np.array(ann['geometry_e']['pos_xyz']).reshape(3, 1)
                            pos_xyz[1, :] += (3.816862089990445 - 2.436862089990445)
                            box = {
                                "class": ann['obj_type'],
                                "attr": ann['obj_attr'],
                                "size": ann['geometry_e']['scale_xyz'],
                                'rotation': R_nus.T @ Rotation.from_euler("XYZ",angles=ann['geometry_e']['rot_xyz'], degrees=False).as_matrix(),
                                'translation': (R_nus.T @ pos_xyz).reshape(3),
                                "track_id": ann['obj_track_id'],
                                "velocity": (R_nus.T @ np.array(ann['velocity']).reshape(3, 1)).reshape(3)
                            }
                            boxes_3d.append(box)
                            break
                elif ann['geometry_type'] == 'polyline3d':
                    polyline_ = np.array(ann['geometry']).transpose(1,0)
                    polyline_[1, :] += (3.816862089990445 - 2.436862089990445)
                    polyline = {
                        'class': ann['obj_type'],
                        'attr': ann['obj_attr'],
                        "points": (R_nus.T @ polyline_).transpose(1,0),
                    }
                    polyline_3d.append(polyline)
            sdf_about = {
                'occ_bev': f"{scene_name}/occ_map/occ_map/{times_id}.png",
                'height_bev': f"{scene_name}/ground_height_map/ground_height_map/{times_id}.tif",
                'sdf_bev': f"{scene_name}/sdf/sdf1/{times_id}.tif",
                'occ_sdf_3d': None
            }
            R_t = np.eye(4)
            R_t[:3, :3] = R_nus
            frame_info[times_id] = {
                "camera_image": camera_image,
                "3d_boxes": boxes_3d,
                "3d_polylines": polyline_3d,
                "timestamp_window": deepcopy(timestamp_window),
                "lidar_points": lidar_point,
                "ego_pose": {
                    "rotation": (R_t.T @ Twes[timestamp])[:3, :3],
                    'translation': (R_t.T @ Twes[timestamp])[:3, 3]
                },
                "camera_image_seg": None, 
                "camera_image_depth": None,
                "occ_sdf": sdf_about,
                "camera_image_depth": camera_image_depth
            }
        scene_info.update({'frame_info': frame_info})
    scene_infos[scene_name] = scene_info
    mmengine.dump(scene_infos, f"{dump_root}/mv_4d_infos_{scene_names[0]}.pkl")