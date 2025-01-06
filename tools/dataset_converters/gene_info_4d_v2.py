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
sys.path.append("/mnt/ssd1/wuhan/mtv4d/scripts/")
sys.path.append("/mnt/ssd1/wuhan/mtv4d/")
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


align_real_v = {"camera8": "VCAMERA_FISHEYE_FRONT",
                "camera5": "VCAMERA_FISHEYE_LEFT",
                "camera1": "VCAMERA_FISHEYE_BACK",
                'camera11': "VCAMERA_FISHEYE_RIGHT"
}

parm_cameras_v = {
    "VCAMERA_FISHEYE_FRONT":(-120, 0, -90),
    "VCAMERA_FISHEYE_LEFT":(-135, 0, 0),
    "VCAMERA_FISHEYE_RIGHT":(-135, 0, -180),
    "VCAMERA_FISHEYE_BACK":(-120, 0, 90)
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

def convert_virtual_camera(src_camear_root, save_img_root, save_mask_root, real_cam_model, v_cam_paramter, calib, W=1280, H=960):
    src_image = mmcv.imread(src_camear_root, channel_order="bgr")
    R = Rotation.from_euler('xyz', angles=v_cam_paramter, degrees=True).as_matrix()
    # R = Quaternion(v_cam_paramter).rotation_matrix
    t = [0,0,0]
    v_cam_rmatrix = R
    v_cam_t = np.array(calib['extrinsic'][:3]).reshape(3)
    if 'FISHEYE' not in save_img_root:
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
        vcamera = PerspectiveCamera((W,H), (R, t), intrinsic)
    else:
        W = 768
        H = 512
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 4
        intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)
        vcamera = FisheyeCamera((W,H), (R, t), intrinsic, fov=180)
    dst_image, dst_mask = render_image(src_image, real_cam_model, vcamera)
    mmcv.imwrite(dst_image, save_img_root)
    # mmcv.imwrite(dst_mask, save_mask_root)

    return v_cam_rmatrix, v_cam_t, intrinsic, src_image, real_cam_model, vcamera


def single_lidar_process(lidar1_filename, save_root):
    try:
        cams_lidar1_point = read_pcd_lidar(lidar1_filename).T
    except:
        cams_lidar1_point = np.zeros((4, 0))
    cams_lidar_point = np.concatenate([cams_lidar1_point], axis=-1)
    # cams_lidar_point[1, :] += (3.816862089990445 - 2.436862089990445)
    intensity = deepcopy(cams_lidar_point[3, :])
    cams_lidar_point[3, :] = 1
    R_nus = np.eye(4)
    R_nus[:3, :3] = Rotation.from_euler("XYZ", angles=(0,0,90), degrees=True).as_matrix()
    lidar_point = R_nus.T @ deepcopy(cams_lidar_point)
    lidar_point[3, :] = intensity
    ori_pcd_lidar_point(save_root, lidar_point.T)

def ori_pcd_lidar_point(save_path, lidar_points):
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    points_intensities = lidar_points[:, 3][:, None]
    points_positions = lidar_points[:,:3]
    lidar_map = o3d.t.geometry.PointCloud(device)
    
    lidar_map.point.positions = o3d.core.Tensor(points_positions , dtype, device)
    lidar_map.point.intensity = o3d.core.Tensor(points_intensities , dtype, device)
    
    o3d.t.io.write_point_cloud(str(save_path), lidar_map)

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

    all_cameras = [
        "camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8", "camera11", "camera12", "camera15"
    ]
    fish_cameras = [
        "camera1", "camera5", "camera8", "camera11"
    ]

    scene_root = "./data/MV4D_12V3L/"
    # scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("2023*") if p.is_dir()]
    scene_names = [args.scene_name]
    print("=="*60)
    print(scene_names)
    print("=="*60)
    for scene_name in scene_names:
        dump_root = scene_root
        scene_root = os.path.join(scene_root,  scene_name)
        # calib_center and calib_back are same without translation
        
        calib_center = Config.fromfile(f"{scene_root}/calibration_center.yml")
        R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()
        lidar1_cali_r = R_nus.T @ Rotation.from_quat(calib_center['rig']['lidar1']['extrinsic'][3:]).as_matrix()
        lidar1_cali_t = R_nus.T @ np.array(calib_center['rig']['lidar1']['extrinsic'][:3]).reshape(3)

        calib_center['rig']['camera8']['extrinsic'] = [*((R_nus.T @ np.array(calib_center['rig']['camera8']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_center['rig']['camera8']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_center['rig']['camera5']['extrinsic'] = [*((R_nus.T @ np.array(calib_center['rig']['camera5']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_center['rig']['camera5']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_center['rig']['camera1']['extrinsic'] = [*((R_nus.T @ np.array(calib_center['rig']['camera1']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_center['rig']['camera1']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        calib_center['rig']['camera11']['extrinsic'] = [*((R_nus.T @ np.array(calib_center['rig']['camera11']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(calib_center['rig']['camera11']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        
        cameras_real = {
            "camera8":FisheyeCamera.init_from_motovis_cfg(calib_center.rig['camera8']),
            "camera5":FisheyeCamera.init_from_motovis_cfg(calib_center.rig['camera5']),
            "camera1":FisheyeCamera.init_from_motovis_cfg(calib_center.rig['camera1']),
            "camera11":FisheyeCamera.init_from_motovis_cfg(calib_center.rig['camera11'])
        }

        ts_4d_rel_path = "4d_anno_infos/ts.json"
        trajectory_prefix = "trajectory"
        calib = read_cal_data(op.join(scene_root, "calibration_center.yml")) # dummy load
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
            "camera1": "ego_mask/camera1.png",
            "camera2": "ego_mask/camera2.png",
            "camera3": "ego_mask/camera3.png",
            "camera4": "ego_mask/camera4.png",
            "camera5": "ego_mask/camera5.png",
            "camera6": "ego_mask/camera6.png",
            "camera7": "ego_mask/camera7.png",
            "camera8": "ego_mask/camera8.png",
            "camera11": "ego_mask/camera11.png",
            "camera12": "ego_mask/camera12.png",
            "camera15": "ego_mask/camera15.png",
            "VCAMERA_FISHEYE_FRONT": "ego_mask/VCAMERA_FISHEYE_FRONT.png",
            "VCAMERA_FISHEYE_BACK": "ego_mask/VCAMERA_FISHEYE_BACK.png",
            "VCAMERA_FISHEYE_LEFT": "ego_mask/VCAMERA_FISHEYE_LEFT.png",
            "VCAMERA_FISHEYE_RIGHT": "ego_mask/VCAMERA_FISHEYE_RIGHT.png",
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
        
        frame_info = {}
        timestamp_window = defaultdict(list)
        for timestamp in track(timestamps, "Process timestamp ... "):
            times_id = str(int(timestamp))
            if times_id[-2:] == "74":
                continue
            frame_info[times_id] = {}
            # convert real_cam to v_cam
            camera_image = {}
            boxes_3d = []
            polyline_3d = []
            lidar_point = {}
            src_lidar_path = f"{scene_root}/lidar/undistort_static_merged_lidar1/{times_id}.pcd"
            dst_lidar_path = f"{scene_root}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd"
            P(f"{scene_root}/lidar/undistort_static_merged_lidar1_model/").mkdir(parents=True, exist_ok=True)
            # process lidar
            single_lidar_process(src_lidar_path, dst_lidar_path)
            scene_info["scene_info"]['calibration'].update({'lidar1':(lidar1_cali_r, lidar1_cali_t)})
            lidar_point['lidar1'] = f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd"
            if "lidar1" in timestamp_window:
                if len(timestamp_window['lidar1']) <= max_sweeps:
                    timestamp_window['lidar1'].append(f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd")
                else:
                    timestamp_window['lidar1'].pop(0)
                    timestamp_window['lidar1'].append(f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd")
            else:
                timestamp_window['lidar1'] = [f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd"]

            for camera_name in all_cameras:
                camera_filename = sensors_timestamps[timestamp][camera_name]
                
                if camera_name in fish_cameras:
                    fish_camera = FisheyeCamera.init_from_motovis_cfg(calib_center['rig'][camera_name])
                    extrinsic_r = Rotation.from_quat(calib_center['rig'][camera_name]['extrinsic'][3:]).as_matrix()
                    extrinsic_t = np.array(calib_center['rig'][camera_name]['extrinsic'][:3]).reshape(3)
                    scene_info["scene_info"]['calibration'][camera_name] = {"extrinsic":(extrinsic_r, extrinsic_t), 
                                                                            "intrinsic": tuple(fish_camera.intrinsic), 'camera_type': 'FisheyeCamera'}
                    camera_image[camera_name] = f"{scene_name}/camera/{camera_name}/{camera_filename}"
                    
                    src_camera_root = f"{scene_root}/camera/{camera_name}/{camera_filename}"
                    v_camera_root = f"{scene_root}/camera/{align_real_v[camera_name]}/{camera_filename}"
                    P(f"{scene_root}/camera/{align_real_v[camera_name]}").mkdir(parents=True, exist_ok=True)
                    camera_model = cameras_real[camera_name]
                    camera_model.ego_mask = mmcv.imread(f"{dump_root}/ego_mask/{camera_name}.png")[..., 0]
                    # process camera to virtual camera 
                    v_camera_rmatrix, v_camera_t, v_camera_intrinsic, d_src_image, d_real_cam_model, d_vcamera \
                        = convert_virtual_camera(src_camera_root, v_camera_root, None, camera_model, parm_cameras_v[align_real_v[camera_name]], calib_center.rig[camera_name])
                    scene_info["scene_info"]['calibration'][align_real_v[camera_name]] = {"extrinsic":(v_camera_rmatrix, v_camera_t), "intrinsic": v_camera_intrinsic, 'camera_type': 'FisheyeCamera'}
                    camera_image[align_real_v[camera_name]] = f"{scene_name}/camera/{align_real_v[camera_name]}/{camera_filename}"
                else:
                    W = calib_center['rig'][camera_name]['image_size'][0]
                    H = calib_center['rig'][camera_name]['image_size'][1]
                    cx = (W - 1) / 2
                    cy = (H - 1) / 2
                    fx = fy = W / 2
                    extrinsic_r = Rotation.from_quat(calib_center['rig'][camera_name]['extrinsic'][3:]).as_matrix()
                    extrinsic_t = np.array(calib_center['rig'][camera_name]['extrinsic'][:3]).reshape(3)
                    scene_info["scene_info"]['calibration'][camera_name] = {"extrinsic":(extrinsic_r, extrinsic_t), 
                                                                            "intrinsic": (cx, cy, fx, fy), 'camera_type': 'PerspectiveCamera'}
                    camera_image[camera_name] = f"{scene_name}/camera/{camera_name}/{camera_filename}"
                
                if camera_name in timestamp_window:
                    if len(timestamp_window[camera_name])<=max_sweeps:
                        timestamp_window[camera_name].append(f"{scene_name}/camera/{camera_name}/{camera_filename}")
                    else:
                        timestamp_window[camera_name].pop(0)
                        timestamp_window[camera_name].append(f"{scene_name}/camera/{camera_name}/{camera_filename}")
                else:
                    timestamp_window[camera_name] = [f"{scene_name}/camera/{camera_name}/{camera_filename}"]

            for ann in all_frames_infos[timestamp]:
                if ann['geometry_type'] == 'box3d':
                    for sensor_name, lidar_point_num in ann['visibility'].items():
                        if "camera" in sensor_name and lidar_point_num > 0:
                            pos_xyz = np.array(ann['geometry_e']['pos_xyz']).reshape(3, 1)
                            # pos_xyz[1, :] += (3.816862089990445 - 2.436862089990445)
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
                    # polyline_[1, :] += (3.816862089990445 - 2.436862089990445)
                    polyline = {
                        'class': ann['obj_type'],
                        'attr': ann['obj_attr'],
                        "points": (R_nus.T @ polyline_).transpose(1,0),
                    }
                    polyline_3d.append(polyline)
            sdf_about = {
                'occ_map': f"{scene_name}/occ/occ_2d/occ_map_-15_-15_15_15/{times_id}.png",
                'ground_height_map': f"{scene_name}/ground/ground_height_map_-15_-15_15_15/{times_id}.tif",
                'sdf_2d': f"{scene_name}/sdf/sdf_2d_-15_-15_15_15/{times_id}.tif",  # 暂时不用，可以不要
                'bev_height_map': f"{scene_name}/occ/occ_2d/bev_height_map_-15_-15_15_15/{times_id}.png",
                'bev_lidar_mask': f"{scene_name}/occ/occ_2d/bev_lidar_mask_-15_-15_15_15/{times_id}.png",
                'occ_edge_height_map': f"{scene_name}/occ/occ_2d/occ_edge_height_map_-15_-15_15_15/{times_id}.png",
                'occ_edge_lidar_mask': f"{scene_name}/occ/occ_2d/occ_edge_lidar_mask_-15_-15_15_15/{times_id}.png",
                'occ_map_sdf': f"{scene_name}/occ/occ_2d/occ_map_sdf_-15_-15_15_15/{times_id}.png",
                'occ_map_overlapped_lidar': f"{scene_name}/occ/occ_2d/occ_map_overlapped_lidar_-15_-15_15_15/{times_id}.png",
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
                    "rotation": (R_t.T @ Twes[timestamp])[:3, :3] @ R_nus,
                    'translation': (R_t.T @ Twes[timestamp])[:3, 3]
                },
                "camera_image_seg": None, 
                "occ_sdf": sdf_about
            }
        scene_info.update({'frame_info': frame_info})
    scene_infos[scene_name] = scene_info
    mmengine.dump(scene_infos, f"{dump_root}/mv_4d_infos_{scene_names[0]}.pkl")