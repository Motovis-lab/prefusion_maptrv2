# s3://mv-4d-annotation/AutoCollaborator/manual_annotation/baidu500/20230823_110018/whole_scene/
# s3://mv-4d-annotation/data/multimodel_data_baidu/20230823_110018/3d_box/3d_box_sync

# manual_map_box
# manual_frame_box
# infer_box
import argparse
import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['PYTORCH_THREADS'] = '1'
os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
import torch
import cv2

torch.set_num_threads(1)
cv2.setNumThreads(1)
import warnings

from mtv4d.utils.misc_base import mp_pool, torch_pool

warnings.filterwarnings("ignore")
import sys

sys.path.append(".")
sys.path.append("/home/wuhan/mtv4d/scripts/")
sys.path.append("/home/wuhan/mtv4d/")
import open3d as o3d
import mmcv
import mmengine
import os.path as op
from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path as P
from mmengine import Config
from contrib.fastbev_det.models.utils.virtual_camera import render_image, PerspectiveCamera, FisheyeCamera, \
    pcd_lidar_point, read_pcd_lidar, load_point_cloud, render_image_with_src_camera_points
from tqdm import tqdm

from mtv4d.annos_4d.helper import (  # type: ignore
    check_whether_points_behind_camera,
    CameraParam,
    transform_points_ego_to_lidar,
    write_json_from_list,
    read_json_to_list,
    anno_box_to_7_values_box,
    get_times,
    get_sync_filename,
    read_cal_data,
    generate_lidar_mask,
)


import warnings
warnings.filterwarnings("ignore")
import sys

sys.path.append(".")
import open3d as o3d
import pickle
from email.policy import default
from os import times
import os.path as op
from collections import defaultdict
from copy import deepcopy
from re import L
from time import time
import numpy as np
import json
from pathlib import Path as P
from typing import List, Tuple, Union
import torch
from tqdm import tqdm
from mtv4d.annos_4d.helper import (
    CameraParam,
    write_json_from_list,
    read_json_to_list,
    read_cal_data,
)

from mtv4d.annos_4d.misc import *

import argparse

from mtv4d.annos_4d.misc import read_ego_paths  # type: ignore
# from scripts.generate_4d_frame_clean import generate_DS4d_from_4dMapJson, solve_ds_occlusion_sub_id, \
#     generate_4d_frame_json_data, read_ts_json  # type: ignore
def update_poly_with_world_lidar_vertices(polyline, world_T, ego_T):
    vertices = polyline["vertices"]
    polyline["vertices_world"] = translate_pts3d_with_T(vertices, world_T).tolist()
    polyline["vertices_ego"] = translate_pts3d_with_T(vertices, ego_T).tolist()

def is_visible_from_box_visiblity(box):
    return any([ data >3 for cam, data in box.items()])
def update_box_with_world_lidar_psr(box, world_T, ego_T):
    psr = box["psr"]
    box["psr_world"] = translate_psr_to_output_geometry(translate_psr_with_T(psr, world_T))
    box["psr_ego"] = translate_psr_to_output_geometry(translate_psr_with_T(psr, ego_T))

def generate_frame_info_box(frame_boxes, ts, varying_state):
    # applicable to varying & not_varying
    # varying_state: 0: varying, 1: not_varying
    frame_info = []
    for track_id, box in frame_boxes.items():
        # if varying_state == 'not_varying':
        if 'sub_id' not in box.keys():
            continue  # 表示不可见
        else:
            track_id = f'{track_id}_{box["sub_id"]}'

        box_single_info = {
            "obj_type": box["obj_type"],
            "obj_track_id": track_id,
            "obj_time_varying_state": varying_state,
            "geometry_type": "box3d",
            "timestamp": ts,
            "obj_attr": {} if "obj_attr" not in box.keys() and varying_state == "varying" else box["obj_attr"],
            "visibility": box["visibility"],
            "velocity": box["velocity"],
            "geometry_e": box["psr_ego"],
            "geometry_w": box["psr_world"],
        }
        frame_info.append(box_single_info)
    return frame_info

def generate_frame_info_polyline(frame_polylines, ts):
    frame_info = []
    for track_id, poly in frame_polylines.items():
        if 'sub_id' not in poly.keys():
            continue  # 表示不可见
        else:
            track_id = f'{track_id}_{poly["sub_id"]}'

        polyline_single_info = {
            "obj_type": poly["obj_type"],
            "obj_track_id": track_id,
            "obj_time_varying_state": "not_varying",
            "geometry_type": "polyline3d",
            "timestamp": ts,
            "obj_attr": poly["obj_attr"],
            "visibility": poly["visibility"],
            "geometry": poly["vertices_ego"],
        }
        frame_info.append(polyline_single_info)
    return frame_info

def is_visible_from_poly_visiblity(poly):
    return any([ '1' in data for cam, data in poly.items()])
def generate_4d_frame_json_data(dn_boxes_vis_dict_id2ts, map_boxes_vis_dict_id2ts, map_polylines_vis_dict_id2ts):

    output_4d_frame_json = defaultdict(list)

    for ts, frame_box_dict in dn_boxes_vis_dict_id2ts.items():
        boxes_info = generate_frame_info_box(frame_box_dict, ts, varying_state="varying")
        output_4d_frame_json[ts] += boxes_info

    for ts, frame_box_dict in map_boxes_vis_dict_id2ts.items():
        boxes_info = generate_frame_info_box(frame_box_dict, ts, varying_state="not_varying")
        output_4d_frame_json[ts] += boxes_info

    for ts, polylines in map_polylines_vis_dict_id2ts.items():
        polylines_info = generate_frame_info_polyline(polylines, ts)
        output_4d_frame_json[ts] += polylines_info

    return output_4d_frame_json


def solve_ds_occlusion_sub_id(
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
        acc_thres=20
):
    for track_id, box_dict in dn_boxes_vis_dict_id2ts.items():
        ts_list = sorted(box_dict.keys())
        vis_list = [is_visible_from_box_visiblity(box_dict[ts]['visibility']) for ts in ts_list]
        acc_ts_list = []
        sub_id = 0
        st_flag = True
        for ts, visible in zip(ts_list, vis_list):
            if not visible:
                acc_ts_list += [ts]
            else:
                if not st_flag:
                    if len(acc_ts_list) >= acc_thres:
                        sub_id += 1
                    else:
                        for j in acc_ts_list:
                            box_dict[j]['sub_id'] = sub_id
                box_dict[ts]['sub_id'] = sub_id
                st_flag = False
                acc_ts_list = []

    for track_id, box_dict in map_boxes_vis_dict_id2ts.items():
        ts_list = sorted(box_dict.keys())
        vis_list = [is_visible_from_box_visiblity(box_dict[ts]['visibility']) for ts in ts_list]
        acc_ts_list = []
        sub_id = 0
        st_flag = True
        for ts, visible in zip(ts_list, vis_list):
            if not visible:
                acc_ts_list += [ts]
            else:
                if not st_flag:
                    if len(acc_ts_list) >= acc_thres:
                        sub_id += 1
                    else:
                        for j in acc_ts_list:
                            box_dict[j]['sub_id'] = sub_id
                box_dict[ts]['sub_id'] = sub_id
                st_flag = False
                acc_ts_list = []

    for track_id, poly_dict in map_polylines_vis_dict_id2ts.items():
        ts_list = sorted(poly_dict.keys())
        vis_list = [is_visible_from_poly_visiblity(poly_dict[ts]['visibility']) for ts in ts_list]
        acc_ts_list = []
        sub_id = 0
        st_flag = True
        for ts, visible in zip(ts_list, vis_list):
            if not visible:
                acc_ts_list += [ts]
            else:
                if not st_flag:
                    if len(acc_ts_list) >= acc_thres:
                        sub_id += 1
                    else:  # 如果没超过，之前的要补上
                        for j in acc_ts_list:
                            poly_dict[j]['sub_id'] = sub_id
                poly_dict[ts]['sub_id'] = sub_id
                st_flag = False
                acc_ts_list = []

    return (
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    )


def read_ts_json(path):
    ts_dict_list = read_json_to_list(path)
    output_dict = {i["lidar"]: i for i in ts_dict_list}
    ts_list = sorted([i["lidar"] for i in ts_dict_list])
    return ts_list, output_dict


def generate_DS4d_from_4dMapJson(data_path, Twes):
    output_json_list = read_json_to_list(data_path)
    dn_boxes_vis_dict_ts2id, map_boxes_vis_dict_ts2id = defaultdict(dict), defaultdict(dict)
    dn_boxes_vis_dict_id2ts, map_boxes_vis_dict_id2ts = defaultdict(dict), defaultdict(dict)
    map_polylines_vis_dict_ts2id, map_polylines_vis_dict_id2ts = defaultdict(dict), defaultdict(dict)
    for box_dict in tqdm(output_json_list, desc='generating 4d DS'):
        if box_dict["geometry_type"] == "polyline3d":
            poly_dict = box_dict
            for poly in poly_dict["ts_list_of_dict"]:
                ts = poly["timestamp"]
                formatted_poly = {k: v for k, v in poly.items()}
                formatted_poly["timestamp"] = ts
                formatted_poly["obj_time_varying_state"] = poly_dict["obj_time_varying_state"]
                formatted_poly["obj_track_id"] = poly_dict["obj_track_id"]
                formatted_poly["geometry_type"] = poly_dict["geometry_type"]
                formatted_poly["obj_type"] = poly_dict["obj_type"]
                formatted_poly["obj_attr"] = poly_dict["obj_attr"]
                formatted_poly["obj_type"] = poly_dict["obj_type"]
                formatted_poly["vertices"] = deepcopy(poly_dict["geometry"])
                world_T = np.eye(4)
                ego_T = np.linalg.inv(Twes[ts])
                update_poly_with_world_lidar_vertices(formatted_poly, world_T, ego_T)
                formatted_poly["velocity"] = [0, 0, 0]
                formatted_poly = deepcopy(formatted_poly)
                map_polylines_vis_dict_ts2id[ts][box_dict["obj_track_id"]] = formatted_poly
                map_polylines_vis_dict_id2ts[box_dict["obj_track_id"]][ts] = formatted_poly
        else:
            for box in box_dict["ts_list_of_dict"]:
                ts = box["timestamp"]
                formatted_box = {k: v for k, v in box.items()}
                formatted_box["obj_time_varying_state"] = box_dict["obj_time_varying_state"]
                formatted_box["obj_track_id"] = box_dict["obj_track_id"]
                formatted_box["geometry_type"] = box_dict["geometry_type"]
                formatted_box["obj_type"] = box_dict["obj_type"]
                formatted_box["timestamp"] = ts

                if box_dict["obj_time_varying_state"] == "not_varying":
                    formatted_box["obj_attr"] = box_dict["obj_attr"]
                    formatted_box["obj_type"] = box_dict["obj_type"]
                    formatted_box["geometry"] = deepcopy(box_dict["geometry"])
                    formatted_box["psr"] = translate_output_geometry_to_psr(box_dict["geometry"])
                    world_T = np.eye(4)
                    ego_T = np.linalg.inv(Twes[ts])
                    update_box_with_world_lidar_psr(formatted_box, world_T, ego_T)
                    formatted_box["velocity"] = [0, 0, 0]
                    formatted_box = deepcopy(formatted_box)
                    map_boxes_vis_dict_ts2id[ts][box_dict["obj_track_id"]] = formatted_box
                    map_boxes_vis_dict_id2ts[box_dict["obj_track_id"]][ts] = formatted_box

                elif box_dict["obj_time_varying_state"] == "varying":
                    formatted_box["obj_attr"] = box["obj_attr"]
                    formatted_box["geometry"] = deepcopy(box["geometry"])
                    formatted_box["psr"] = translate_output_geometry_to_psr(box["geometry"])
                    world_T = np.eye(4)
                    ego_T = np.linalg.inv(Twes[ts])
                    update_box_with_world_lidar_psr(formatted_box, world_T, ego_T)
                    formatted_box["velocity"] = box["velocity"]
                    formatted_box = deepcopy(formatted_box)
                    dn_boxes_vis_dict_ts2id[ts][box_dict["obj_track_id"]] = formatted_box
                    dn_boxes_vis_dict_id2ts[box_dict["obj_track_id"]][ts] = formatted_box

    return (
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    )



align_real_v = {"camera8": "VCAMERA_FISHEYE_FRONT",
                "camera5": "VCAMERA_FISHEYE_LEFT",
                "camera1": "VCAMERA_FISHEYE_BACK",
                'camera11': "VCAMERA_FISHEYE_RIGHT"
                }

parm_cameras_v = {
    "VCAMERA_FISHEYE_FRONT": (-120, 0, -90),
    "VCAMERA_FISHEYE_LEFT": (-135, 0, 0),
    "VCAMERA_FISHEYE_RIGHT": (-135, 0, -180),
    "VCAMERA_FISHEYE_BACK": (-120, 0, 90)
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
    output_json_frame_dlist = generate_4d_frame_json_data(dn_boxes_vis_dict_ts2id, map_boxes_vis_dict_ts2id,
                                                          map_polylines_vis_dict_ts2id)
    # for ts, frames_labels in tqdm(output_json_frame_dlist.items(), desc='generating frame json'):
    #     write_json_from_list(
    #         frames_labels, op.join(scene_root, f"4d_anno_infos/4d_anno_infos_frame/frames_labels_all/{int(ts)}.json"), format_float=True, indent=4
    #     )
    return output_json_frame_dlist


def convert_virtual_camera(src_camear_root, save_img_root, save_mask_root, real_cam_model, v_cam_paramter, calib,
                           W=1280, H=960, render=False):
    src_image = mmcv.imread(src_camear_root, channel_order="bgr")
    R = Rotation.from_euler('xyz', angles=v_cam_paramter, degrees=True).as_matrix()
    t = [0, 0, 0]
    v_cam_rmatrix = R
    v_cam_t = np.array(calib['extrinsic'][:3]).reshape(3)
    if 'FISHEYE' not in save_img_root:
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
        vcamera = PerspectiveCamera((W, H), (R, t), intrinsic)
    else:
        W = 768
        H = 512
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 4
        intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)
        vcamera = FisheyeCamera((W, H), (R, t), intrinsic, fov=180)
    if render:
        dst_image, dst_mask = render_image(src_image, real_cam_model, vcamera)
        P(save_img_root).parent.mkdir(exist_ok=True, parents=True)
        mmcv.imwrite(dst_image, save_img_root)
    return v_cam_rmatrix, v_cam_t, intrinsic, src_image, real_cam_model, vcamera


def process_lidar(data):
    scene_root, times_id = data
    # pt1 = read_pcd_lidar(f"{scene_root}/lidar/undistort_static_lidar1/{times_id}.pcd")
    # pt2 = read_pcd_lidar(f"{scene_root}/lidar/undistort_static_lidar2/{times_id}.pcd")
    # pt3 = read_pcd_lidar(f"{scene_root}/lidar/undistort_static_lidar3/{times_id}.pcd")
    src_lidar_path = f"{scene_root}/lidar/undistort_static_merged_lidar1/{times_id}.pcd"
    dst_lidar_path = f"{scene_root}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd"
    # ori_pcd_lidar_point(src_lidar_path, np.concatenate([pt1, pt2, pt3]))
    P(f"{scene_root}/lidar/undistort_static_merged_lidar1_model/").mkdir(parents=True, exist_ok=True)
    single_lidar_process(src_lidar_path, dst_lidar_path)


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
    R_nus[:3, :3] = Rotation.from_euler("XYZ", angles=(0, 0, 90), degrees=True).as_matrix()
    lidar_point = R_nus.T @ deepcopy(cams_lidar_point)
    lidar_point[3, :] = intensity
    ori_pcd_lidar_point(save_root, lidar_point.T)


def ori_pcd_lidar_point(save_path, lidar_points):
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    points_intensities = lidar_points[:, 3][:, None]
    points_positions = lidar_points[:, :3]
    lidar_map = o3d.t.geometry.PointCloud(device)

    lidar_map.point.positions = o3d.core.Tensor(points_positions, dtype, device)
    lidar_map.point.intensity = o3d.core.Tensor(points_intensities, dtype, device)

    o3d.t.io.write_point_cloud(str(save_path), lidar_map)


def create_moving_object(all_frames_infos, timestamps):
    R_nus = Rotation.from_euler("xyz", angles=(0, 0, 90), degrees=True).as_matrix()
    moving_objects_track_id_trajectory = {}  # defaultdict(list)
    for timestamps in timestamps:
        for ann in all_frames_infos[timestamps]:
            if ann['geometry_type'] == 'box3d' and int(ann['obj_track_id'].split('_')[0]) < 10000:
                pose_xyz = R_nus.T @ np.array(ann['geometry_w']["pos_xyz"]).reshape(3, 1)
                rot_xyz = R_nus.T @ Rotation.from_euler("XYZ", angles=ann['geometry_w']["rot_xyz"],
                                                        degrees=False).as_matrix()
                if ann['obj_track_id'] in moving_objects_track_id_trajectory:
                    moving_objects_track_id_trajectory[ann['obj_track_id']]['timestamps'].append(int(ann['timestamp']))
                    moving_objects_track_id_trajectory[ann['obj_track_id']]['attr'].append(ann['obj_attr'])
                    moving_objects_track_id_trajectory[ann['obj_track_id']]['pose'].append((pose_xyz, rot_xyz))
                else:
                    moving_objects_track_id_trajectory[ann['obj_track_id']] = {
                        'class': ann['obj_type'],
                        'timestamps': [int(ann['timestamp'])],
                        'attr': [ann['obj_attr']],
                        'pose': [(pose_xyz, rot_xyz)]
                    }
    return moving_objects_track_id_trajectory
def generate_single_vcam(data):
    scene_root, ts, cameras_real, calib_center, parm_cameras_v, fish_cameras = data
    for camera_name in fish_cameras:
        camera_filename = f"{int(ts)}.jpg"
        camera_model = cameras_real[camera_name]
        src_camera_root = f"{scene_root}/camera/{camera_name}/{camera_filename}"
        v_camera_root = f"{scene_root}/camera/{align_real_v[camera_name]}/{camera_filename}"
        P(f"{scene_root}/camera/{align_real_v[camera_name]}").mkdir(parents=True, exist_ok=True)
        try:
            convert_virtual_camera(src_camera_root, v_camera_root, None, camera_model,
                               parm_cameras_v[align_real_v[camera_name]],
                               calib_center.rig[camera_name])
        except Exception:
            print('error', src_camera_root)

if __name__ == "__main__":
    # ---args
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['BLIS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['PYTORCH_THREADS'] = '1'
    os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--scene_root', default="/ssd1/tmp/20231104_170321_1699088601564_1699088721564")
    parser.add_argument('--pkl_save_path', default="/ssd1/tmp/20231104_170321_1699088601564_1699088721564/mv_4d_infos_dbg.pkl")
    # parser.add_argument('--scene_id', default="20231104_170321_1699088601564_1699088721564")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    scene_infos = defaultdict()
    sphere_flip_radius = 1000
    max_sweeps = 20

    all_cameras = ["camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8", "camera11",
                   "camera12", "camera15"]
    fish_cameras = ["camera1", "camera5", "camera8", "camera11"]


    def process_calib(scene_root):
        data_root = str(P(scene_root).parent)
        calib_center = Config.fromfile(f"{scene_root}/calibration_center.yml")
        R_nus = Rotation.from_euler("xyz", angles=(0, 0, 90), degrees=True).as_matrix()
        lidar1_cali_r = R_nus.T @ Rotation.from_quat(calib_center['rig']['lidar1']['extrinsic'][3:]).as_matrix()
        lidar1_cali_t = R_nus.T @ np.array(calib_center['rig']['lidar1']['extrinsic'][:3]).reshape(3)
        for cam in fish_cameras:
            calib_center['rig'][cam]['extrinsic'] = [
                *((R_nus.T @ np.array(calib_center['rig'][cam]['extrinsic'][:3]).reshape(3)).tolist()), *(
                    Rotation.from_matrix((R_nus.T @ Rotation.from_quat(
                        calib_center['rig'][cam]['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
        cameras_real = {cam: FisheyeCamera.init_from_motovis_cfg(calib_center.rig[cam]) for cam in fish_cameras}
        for cam in fish_cameras:
            camera_model = cameras_real[cam]
            camera_model.ego_mask = mmcv.imread(f"{data_root}/ego_mask/{cam}.png")[..., 0]
        return calib_center, cameras_real, lidar1_cali_r, lidar1_cali_t




    def get_ts_dictionary(timestamps, scene_root):
        return {ts: {
            'lidar': ts if P(f"{scene_root}/lidar/undistort_static_lidar1/{int(ts)}.pcd").exists() or
                           P(f"{scene_root}/lidar/undistort_static_merged_lidar1/{int(ts)}.pcd").exists() else None,
            # 'overlapped_lidar1': f"{ts}.pcd" if P(f"{scene_root}/lidar/overlapped_lidar1/{int(ts)}.pcd").exists() else None,
            'camera1': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera1/{int(ts)}.jpg").exists() else None,
            'camera5': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera5/{int(ts)}.jpg").exists() else None,
            'camera8': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera8/{int(ts)}.jpg").exists() else None,
            'camera2': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera2/{int(ts)}.jpg").exists() else None,
            'camera3': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera3/{int(ts)}.jpg").exists() else None,
            'camera4': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera4/{int(ts)}.jpg").exists() else None,
            'camera6': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera6/{int(ts)}.jpg").exists() else None,
            'camera7': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera7/{int(ts)}.jpg").exists() else None,
            'camera11': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera11/{int(ts)}.jpg").exists() else None,
            'camera12': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera12/{int(ts)}.jpg").exists() else None,
            'camera15': f'{int(ts)}.jpg' if P(f"{scene_root}/camera/camera15/{int(ts)}.jpg").exists() else None,
        } for ts in timestamps}


    def process_one_scene(scene_root, pkl_save_path):
        scene_name = scene_id = P(scene_root).name
        dump_root = str(P(scene_root).parent)
        calib_center, cameras_real, lidar1_cali_r, lidar1_cali_t = process_calib(scene_root)
        Twes, timestamps = read_ego_paths(op.join(scene_root, f"trajectory.txt"))
        sensors_timestamps = timestamps_dict = get_ts_dictionary(timestamps, scene_root)
        # timestamps = [ts for ts, val in timestamps_dict.items() if all(val.values())]  # lidar camera pose all!
        # timestamps = [float(p.stem) for p in P(f'{scene_root}/4d_anno_infos/4d_anno_infos_frame/frames_labels').glob('*.json') ]
        if False:
            # 1 process lidar
            timestamps = [i.stem for i in P(f'{scene_root}/lidar/undistort_static_merged_lidar1').glob('*.pcd')]
            torch_pool(process_lidar, [(scene_root, str(int(ts))) for ts in timestamps])
            # 2 process virtual camera
            timestamps = [float(p.stem) for p in
                          P(f'{scene_root}/4d_anno_infos/4d_anno_infos_frame/frames_labels').glob('*.json')]

            torch_pool(generate_single_vcam,
                    [[scene_root, ts, cameras_real, calib_center, parm_cameras_v, fish_cameras] for ts in timestamps])
            exit()
        R_nus = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        all_frames_infos = generate_labels_scene_from_4dMapjson(scene_root, Twes)
        # timestamps = [i for i in timestamps if float(i) in all_frames_infos.keys()]
        timestamps = sorted(all_frames_infos.keys())
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
        scene_info["scene_info"]['moving_objects_track_id_trajectory'] = create_moving_object(all_frames_infos,
                                                                                              timestamps)
        # meta_info 
        scene_info['meta_info'] = {
            "description": "some information with this scene",
            "space_range": {
                "map": [36, -12, -12, 12, 10, -10],
                "det": [36, -12, -12, 12, 10, -10],
                "occ": [36, -12, -12, 12, 10, -10]
            },
            "time_range": 2
        }

        frame_info = {}
        timestamp_window = defaultdict(list)
        # timestamps = [ts for ts in timestamps if ts in ]
        # for timestamp in tqdm(timestamps, "Process timestamp ... "):
        for timestamp in tqdm(timestamps):
            times_id = str(int(timestamp))
            if times_id[-2:] != "64":
                print('error', scene_id, timestamp)
                continue
            sdf_about = {
                'occ_map': f"{scene_name}/occ/occ_2d/occ_map_-15_-15_15_15/{times_id}.png",
                'ground_height_map': f"{scene_name}/ground/ground_height_map_-15_-15_15_15/{times_id}.tif",
                'sdf_2d': f"{scene_name}/sdf/sdf_2d_-15_-15_15_15/{times_id}.tif",  # 暂时不用，可以不要
                'bev_height_map': f"{scene_name}/occ/occ_2d/bev_height_map_-15_-15_15_15/{times_id}.png",
                'bev_lidar_mask': f"{scene_name}/occ/occ_2d/bev_lidar_mask_-15_-15_15_15/{times_id}.png",
                'occ_edge_height_map': f"{scene_name}/occ/occ_2d/occ_edge_height_map_-15_-15_15_15/{times_id}.png",
                'occ_edge_lidar_mask': f"{scene_name}/occ/occ_2d/occ_edge_lidar_mask_-15_-15_15_15/{times_id}.png",
                'occ_map_sdf': f"{scene_name}/occ/occ_2d/occ_map_sdf_-15_-15_15_15/{times_id}.png",
                'occ_edge': f"{scene_name}/occ/occ_2d/occ_edge_-15_-15_15_15/{times_id}.png",
                'occ_sdf_3d': None
            }
            occ_ok = True
            for key in sdf_about:
                if key != "sdf_2d" and key != "occ_sdf_3d":
                    if not os.path.exists(P(dump_root) / P(sdf_about[key])):
                        occ_ok = False
                        break
            # if not occ_ok:
            #     continue
            # convert real_cam to v_cam
            camera_image = {}
            boxes_3d = []
            polyline_3d = []
            lidar_point = {}
            scene_info["scene_info"]['calibration'].update({'lidar1': (lidar1_cali_r, lidar1_cali_t)})
            lidar_point['lidar1'] = f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd"
            if "lidar1" in timestamp_window:
                if len(timestamp_window['lidar1']) <= max_sweeps:
                    timestamp_window['lidar1'].append(
                        f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd")
                else:
                    timestamp_window['lidar1'].pop(0)
                    timestamp_window['lidar1'].append(
                        f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd")
            else:
                timestamp_window['lidar1'] = [f"{scene_name}/lidar/undistort_static_merged_lidar1_model/{times_id}.pcd"]

            for camera_name in all_cameras:
                camera_filename = sensors_timestamps[timestamp][camera_name]


                if camera_name in fish_cameras:
                    fish_camera = FisheyeCamera.init_from_motovis_cfg(calib_center['rig'][camera_name])
                    extrinsic_r = Rotation.from_quat(calib_center['rig'][camera_name]['extrinsic'][3:]).as_matrix()
                    extrinsic_t = np.array(calib_center['rig'][camera_name]['extrinsic'][:3]).reshape(3)
                    scene_info["scene_info"]['calibration'][camera_name] = {
                        "extrinsic": (extrinsic_r, extrinsic_t),
                        "intrinsic": tuple(fish_camera.intrinsic),
                        'camera_type': 'FisheyeCamera'}
                    camera_image[camera_name] = f"{scene_name}/camera/{camera_name}/{camera_filename}"

                    src_camera_root = f"{scene_root}/camera/{camera_name}/{camera_filename}"
                    v_camera_root = f"{scene_root}/camera/{align_real_v[camera_name]}/{camera_filename}"
                    P(f"{scene_root}/camera/{align_real_v[camera_name]}").mkdir(parents=True, exist_ok=True)
                    camera_model = cameras_real[camera_name]
                    camera_model.ego_mask = mmcv.imread(f"{dump_root}/ego_mask/{camera_name}.png")[..., 0]
                    # process camera to virtual camera 
                    v_camera_rmatrix, v_camera_t, v_camera_intrinsic, d_src_image, d_real_cam_model, d_vcamera \
                        = convert_virtual_camera(src_camera_root, v_camera_root, None, camera_model,
                                                 parm_cameras_v[align_real_v[camera_name]],
                                                 calib_center.rig[camera_name],
                                                 render=False)
                    scene_info["scene_info"]['calibration'][align_real_v[camera_name]] = {
                        "extrinsic": (v_camera_rmatrix, v_camera_t), "intrinsic": v_camera_intrinsic,
                        'camera_type': 'FisheyeCamera'}
                    camera_image[align_real_v[
                        camera_name]] = f"{scene_name}/camera/{align_real_v[camera_name]}/{camera_filename}"
                else:
                    W = calib_center['rig'][camera_name]['image_size'][0]
                    H = calib_center['rig'][camera_name]['image_size'][1]
                    cx = (W - 1) / 2
                    cy = (H - 1) / 2
                    fx = fy = W / 2
                    extrinsic_r = Rotation.from_quat(calib_center['rig'][camera_name]['extrinsic'][3:]).as_matrix()
                    extrinsic_t = np.array(calib_center['rig'][camera_name]['extrinsic'][:3]).reshape(3)
                    scene_info["scene_info"]['calibration'][camera_name] = {"extrinsic": (extrinsic_r, extrinsic_t),
                                                                            "intrinsic": (cx, cy, fx, fy),
                                                                            'camera_type': 'PerspectiveCamera'}
                    camera_image[camera_name] = f"{scene_name}/camera/{camera_name}/{camera_filename}"

                if camera_name in timestamp_window:
                    if len(timestamp_window[camera_name]) <= max_sweeps:
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
                            box = {
                                "class": ann['obj_type'],
                                "attr": ann['obj_attr'],
                                "size": ann['geometry_e']['scale_xyz'],
                                'rotation': R_nus.T @ Rotation.from_euler("XYZ", angles=ann['geometry_e']['rot_xyz'],
                                                                          degrees=False).as_matrix(),
                                'translation': (R_nus.T @ pos_xyz).reshape(3),
                                "track_id": ann['obj_track_id'],
                                "velocity": (R_nus.T @ np.array(ann['velocity']).reshape(3, 1)).reshape(3)
                            }
                            boxes_3d.append(box)
                            break
                elif ann['geometry_type'] == 'polyline3d':
                    polyline_ = np.array(ann['geometry']).transpose(1, 0)
                    polyline = {
                        'class': ann['obj_type'],
                        'attr': ann['obj_attr'],
                        "points": (R_nus.T @ polyline_).transpose(1, 0),
                    }
                    polyline_3d.append(polyline)

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
        P(pkl_save_path).parent.mkdir(exist_ok=True, parents=True)
        mmengine.dump(scene_infos, pkl_save_path)


    save_pkl_path = args.pkl_save_path if args.pkl_save_path is not None else f"{args.scene_root}/mv_4d_infos_{args.scene_id}.pkl"
    process_one_scene(args.scene_root, args.pkl_save_path)
