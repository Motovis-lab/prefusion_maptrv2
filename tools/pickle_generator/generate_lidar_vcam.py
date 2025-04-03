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
import os.path as op
from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path as P
from mmengine import Config
from contrib.fastbev_det.models.utils.virtual_camera import render_image, PerspectiveCamera, FisheyeCamera, \
    pcd_lidar_point, read_pcd_lidar, load_point_cloud, render_image_with_src_camera_points

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

from mtv4d.annos_4d.misc import read_ego_paths  # type: ignore
from scripts.generate_4d_frame_clean import generate_DS4d_from_4dMapJson, solve_ds_occlusion_sub_id, \
    generate_4d_frame_json_data, read_ts_json  # type: ignore

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
                               calib_center.rig[camera_name], render=True)
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
    parser.add_argument('--pkl_save_path', default="/ssd1/tmp/20231104_170321_1699088601564_1699088721564/mv_4d_infos.pkl")
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
        if True:
            # 1 process lidar
            timestamps = [i.stem for i in P(f'{scene_root}/lidar/undistort_static_merged_lidar1').glob('*.pcd')]
            torch_pool(process_lidar, [(scene_root, str(int(ts))) for ts in timestamps], 32)
            # 2 process virtual camera
            timestamps = [float(p.stem) for p in
                          P(f'{scene_root}/4d_anno_infos/4d_anno_infos_frame/frames_labels').glob('*.json')]
            # for i in [[scene_root, ts, cameras_real, calib_center, parm_cameras_v, fish_cameras] for ts in timestamps]:
            #     generate_single_vcam(i)
            torch_pool(generate_single_vcam,
                    [[scene_root, ts, cameras_real, calib_center, parm_cameras_v, fish_cameras] for ts in timestamps])


    save_pkl_path = args.pkl_save_path if args.pkl_save_path is not None else f"{args.scene_root}/mv_4d_infos_{args.scene_id}.pkl"
    process_one_scene(args.scene_root, args.pkl_save_path)
    # --scene_root /ssd1/MV4D_DATA/20230823_113013 --pkl_save_path /tmp/1.pkl
