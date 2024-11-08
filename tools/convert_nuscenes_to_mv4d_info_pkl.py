import argparse
from typing import Dict
from pathlib import Path
from collections import defaultdict

from scipy.spatial.transform import Rotation
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from loguru import logger
from tqdm import tqdm
from easydict import EasyDict as edict
import numpy as np
from copious.data_structure.dict import defaultdict2dict


NUSC_CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc-version", default="v1.0-mini")
    parser.add_argument("--nusc-data-root", type=Path, required=True)
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})


args = parse_arguments()


def main():
    trainval_nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_data_root, verbose=True)
    train_scene_list = ["scene-0001", "scene-0004"]  # TODO: to replace with splits.train
    train_infos = generate_info(trainval_nusc, train_scene_list)
    a = 100


def generate_info(nusc, scenes):
    infos = {}
    for scene in tqdm(nusc.scene):
        if scene["name"] not in scenes:
            continue
        infos[scene["name"]] = build_scene(nusc, scene["first_sample_token"])
    return infos


def build_scene(nusc, first_sample_token, max_cam_sweeps=6, max_lidar_sweeps=10):
    cur_sample = nusc.get("sample", first_sample_token)
    scene = dict(
        scene_info=defaultdict(lambda: defaultdict(dict)),
        meta_info=defaultdict(dict),
        frame_info=defaultdict(dict),
    )
    while True:
        info = dict()
        sweep_cam_info = dict()
        cam_datas = list()
        lidar_datas = list()
        info["sample_token"] = cur_sample["token"]
        info["timestamp"] = cur_sample["timestamp"]
        info["scene_token"] = cur_sample["scene_token"]

        lidar_names = ["LIDAR_TOP"]
        cam_infos = dict()
        lidar_infos = dict()

        for cam_name in NUSC_CAM_NAMES:
            cam_data = nusc.get("sample_data", cur_sample["data"][cam_name])
            cam_datas.append(cam_data)
            sweep_cam_info = dict()
            sweep_cam_info["sample_token"] = cam_data["sample_token"] # back reference to `cur_sample``
            sweep_cam_info["ego_pose"] = nusc.get("ego_pose", cam_data["ego_pose_token"])  # ego global 标定参数
            sweep_cam_info["timestamp"] = cam_data["timestamp"]
            sweep_cam_info["is_key_frame"] = cam_data["is_key_frame"]
            sweep_cam_info["height"] = cam_data["height"]
            sweep_cam_info["width"] = cam_data["width"]
            sweep_cam_info["filename"] = cam_data["filename"]
            cam_infos[cam_name] = sweep_cam_info

            calibrated_sensor = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            scene['scene_info']['calibration'][cam_name] = build_camera_calibration(calibrated_sensor)

            sweep_cam_info["calibrated_sensor"]

        for lidar_name in lidar_names:
            lidar_data = nusc.get("sample_data", cur_sample["data"][lidar_name])
            lidar_datas.append(lidar_data)
            sweep_lidar_info = dict()
            sweep_lidar_info["sample_token"] = lidar_data["sample_token"]
            sweep_lidar_info["ego_pose"] = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            sweep_lidar_info["timestamp"] = lidar_data["timestamp"]
            sweep_lidar_info["filename"] = lidar_data["filename"]
            sweep_lidar_info["calibrated_sensor"] = nusc.get(
                "calibrated_sensor", lidar_data["calibrated_sensor_token"]
            )  # everyone cam标定参数 原点为车辆后轴中 心，X朝前 Y朝左 Z朝上, 相对于ego做的标定
            lidar_infos[lidar_name] = sweep_lidar_info

        lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
        cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
        info["cam_infos"] = cam_infos
        info["lidar_infos"] = lidar_infos
        # for i in range(max_cam_sweeps):
        #     cam_sweeps.append(dict())
        for k, cam_data in enumerate(cam_datas):
            sweep_cam_data = cam_data
            # 计算存储每个相机前后帧的信息
            for j in range(max_cam_sweeps):
                if sweep_cam_data["prev"] == "":
                    break
                else:
                    sweep_cam_data = nusc.get("sample_data", sweep_cam_data["prev"])
                    sweep_cam_info = dict()
                    sweep_cam_info["sample_token"] = sweep_cam_data["sample_token"]
                    if sweep_cam_info["sample_token"] != cam_data["sample_token"]:
                        break
                    sweep_cam_info["ego_pose"] = nusc.get(
                        "ego_pose", cam_data["ego_pose_token"]
                    )  # 类似于上边ego的 global 标定参数
                    sweep_cam_info["timestamp"] = sweep_cam_data["timestamp"]
                    sweep_cam_info["is_key_frame"] = sweep_cam_data["is_key_frame"]
                    sweep_cam_info["height"] = sweep_cam_data["height"]
                    sweep_cam_info["width"] = sweep_cam_data["width"]
                    sweep_cam_info["filename"] = sweep_cam_data["filename"]
                    sweep_cam_info["calibrated_sensor"] = nusc.get(
                        "calibrated_sensor", cam_data["calibrated_sensor_token"]
                    )  # lidar 标定参数 相对于ego做的标定
                    cam_sweeps[j][NUSC_CAM_NAMES[k]] = sweep_cam_info

        # for k, lidar_data in enumerate(lidar_datas):
        #     sweep_lidar_data = lidar_data
        #     # 计算存储每个lidar前后帧的信息
        #     for j in range(max_lidar_sweeps):
        #         if sweep_lidar_data["prev"] == "":
        #             break
        #         else:
        #             sweep_lidar_data = nusc.get("sample_data", sweep_lidar_data["prev"])
        #             sweep_lidar_info = dict()
        #             sweep_lidar_info["sample_token"] = sweep_lidar_data["sample_token"]
        #             if sweep_lidar_info["sample_token"] != lidar_data["sample_token"]:
        #                 break
        #             sweep_lidar_info["ego_pose"] = nusc.get("ego_pose", sweep_lidar_data["ego_pose_token"])
        #             sweep_lidar_info["timestamp"] = sweep_lidar_data["timestamp"]
        #             sweep_lidar_info["is_key_frame"] = sweep_lidar_data["is_key_frame"]
        #             sweep_lidar_info["filename"] = sweep_lidar_data["filename"]
        #             sweep_lidar_info["calibrated_sensor"] = nusc.get(
        #                 "calibrated_sensor", cam_data["calibrated_sensor_token"]
        #             )
        #             lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
        
        # Remove empty sweeps.
        for i, sweep in enumerate(cam_sweeps):
            if len(sweep.keys()) == 0:
                cam_sweeps = cam_sweeps[:i]
                break
        for i, sweep in enumerate(lidar_sweeps):
            if len(sweep.keys()) == 0:
                lidar_sweeps = lidar_sweeps[:i]
                break
        info["cam_sweeps"] = cam_sweeps
        # info["lidar_sweeps"] = lidar_sweeps
        ann_infos = list()
        if "anns" in cur_sample:
            for ann in cur_sample["anns"]:
                ann_info = nusc.get("sample_annotation", ann)
                velocity = nusc.box_velocity(ann_info["token"])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info["velocity"] = velocity
                ann_infos.append(ann_info)
            info["ann_infos"] = ann_infos
        infos.append(info)
        if cur_sample["next"] == "":
            break
        else:
            cur_sample = nusc.get("sample", cur_sample["next"])


def build_camera_calibration(nusc_calibrated_sensor: Dict):
    intr = np.array(nusc_calibrated_sensor["camera_intrinsic"])
    rot_quat = np.array(nusc_calibrated_sensor["rotation"])[[1, 2, 3, 0]]
    calib = {
        "camera_type": "PerspectiveCamera",
        "extrinsic": (
            Rotation.from_quat(rot_quat).as_matrix(),
            np.array(nusc_calibrated_sensor["translation"])
        ),
        "intrinsic": np.array([intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]])
    }
    return calib


if __name__ == "__main__":
    main()
