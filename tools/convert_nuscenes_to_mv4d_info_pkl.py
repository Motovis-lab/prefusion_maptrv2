import argparse
import pickle
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
from copious.cv.geometry import xyzq2mat
from copious.io.fs import ensured_path

from prefusion.dataset.utils import T4x4


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
    parser.add_argument("--info-pkl-save-dir", type=ensured_path, required=True)
    parser.add_argument("--info-pkl-filename-prefix", default="nusc")
    parser.add_argument("--train-scenes", nargs="*", default=splits.train)
    parser.add_argument("--val-scenes", nargs="*", default=splits.val)
    parser.add_argument("--test-scenes", nargs="*", default=splits.test)
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})


args = parse_arguments()


def main():
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_data_root, verbose=True)
    nusc_map = load_map_info(nusc)
    if args.nusc_version == "v1.0-test":
        test_infos = generate_info_pkl(nusc, nusc_map["test"], args.test_scenes)
        save_pickle(test_infos, args.info_pkl_save_dir / f"{args.info_pkl_filename_prefix}_test_info.pkl")
    elif args.nusc_version == "v1.0-trainval":
        train_infos = generate_info_pkl(nusc, nusc_map["train"], args.train_scenes)
        val_infos = generate_info_pkl(nusc, nusc_map["val"], args.val_scenes)
        save_pickle(train_infos, args.info_pkl_save_dir / f"{args.info_pkl_filename_prefix}_train_info.pkl")
        save_pickle(val_infos, args.info_pkl_save_dir / f"{args.info_pkl_filename_prefix}_val_info.pkl")


def load_map_info(nusc: NuScenes):
    nusc_map = defaultdict(lambda: defaultdict(dict))
    phases = ["train", "val", "test"]
    for ph in phases:
        filepath = args.nusc_data_root / f"nuscenes_map_infos_temporal_{ph}.pkl"
        map_info = read_info_pkl(filepath)
        for frame in map_info["infos"]:
            if frame["scene_token"] not in nusc._token2ind['scene']:
                continue
            scene_id = nusc.get("scene", frame["scene_token"])["name"]
            nusc_map[ph][scene_id][frame["timestamp"]] = frame["annotation"]
    logger.info(f"Number of scenes loaded from map-info: {[(k, len(m)) for k, m in nusc_map.items()]}")
    return nusc_map


def read_info_pkl(filepath: Path):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Can't find {filepath.name} in {args.nusc_data_root}. "
            f"Please prepare map pkl into {args.nusc_data_root} by running MapTRv2 script custom_nusc_map_converter.py"
        )


def generate_info_pkl(nusc: NuScenes, nusc_map, scenes):
    infos = {}
    for scene in tqdm(nusc.scene):
        if scene["name"] not in scenes:
            continue
        infos[scene["name"]] = build_scene(nusc, nusc_map, scene["first_sample_token"])
    return infos


def build_scene(nusc: NuScenes, nusc_map, first_sample_token):
    return dict(
        scene_info=build_scene_info(),
        meta_info=build_meta_info(),
        frame_info=build_frame_info(nusc, nusc_map, first_sample_token),
    )


def build_scene_info() -> Dict:
    scene_info = defaultdict(dict)
    for cam_name in NUSC_CAM_NAMES:
        scene_info["camera_mask"][cam_name] = build_camera_mask(cam_name)
        scene_info["depth_mode"][cam_name] = build_depth_mode(cam_name)
    return defaultdict2dict(scene_info)


def build_meta_info() -> Dict:
    return {
        "space_range": {
            "map": [50, -50, -50, 50, 5, -3],
            "det": [50, -50, -50, 50, 5, -3],
            "occ": [50, -50, -50, 50, 5, -3],
        },
        "time_range": 2,
        "time_unit": 0.001,
    }


def build_frame_info(nusc: NuScenes, nusc_map, first_sample_token) -> Dict:
    cur_sample = nusc.get("sample", first_sample_token)
    frame_info = defaultdict(lambda: defaultdict(dict))

    while True:
        ts = str(cur_sample["timestamp"] // 1000)  # convert time-unit to ms
        lidar_ego_pose = build_ego_pose(nusc, cur_sample)

        for cam_name in NUSC_CAM_NAMES:
            frame_info[ts]["camera_image"][cam_name] = build_camera_image(nusc, cur_sample, cam_name, lidar_ego_pose)

        frame_info[ts]["3d_boxes"] = build_3d_boxes(nusc, cur_sample, lidar_ego_pose)
        frame_info[ts]["3d_polylines"] = build_3d_polylines(nusc, nusc_map, cur_sample, lidar_ego_pose)
        frame_info[ts]["ego_pose"] = lidar_ego_pose
        frame_info[ts]["timestamp_window"] = [None]
        frame_info[ts]["sample_token"] = cur_sample["token"]

        if cur_sample["next"] == "":
            break
        else:
            cur_sample = nusc.get("sample", cur_sample["next"])

    return defaultdict2dict(frame_info)


def build_camera_calibration(nusc_calibrated_sensor: Dict):
    intr = np.array(nusc_calibrated_sensor["camera_intrinsic"])
    rot_quat = np.array(nusc_calibrated_sensor["rotation"])[[1, 2, 3, 0]]
    calib = {
        "camera_type": "PerspectiveCamera",
        "extrinsic": (Rotation.from_quat(rot_quat).as_matrix(), np.array(nusc_calibrated_sensor["translation"])),
        "intrinsic": np.array([intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]]),
    }
    return calib


def build_ego_pose(nusc: NuScenes, cur_sample):
    lidar_info = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_info["ego_pose_token"])
    rot_quat = np.array(ego_pose["rotation"])[[1, 2, 3, 0]]
    return {"rotation": Rotation.from_quat(rot_quat).as_matrix(), "translation": np.array(ego_pose["translation"])}


def build_camera_mask(cam_name: str) -> Path:
    return Path("self_mask") / f"{cam_name}.png"


def build_depth_mode(cam_name: str) -> str:
    return "d"


def build_camera_image(nusc: NuScenes, cur_sample, cam_name, lidar_ego_pose: Dict) -> Path:
    """Build camera Image

    Parameters
    ----------
    nusc : _type_
        nuscenes-devkit object.
    cur_sample : _type_
        current sample of nuscenes, including the tokens of all realted info
    cam_name : _type_
        camera name
    lidar_ego_pose : Dict
        ego_pose of LIDAR_TOP (as the true ego_pose, while camera's ego pose is encoded into camera's extrinsic)

    Returns
    -------
    Path
        _description_
    """
    cam_data = nusc.get("sample_data", cur_sample["data"][cam_name])
    cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    cam_ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
    intr = np.array(cam_calib["camera_intrinsic"])
    extr = get_extrinsic_wrt_lidar_ego_pose_of_sync_ts(cam_calib, cam_ego_pose, lidar_ego_pose)
    return {
        "path": Path(cam_data["filename"]),
        "calibration": {
            "camera_type": "PerspectiveCamera",
            "extrinsic": extr,
            "intrinsic": np.array([intr[0, 2], intr[1, 2], intr[0, 0], intr[1, 1]]),
        },
    }


def get_extrinsic_wrt_lidar_ego_pose_of_sync_ts(cam_calib, cam_ego_pose, lidar_ego_pose):
    """Calculate the the delta pose between cur camera and lidar and apply it to the camera's extrinsic as the new extrinsic.
    Equation:
        T_e2_c = T_e2_e1 * T_e1_c
               = T_e2_w * T_w_e1 * T_e1_c
               = (T_w_e2)' * T_w_e1 * T_e1_c
        , where e1 is the ego-coord-sys at the time of camera exposure; while e2 is the ego-coord-sys at the time of lidar saves point-bag
        and c is the camera-coord-sys.
        T_w_e1 is the ego_pose at the time of camera exposure; while T_w_e2 is the ego_pose at the time of lidar saves point-bag
        and T_e1_c is the extrinsic of camera.

    Parameters
    ----------
    cam_calib : _type_
        _description_
    cam_ego_pose : _type_
        _description_
    lidar_ego_pose : Dict
        {
            "rotation": np.ndarray, of shape (3, 3),
            "translation: np.ndarray, of shape (3, )
        }

    Returns
    -------
    _type_
        _description_
    """
    T_e1_c = xyzq2mat(
        *cam_calib["translation"], *cam_calib["rotation"][1:], cam_calib["rotation"][0], as_homo=True
    )  # cam_ego_pose["rotation"] is in wxyz
    T_w_e1 = xyzq2mat(
        *cam_ego_pose["translation"], *cam_ego_pose["rotation"][1:], cam_ego_pose["rotation"][0], as_homo=True
    )
    T_w_e2 = T4x4(lidar_ego_pose["rotation"], lidar_ego_pose["translation"])
    T_e2_w = np.linalg.inv(T_w_e2)
    T_e2_c = T_e2_w @ T_w_e1 @ T_e1_c
    return (T_e2_c[:3, :3], T_e2_c[:3, 3])


def build_3d_boxes(nusc: NuScenes, cur_sample, lidar_ego_pose):
    if "anns" not in cur_sample:
        return []

    T_w_e = T4x4(lidar_ego_pose["rotation"], lidar_ego_pose["translation"])
    T_e_w = np.linalg.inv(T_w_e)

    annos = []
    for ann in cur_sample["anns"]:
        ann_info = nusc.get("sample_annotation", ann)
        valid_flag = (ann_info['num_lidar_pts'] + ann_info['num_radar_pts']) > 0  # this logic and naming is aligned with mmdet3d and CMT
        if not valid_flag:
            # logger.debug(f"{ann_info['num_lidar_pts']}, {ann_info['num_radar_pts']}, {ann_info['visibility_token']}")
            continue

        velocity_world = nusc.box_velocity(ann_info["token"])
        if np.any(np.isnan(velocity_world)):
            velocity_ego = np.zeros(3)
        else:
            velocity_ego = T_e_w[:3, :3] @ velocity_world[:, None]

        ann_rot = np.eye(4)
        ann_rot[:3, :3] = Rotation.from_quat(np.array(ann_info["rotation"])[[1, 2, 3, 0]]).as_matrix()
        ann_rot_ego = T_e_w @ ann_rot
        ann_translation_ego = T_e_w @ np.array([*ann_info["translation"], 1])[:, None]
        annos.append(
            {
                "class": ann_info["category_name"],
                "attr": {},
                "size": np.array(ann_info["size"])[[1, 0, 2]].tolist(),
                "rotation": ann_rot_ego[:3, :3],
                "translation": ann_translation_ego.flatten()[:3],
                "track_id": ann_info["instance_token"],
                "velocity": velocity_ego.flatten()[:3],
            }
        )
    return annos


def build_3d_polylines(nusc: NuScenes, nusc_map: Dict[str, Dict[str, dict]], cur_sample, lidar_ego_pose):
    scene_id = nusc.get("scene", cur_sample["scene_token"])["name"]
    timestamp = cur_sample["timestamp"]
    map_data = nusc_map[scene_id]
    frame = map_data.get(timestamp)
    elements = []
    if frame:
        for data_type, vertices_list in frame.items():
            if data_type == "centerline":
                continue
            
            for i, vertices in enumerate(vertices_list):
                rotated_vertices = vertices @ np.array([[0, 1], [-1, 0]]).T # 不知道为啥，这里要旋转一个 90度 结果看起来才正确
                ele = {
                    "class": data_type,
                    "attr": {},
                    "points": np.concatenate((rotated_vertices, np.zeros((len(rotated_vertices), 1))), axis=1),
                    "track_id": f"{cur_sample['token']}-{data_type}-{i}",
                }
            
                elements.append(ele)
    return elements

def save_pickle(data, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
