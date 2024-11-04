import argparse
from typing import Dict, List, Tuple, Sequence, Any
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger
from easydict import EasyDict as edict
from copious.io.fs import parent_ensured_path, read_yaml, read_json, write_pickle
from copious.io.parallelism import maybe_multithreading
from copious.cv.geometry import xyzq2mat, euler2mat, points3d_to_homo

from prefusion.dataset.utils import T4x4


class Mat4x4(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Sequence[Any],
        option_string: str = None,
    ) -> None:
        if len(values) == 1 and isinstance(values[0], np.ndarray):
            setattr(namespace, self.dest, values[0])
        else:
            assert len(values) == 16
            setattr(namespace, self.dest, np.array(values).reshape(4, 4))


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv4d-data-root", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="*")
    parser.add_argument("--save-path", type=parent_ensured_path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calib-filename", default="vcalib_center.yml")
    parser.add_argument("--camera-root-name", default="vcamera")
    parser.add_argument(
        "--ego-coordsys-align-mat",
        nargs=16,
        action=Mat4x4,
        type=float,
        default=np.eye(4),
        help="4x4 mat, 用来将原本的右前上ego系变换为prefusion框架要求的前左上ego系",
    )
    parser.add_argument("--timestamp-range", type=int, nargs=2)
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})


args = parse_argument()


def main():
    scene_ids = args.scene_ids or list([p.parent.name for p in args.mv4d_data_root.glob("*/4d_anno_infos")])
    logger.info(f"Total number of scenes: {len(scene_ids)}, scene_list: {scene_ids}")
    infos = {sid: prepare_scene(args, args.mv4d_data_root / sid) for sid in scene_ids}
    write_pickle(infos, args.save_path)
    logger.info(f"All the info has been saved to {args.save_path}")


def prepare_scene(args, scene_root: Path) -> Dict:
    logger.info(f"Generating info pkl for scene {scene_root.name}")
    calib = prepare_calibration(scene_root)
    return {
        "scene_info": {
            "calibration": calib,
            "camera_mask": prepare_camera_mask(scene_root),
            "depth_mode": prepare_depth_mode(scene_root),
        },
        "meta_info": {
            "space_range": {
                "map": [36, -12, -12, 12, 10, -10],
                "det": [36, -12, -12, 12, 10, -10],
                "occ": [36, -12, -12, 12, 10, -10],
            },
            "time_range": 2,
            "time_unit": 1e-3,
        },
        "frame_info": prepare_all_frame_infos(args, scene_root, calib),
    }


def prepare_calibration(scene_root: Path) -> Dict:
    raw_calib = edict(read_yaml(scene_root / args.calib_filename)["rig"])
    calib = {}
    for sensor_id, sensor_info in raw_calib.items():
        standard_extrinsic = args.ego_coordsys_align_mat @ xyzq2mat(*sensor_info.extrinsic, as_homo=True)
        calib[sensor_id] = {"extrinsic": (standard_extrinsic[:3, :3], standard_extrinsic[:3, 3])}
        if "camera" in sensor_info.sensor_model.lower():
            if "fisheye" in sensor_info.sensor_model.lower():
                calib[sensor_id]["camera_type"] = "FisheyeCamera"
                calib[sensor_id]["intrinsic"] = np.array(
                    sensor_info.pp + sensor_info.focal + sensor_info.inv_poly[:4], dtype=np.float32
                )
            else:
                calib[sensor_id]["camera_type"] = "PerspectiveCamera"
                calib[sensor_id]["intrinsic"] = np.array(sensor_info.pp + sensor_info.focal, dtype=np.float32)
    return calib


def prepare_camera_mask(scene_root: Path) -> Dict:
    camera_mask_dir = scene_root / "self_mask" / "camera"
    return {cam_path.stem: cam_path.relative_to(scene_root.parent) for cam_path in camera_mask_dir.iterdir()}


def prepare_depth_mode(scene_root: Path) -> Dict:
    return {p.stem: "d" for p in (scene_root / "camera").iterdir()}


def prepare_ego_poses(scene_root: Path) -> Dict[int, np.ndarray]:
    trajectory = np.loadtxt(scene_root / "trajectory.txt")
    poses = {}
    for t, *xyzq in trajectory:
        mat = xyzq2mat(*xyzq, as_homo=True)
        standard_mat = mat @ np.linalg.inv(args.ego_coordsys_align_mat) # i.e. R_w_e' = R_w_e @ R_e_e' (e is right-front-up, e' is front-left-up)
        poses[int(t)] = {"rotation": standard_mat[:3, :3], "translation": standard_mat[:3, 3]}
    return poses


def prepare_all_frame_infos(args, scene_root: Path, calib: dict) -> Dict:
    common_ts = read_common_ts(scene_root)

    if args.timestamp_range is not None and len(args.timestamp_range) == 2:
        common_ts = [ts for ts in common_ts if args.timestamp_range[0] <= ts <= args.timestamp_range[1]]
        logger.info(f"Timestamp Range has been set to {args.timestamp_range}, only {len(common_ts)} frames will kept in the dataset.")

    frame_infos = {}
    data_args = [(scene_root, ts) for ts in common_ts]
    res = maybe_multithreading(prepare_object_info, data_args, num_threads=args.num_workers, use_tqdm=True)
    ego_poses = prepare_ego_poses(scene_root)
    for ts, boxes, polylines in res:
        transform_velocity_to_ego_(boxes, ego_poses[ts])  # box velo from upstream is assumed to be direction-vector in the world sys, so we need to transform it to ego sys
        frame_infos[str(ts)] = {  # convert to str to make it align with the design
            "camera_image": prepare_camera_image_paths(scene_root, ts, calib),
            "3d_boxes": boxes,
            "3d_polylines": polylines,
            "ego_pose": ego_poses[ts],
            "timestamp_window": [None],  # TODO: populate previous N frames info (window size is time_range)
        }
    return frame_infos


def read_common_ts(scene_root: Path) -> List[int]:
    ts_info = read_json(scene_root / "4d_anno_infos" / "ts.json")
    # TODO: should specify a list of sensors, the common ts will be the intersection of timestamps of these sensors
    return [int(i["lidar"]) for i in ts_info]


def prepare_camera_image_paths(scene_root: Path, ts: int, calib: dict) -> Dict[str, str]:
    camera_image_paths = {}
    for p in (scene_root / args.camera_root_name).rglob(f"*{ts}*.jpg"):
        cam_id = p.parent.name
        if cam_id in calib:
            camera_image_paths[cam_id] = p.relative_to(scene_root.parent)
    return camera_image_paths


def prepare_object_info(scene_root: Path, ts: int) -> Tuple[List[dict], List[dict]]:
    object_info = read_json(scene_root / "4d_anno_infos" / "4d_anno_infos_frame" / "frames_labels" / f"{ts}.json")
    boxes, polylines = [], []
    for obj in object_info:
        obj = edict(obj)
        if obj.geometry_type == "box3d":
            boxes.append(convert_box3d_format(obj))
        elif obj.geometry_type == "polyline3d":
            polylines.append(convert_polyline3d_format(obj))
    return ts, boxes, polylines


def transform_velocity_to_ego_(boxes: List[Dict], ego_pose: Dict[str, np.ndarray]) -> None:
    rot_world_to_ego = np.linalg.inv(ego_pose["rotation"])
    for bx in boxes:
        bx["velocity"] = (bx["velocity"][None] @ rot_world_to_ego.T)[0]


############################################################################################################################################
# TODO: put rearrange_object_class_attr_ and unify_longer_shorter_edge_definition_ to Bbox3DLoader, and use config to control the behavior
# rearrange_object_class_attr_:
#   attr_translate = {
#       "attr.traffic_facility.box.type",
#       "attr.traffic_facility.soft_barrier.type",
#       "attr.traffic_facility.hard_barrier.type"
#       "attr.parking.indoor_column.shape"
#   }
#
# unify_longer_shorter_edge_definition_:
#     standard direction: X-axis perpendicular to the longer edge
#     steps: check if X-axis not perpendicular to the longer edge, if yes, rotate the box (RotMat) by 90 deg, and switch scale[0] and scale[1]
############################################################################################################################################


def convert_box3d_format(box_info: Dict):
    Rt = euler2mat(*box_info.geometry.rot_xyz, degrees=False, as_homo=True)
    Rt[:3, 3] = box_info.geometry.pos_xyz
    standard_Rt = args.ego_coordsys_align_mat @ Rt
    
    return {
        "class": box_info.obj_type,
        "attr": box_info.obj_attr,
        "size": box_info.geometry.scale_xyz,
        "rotation": standard_Rt[:3, :3],
        "translation": standard_Rt[:3, 3],
        "track_id": box_info.obj_track_id,
        "velocity": np.array(box_info.velocity, dtype=np.float32),
    }


def convert_polyline3d_format(polyline_info: Dict):
    _pts = np.array(polyline_info.geometry, dtype=np.float32)
    if _pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    standard_points = points3d_to_homo(_pts) @ args.ego_coordsys_align_mat.T
    return {
        "class": polyline_info.obj_type,
        "attr": polyline_info.obj_attr,
        "points": standard_points[:, :3],
        "track_id": polyline_info.obj_track_id,
    }


if __name__ == "__main__":
    main()
