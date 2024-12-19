import argparse
import pickle
from typing import Dict
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from loguru import logger
from scipy.spatial.transform import Rotation
from nuscenes.nuscenes import NuScenes
from easydict import EasyDict as edict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefusion-info-pkl-path", type=Path, required=True)
    parser.add_argument("--cmt-info-pkl-path", type=Path, required=True)
    parser.add_argument("--nusc-data-root", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="+", required=True)
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})

args = parse_arguments()


def main():
    cmt_df = load_cmt_box_info(args.cmt_info_pkl_path)
    pfs_df = load_prefusion_box_info(args.prefusion_info_pkl_path)
    a = 100


def build_token2scene_mapping() -> Dict:
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_data_root, verbose=True)
    _mapping = {}
    for scene in nusc.scene:
        if scene["name"] not in args.scene_ids:
            continue

        cur_sample = nusc.get("sample", scene["first_sample_token"])

        while cur_sample["next"] != "":
            _mapping[cur_sample["token"]] = scene["name"]
            cur_sample = nusc.get("sample", cur_sample["next"])

        _mapping[cur_sample["token"]] = scene["name"] # the last token

    logger.info(f"Finished building token2scene mapping: {len(set(_mapping.values()))} scenes ({len(_mapping)} tokens) have been mapped.")

    return _mapping


class_mapping = {
    "vehicle.bicycle": "bicycle",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "movable_object.trafficcone": "traffic_cone",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_prefusion_box_info(pkl_path):
    info = load_pickle(pkl_path)
    all_data = {}
    for scene_id, scene_data in info.items():
        if scene_id not in args.scene_ids:
            continue
        all_data[scene_id] = prefusion_scene_to_df(scene_id, scene_data)
    all_data_df = pd.concat(list(all_data.values())).reset_index()
    logger.info(f"Finished loading prefusion box info, {len(all_data)} scenes ({len(all_data_df)} boxes) has been loaded.")
    return all_data_df


def prefusion_scene_to_df(scene_id: str, scene_data: Dict) -> pd.DataFrame:
    box_data = [
        {'scene_id': scene_id, 'timestamp': frm_id, 'token': frm_data['sample_token'], 'class': class_mapping[box['class']], 'translation': box['translation'], 'size': box['size'], 'yaw': get_yaw(box['rotation'])} 
        for frm_id, frm_data in scene_data['frame_info'].items() for box in frm_data['3d_boxes']
        if box['class'] in class_mapping
    ]
    return pd.DataFrame.from_records(box_data)


def get_yaw(rot_mat):
    return Rotation.from_matrix(rot_mat).as_euler('XYZ', degrees=False)[2]


def load_cmt_box_info(pkl_path):
    info = load_pickle(pkl_path)
    token2scene = build_token2scene_mapping()
    all_data = []
    for sample in info['infos']:
        if sample['token'] not in token2scene:
            continue

        token = sample['token']
        scene_id = token2scene[token]
        timestamp = str(sample["timestamp"] // 1000)
        
        all_data.extend(
            [
                {'scene_id': scene_id, 'timestamp': timestamp, 'token': token, 'class': cls, 'translation': box[:3], 'size': box[3:6], 'yaw': box[6]}
                for cls, box, isvalid in zip(sample['gt_names'], sample['gt_boxes'], sample['valid_flag'])
                if isvalid and cls in set(class_mapping.values())
            ]
        )
    
    all_data_df = pd.DataFrame.from_records(all_data)
    logger.info(f"Finished loading CMT box info, {len(all_data)} boxes has been loaded.")
    return all_data_df


def generate_comparison_report(pfs_df, cmt_df):
    pass


if __name__ == "__main__":
    main()
