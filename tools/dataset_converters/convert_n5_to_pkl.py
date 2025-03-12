import cv2
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation

from copious.io.fs import write_pickle


def load_calpara(file_path):
    with open(file_path, 'r') as f:
        extr_motovis = {}
        intr_motovis = {}
        for line in f:
            line_list = line.strip().split()
            if len(line_list) == 7:
                extr_motovis[line_list[0]] = np.array(line_list[1:], dtype=np.float64)
            if len(line_list) == 9:
                intr_motovis[line_list[0]] = np.array(line_list[1:], dtype=np.float64)
        return extr_motovis, intr_motovis
    
def get_extrinsic_from_calpara(extr_motovis):
    h, w, z, pitch, roll, yaw = extr_motovis
    R = Rotation.from_euler('xyz', (pitch, roll, yaw), degrees=False).as_matrix()
    t = np.float32([h, w, z])
    return R, t

def get_intrinsic_from_calpara(intr_motovis):
    p0, p1, p2, p3, cx, cy, fx, fy = intr_motovis
    return cx, cy, fx, fy, p0, p1, p2, p3


R_ego_motovis = Rotation.from_euler('xyz', (0, 0, -90), degrees=True).as_matrix()

def prepare_calibration(scene_root: Path):
    extr_motovis, intr_motovis = load_calpara(Path(scene_root) / 'calpara.txt')
    calibration = {}
    for cam_id in extr_motovis:
        R_motovis_c, t_motovis = get_extrinsic_from_calpara(extr_motovis[cam_id])
        R_ego_c = R_ego_motovis @ R_motovis_c
        t_ego = t_motovis @ R_ego_motovis.T
        calibration[cam_id.lower()] = {
            'extrinsic': (R_ego_c, t_ego),
            'intrinsic': get_intrinsic_from_calpara(intr_motovis[cam_id]),
            'camera_type': 'FisheyeCamera'
        }
    return calibration


def prepare_camera_mask(ego_mask_path: Path):
    cameras = ['front', 'left', 'right', 'rear']
    camera_mask_dict = {}
    for cam_id in cameras:
        mask_json = json.load(open(ego_mask_path / f'{cam_id}_ego_mask.json'))
        mask = np.ones((mask_json['imageHeight'], mask_json['imageWidth']))
        mask = cv2.fillPoly(mask, [np.round(mask_json['shapes'][0]['points']).astype(np.int32)], 0)
        mask_path = ego_mask_path / f'{cam_id}_ego_mask.png' 
        plt.imsave(mask_path, mask)
        camera_mask_dict[cam_id] = str(mask_path.relative_to(ego_mask_path.parent))
    return camera_mask_dict


def prepare_frame_info(scene_root: Path):
    cameras = {
        'front': 'FrontRoad',
        'left': 'LeftRoad',
        'right': 'RightRoad',
        'rear': 'RearRoad'
    }
    # gen timestamps
    ts_sets = []
    for cam_id in cameras:
        all_jpgs = sorted(Path(scene_root / cameras[cam_id]).glob('*.jpg'))
        ts_sets.append([jpg.name[:-4] for jpg in all_jpgs])
    ts_valid = sorted(set(ts_sets[0]) & set(ts_sets[1]) & set(ts_sets[2]) & set(ts_sets[3]))

    frame_info = {}
    for ts in tqdm(ts_valid):
        frame_info[ts] = {'camera_image': {
            cam_id: str(Path(scene_root / cameras[cam_id] / f'{ts}.jpg').relative_to(scene_root.parent))
            for cam_id in cameras
        }}
    return frame_info


def prepare_scene(scene_root):
    return {scene_root.name: {
        "scene_info": {
            "calibration": prepare_calibration(scene_root),
            "camera_mask": prepare_camera_mask(scene_root.parent / 'ego_mask'),
        },
        "frame_info": prepare_frame_info(scene_root),
    }}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    args = parser.parse_args()
    scene_root = Path(args.scene_root)
    scene_info = prepare_scene(scene_root)
    write_pickle(scene_info, scene_root.parent / f"{scene_root.name}.pkl")