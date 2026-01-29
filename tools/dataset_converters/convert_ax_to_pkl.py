import cv2
import json
import argparse

import yaml

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation

from copious.io.fs import write_pickle



camera_mappings = {
    'FISHEYE_FRONT': 'cam_sy_x3j_avm_front/jpg',
    'FISHEYE_LEFT': 'cam_sy_x3j_avm_left/jpg',
    'FISHEYE_RIGHT': 'cam_sy_x3j_avm_right/jpg',
    'FISHEYE_BACK': 'cam_sy_x3j_avm_back/jpg',
}


def load_ax_calib(scene_root):
    # ax_calib = yaml.safe_load(open('./sensors_xcap01_v1.21.yaml'))
    ax_calib = yaml.safe_load(open(scene_root / 'sensors_xcap01_v1.21.yaml'))
    camera_calibs = ax_calib['camera']
    calibration = {}
    for cam_calib in camera_calibs:
        cam_id = cam_calib['camera_config']['frame_id']
        if cam_id not in camera_mappings:
            continue
        camera_config = cam_calib['camera_config']
        pose = camera_config['pose']
        extrinsic = [
            Rotation.from_quat([pose['qx'], pose['qy'], pose['qz'], pose['qw']]).as_matrix(),
            [pose['x'], pose['y'], pose['z']]
        ]
        intrinsic = [
            camera_config['intrinsics']['cx'],
            camera_config['intrinsics']['cy'],
            camera_config['intrinsics']['fx'],
            camera_config['intrinsics']['fy']
        ] + camera_config['distortion']['params']
        calibration[cam_id] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'camera_type': 'FisheyeCamera'
        }
    return calibration


def prepare_camera_mask(ego_mask_path: Path):
    camera_mask_dict = {}
    for cam_id in camera_mappings:
        mask_path = ego_mask_path / f'{cam_id}.png' 
        if not mask_path.exists():
            mask_json = json.load(open(ego_mask_path / f'{cam_id}.json'))
            mask = np.ones((mask_json['imageHeight'], mask_json['imageWidth']))
            mask = cv2.fillPoly(mask, [np.round(mask_json['shapes'][0]['points']).astype(np.int32)], 0)
            plt.imsave(mask_path, mask)
        camera_mask_dict[cam_id] = str(mask_path)
    return camera_mask_dict


def prepare_frame_info(scene_root: Path):
    # gen timestamps
    ts_sets = []
    for cam_id in camera_mappings:
        all_jpgs = sorted(Path(scene_root / camera_mappings[cam_id]).glob('*.jpg'))
        ts_sets.append([jpg.name[:-4] for jpg in all_jpgs])
    ts_valid = sorted(set(ts_sets[0]) & set(ts_sets[1]) & set(ts_sets[2]) & set(ts_sets[3]))

    frame_info = {}
    for ts in tqdm(ts_valid[-1120:]):
        frame_info[ts] = {'camera_image': {
            cam_id: str(Path(scene_root / camera_mappings[cam_id] / f'{ts}.jpg').relative_to(scene_root.parent))
            for cam_id in camera_mappings
        }}
    return frame_info


def prepare_scene(scene_root):
    return {scene_root.name: {
        "scene_info": {
            "calibration": load_ax_calib(scene_root),
            "camera_mask": prepare_camera_mask(scene_root / 'ego_mask'),
        },
        "frame_info": prepare_frame_info(scene_root),
    }}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_root", type=Path)
    args = parser.parse_args()
    print(args)
    scene_root = Path(args.scene_root)
    scene_info = prepare_scene(scene_root)
    write_pickle(scene_info, scene_root / f"{scene_root.name}.pkl")