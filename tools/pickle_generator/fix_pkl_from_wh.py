# fix pickle; both camera & lidar
import pickle
from mtv4d.annos_4d.misc import read_ego_paths
from mtv4d.utils.calib_base import read_cal_data
from pathlib import Path as P
import os.path as op
import numpy as np
from scipy.spatial.transform import Rotation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='/ssd1/MV4D_12V3L')  # maybe dataset root is better
parser.add_argument('--scene-name')  # single one, not list
parser.add_argument('--output_pickle_path')  # single one, not list
parser.add_argument('--input_pickle_path')  # single one, not list


def transform_pose_from_RFU_to_FLU(T):
    # R = np.array([
    #     [0,-1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1]
    # ])
    T_t = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # T_out = np.eye(4)
    # T_out[:3, 3] = [T[1, 3], -T[0, 3], T[2, 3]]
    # T_out[:3, :3] = T[:3, :3] @ R

    return T @ T_t


def handle_camera(frame_camera, Twes):
    for q, v in frame_camera.items():
        ts = float(P(v).stem)
        frame_camera[q] = {
            'path': v,
            'Twe': translate_Twrer_to_Wrfu_Erfu(Twes[float(ts)]),
            'timestamp': ts
        }


def find_previous_20_timestamps(timestamps, ts):
    a = np.array(sorted([float(i) for i in timestamps]))
    idx = int(np.where(a == ts)[0])
    return a[max(0, idx - 20):idx]


def handle_lidar(frame, timestamps, Twes):
    frame_lidar = frame['lidar_points']
    path =  P(args.data_root) /frame_lidar['lidar1']
    if not path.exists():
        return False
    ts = float(P(frame_lidar['lidar1']).stem)
    picked_timestamps = find_previous_20_timestamps(timestamps, ts)
    picked_timestamps = [p_ts for p_ts in picked_timestamps
                         if (P('/ssd1/MV4D_12V3L') / op.join(P(frame_lidar['lidar1']).parent, f'{int(p_ts)}.pcd')).exists()]

    frame['lidar_points'] = {
        "lidar1": frame_lidar['lidar1'],
        'lidar1_sweeps': [{
            'path': op.join(P(frame_lidar['lidar1']).parent, f'{int(p_ts)}.pcd'),
            "Twe": translate_Twrer_to_Wrfu_Erfu(Twes[float(p_ts)]),
            "timestamp": p_ts
        } for p_ts in picked_timestamps]
    }
    return True

def Rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def translate_Twrer_to_Wrfu_Erfu(T):
    R_nus = Rotation.from_euler("xyz", angles=(0, 0, 90), degrees=True).as_matrix()
    R_t = np.eye(4)
    R_t[:3, :3] = R_nus
    R1 = (R_t.T @ T)[:3, :3] @ R_nus
    t1 = (R_t.T @ T)[:3, 3]
    T_new = np.eye(4)
    T_new[:3, :3] = R1
    T_new[:3, 3] = t1
    return T_new


def handle_ego_pose(frame, Twes):
    ts = float(P(frame['lidar_points']['lidar1']).stem)
    ego_pose = Twes[ts]
    ego_pose_flu = translate_Twrer_to_Wrfu_Erfu(ego_pose)
    frame['ego_pose'] = {
        'rotation': ego_pose_flu[:3, :3],
        'translation': ego_pose_flu[:3, 3]
    }


def handle_camera_intrinsic(scene_info_calibration):
    for sensor, cal in scene_info_calibration.items():
        if 'cam' in sensor.lower():
            cal['intrinsic'] = list(cal['intrinsic'])


def process_one_scene_with_pickle(data_root, scene_name, input_pickle_path, output_pickle_path):

    with open(input_pickle_path, 'rb') as f:
        a = pickle.load(f)
    frame_info = a[scene_name]['frame_info']
    scene_info = a[scene_name]['scene_info']
    timestamps = frame_info.keys()

    calib = read_cal_data(f'{data_root}/{scene_name}/calibration_center.yml')
    traj_p = f'{data_root}/{scene_name}/trajectory.txt'
    Twes, _ = read_ego_paths(traj_p)

    for ts in sorted(frame_info.keys()):
        frame = frame_info[ts]
        handle_ego_pose(frame, Twes)
        # handle_camera(frame['camera_image'], Twes)  # both not good function, because change the content of the parameter
        is_lidar_exist = handle_lidar(frame, timestamps, Twes)
        if not is_lidar_exist:
            frame_info.pop(ts)
            continue
        handle_camera_intrinsic(scene_info['calibration'])
    print(len(frame_info))
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(a, f)


if __name__ == '__main__':
    # --------------------------------------------------------------
    # scene_name = '20231027_185823'
    args = parser.parse_args()
    # --------------------------------------------------------------
    # 1. get ego pose
    # 2. fix camera pose to world_left
    # 3. fix data
    process_one_scene_with_pickle(args.data_root, args.scene_name,
                                  args.input_pickle_path, args.output_pickle_path)


