import pickle  
import numpy as np
from mtv4d.utils.io_base import read_pickle, write_pickle
from scipy.spatial.transform import Rotation
from tqdm import tqdm
input_path = "/ssd1/data/4d/cmt_pkl_lidar_ego.pkl"  
output_sample_path = "/ssd1/data/4d/mv4d_infos_tmp_mini1.pkl"  
with open(input_path, "rb") as f:  
    a = pickle.load(f)  
a = a["infos"]  
print(a[1].keys())  
# 'lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag'])

with open(output_sample_path, "rb") as f:  
    b = pickle.load(f)  
# b['20230823_110018']['scene_info'].keys() dict_keys(['calibration', 'camera_mask', 'depth_mode'])
c = b["20230823_110018"]["frame_info"]["1692759619764"]  
# dict_keys(['camera_image', '3d_boxes', '3d_polylines', 'ego_pose', 'timestamp_window', 'lidar_points'])
# next i'll match all these two things
print(b.keys())   
def from_pyquaternion_to_R(q):
    # q = Rotation.from_matrix(R).as_quat()  # xyzw
    # return q[[3, 0, 1, 2]]
    if len(q) == 4:
        return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return q

def get_T_from_Rt(R, t):
    T = np.eye(4)
    T[:3, :3] = from_pyquaternion_to_R(R)
    T[:3, 3] = t
    return T

cameras = ['camera1', 'camera11', 'camera5', 'camera8']
def get_fusion_frame_from_cmt_pkl(input_frame, example=None):  
    output_frame = {  
        "camera_image": {cam: input_frame['cams'][cam]['data_path'][9:] for cam in cameras}, 
        "3d_boxes": [{  
            'class': input_frame['gt_names'][i],   
            "attr": input_frame['gt_names'][i],  # 这个暂时没用，之后要整理一个4d_frame_info 直接转到pickle的数据
            "size": input_frame['gt_boxes'][i][3:6].tolist(),   # box center; not bottom center
            "rotation": Rotation.from_euler('XYZ', np.array([0, 0, input_frame['gt_boxes'][i][6]])).as_matrix(),  
            'translation': input_frame['gt_boxes'][i][:3],  
            "track_id": 0,   # 有这个限制，必须从4d_frame_info开始转  # TODO: you have to generate from 4d frame info
            "velocity": input_frame['gt_velocity'][i],  
        } for i, box in enumerate(input_frame['gt_boxes'])
        ],  
        "3d_polylines": [  
            {  
                'class': 1, 
                'attr': 1, 
                'points': 1,
            } for poly in []
        ],
        "ego_pose": {"rotation": from_pyquaternion_to_R(input_frame['ego2global_rotation']), "translation": input_frame['ego2global_translation']},
        # 虽然最好直接转到lidar系，这样代码都不用改，会更加舒服一点
        "timestamp_window": [None],  # 也不知道这个用来干嘛
        "lidar_points": {
            "lidar1": input_frame['lidar_path'][9:],  # in ego coord
            "lidar1_sweeps": [
                {'path': swp['data_path'][9:],
                 'Twe': get_T_from_Rt(from_pyquaternion_to_R(swp['ego2global_rotation']), swp['ego2global_translation']),   
                 'timestamp': swp['timestamp'],  
                 }  for swp in input_frame['sweeps']  
            ],  
        },  
    }
    return output_frame


# get_fusion_frame_from_cmt_pkl(a[0], c)

from mtv4d.utils.calib_base import read_cal_data
def main(input_pkl_path, output_pkl_path):
    scene_name = '20230823_110018'
    a = read_pickle(input_pkl_path)
    a['infos'] = [p for p in a['infos'] if p['lidar_path'].split('/')[2] == scene_name][:4]
    for info in a['infos']:
        info['timestamp'] = str(int(info['timestamp']))
    output_frame_info = {}
    for info in tqdm(a['infos']):
        ts = info['timestamp']
        output_frame_info[ts] = get_fusion_frame_from_cmt_pkl(info)
    calib = read_cal_data(f'/ssd1/data/4d/{scene_name}/calibration_center.yml')
    output = {}
    output[scene_name] = {
        'scene_info': {
            'calibration':
                {cam:{
                        'extrinsic': (from_pyquaternion_to_R(info['cams'][cam]['sensor2ego_rotation']), info['cams'][cam]['sensor2ego_translation']), 
                        'camera_type': 'FisheyeCamera', 
                        'intrinsic': info['cams'][cam]['cam_intrinsic']['pp'] + info['cams'][cam]['cam_intrinsic']['focal'] + info['cams'][cam]['cam_intrinsic']['D'].tolist()
                    } for cam in cameras} | {'lidar1': {
                        # 'extrinsic': (from_pyquaternion_to_R(info['lidar2ego_rotation']), info['lidar2ego_translation']), 
                        'extrinsic': (calib['lidar1']["T_es"][:3, :3], calib['lidar1']["T_es"][:3, 3]), 
                    }
                },
                'camera_mask': {
                    cam: f"{scene_name}/self_mask/camera/{cam}.png" for cam in cameras
                },
                'depth_mode': {'camera5': 'd', 'camera4': 'd', 'camera15': 'd', 'camera2': 'd', 'camera11': 'd', 'camera8': 'd', 'camera3': 'd', 'camera6': 'd', 'camera12': 'd', 'camera1': 'd', 'camera13': 'd', 'camera7': 'd'}}, 
        'meta_info': {'space_range':{'map': [36, -12, -12, 12, 10, -10], 'det': [36, -12, -12, 12, 10, -10], 'occ': [36, -12, -12, 12, 10, -10]}, 'time_range': 2, 'time_unit': 0.001},
        'frame_info': output_frame_info
    }
    write_pickle(output, output_pkl_path)  


main( "/ssd1/data/4d/cmt_pkl_lidar_ego.pkl",  "/ssd1/data/4d/cmt_prefusion_dbg_mini.pkl"  )



