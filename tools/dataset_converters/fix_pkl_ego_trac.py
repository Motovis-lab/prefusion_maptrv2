import mmengine
from scipy.spatial.transform import Rotation
from pathlib import Path as P
R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()


scene_root = "/home/wuhan/mv_4d_data"
scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("mv_4d_infos_2023*") if p.is_file()]

for scene_name in scene_names:
    data = mmengine.load(P(scene_root) / P(scene_name))
    name = scene_name[12:-4]
    frame_ids = list(data[name]['frame_info'].keys())
    for frame_id in frame_ids:
        data[name]['frame_info'][frame_id]['ego_pose']['rotation'] = data[name]['frame_info'][frame_id]['ego_pose']['rotation'] @ R_nus

    mmengine.dump(data, P(scene_root) / P(scene_name))