import mmengine 
from pathlib import Path as P

scene_root = "/home/wuhan/mv_4d_data"
scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("mv_4d_infos_2023*") if p.is_file()]


all_train_data = dict()
all_val_data = dict()
for scene_name in scene_names:
    data = mmengine.load(P(scene_root) / P(scene_name))
    name = scene_name[12:-4]
    if name not in ["20231028_150815", "20231107_123645"]:
        for key in data[name]['scene_info']['calibration']:
            if "PERSPECTIVE" in key:
                data[name]['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
            elif "FISHEYE" in key:
                data[name]['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'
        all_train_data.update(data)
    else:
        for key in data[name]['scene_info']['calibration']:
            if "PERSPECTIVE" in key:
                data[name]['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
            elif "FISHEYE" in key:
                data[name]['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'
        all_val_data.update(data)


mmengine.dump(all_train_data, "/home/wuhan/mv_4d_data/mv_4d_infos_train.pkl")
mmengine.dump(all_val_data, "/home/wuhan/mv_4d_data/mv_4d_infos_val.pkl")