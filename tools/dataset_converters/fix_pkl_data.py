import mmengine
from pathlib import Path as P

file_names = P("/home/wuhan/prefusion/data/mv_4d_data/").glob("mv_4d_infos_2023*.pkl")

for file in file_names:
    data = mmengine.load(file)
    scene_name = file.stem[12:]
    for key in data[scene_name]['scene_info']['calibration']:
        if "PERSPECTIVE" in key:
            data[scene_name]['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
        elif "FISHEYE" in key:
            data[scene_name]['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'

    mmengine.dump(data, file)