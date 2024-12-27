import mmengine
from pathlib import Path as P

file_names = P("/mnt/ssd1/wuhan/prefusion/data/MV4D_12V3L/").glob("mv_4d_infos_2023*.pkl")

fish_filenames = ["VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT", "VCAMERA_FISHEYE_BACK", "VCAMERA_FISHEYE_RIGHT", "camera1", "camera5", "camera8", "camera11"]

for file in file_names:
    data = mmengine.load(file)
    scene_name = file.stem[12:]
    for key in data[scene_name]['scene_info']['camera_mask']:
        if key in fish_filenames:
            data[scene_name]['scene_info']['camera_mask'][key] = f"ego_mask/{key}.png"
        if key not in fish_filenames:
            data[scene_name]['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
        elif key in fish_filenames:
            data[scene_name]['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'

    mmengine.dump(data, file)