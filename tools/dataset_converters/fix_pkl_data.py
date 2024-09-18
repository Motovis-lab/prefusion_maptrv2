import mmengine

data = mmengine.load("/ssd4/home/wuhan/prefusion/data/mv_4d_data/mv_4d_infos_100.pkl")

for key in data['20230823_110018']['scene_info']['calibration']:
    if "PERSPECTIVE" in key:
        data['20230823_110018']['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
    elif "FISHEYE" in key:
        data['20230823_110018']['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'

mmengine.dump(data, "/ssd4/home/wuhan/prefusion/data/mv_4d_data/mv_4d_infos_fix_100.pkl")