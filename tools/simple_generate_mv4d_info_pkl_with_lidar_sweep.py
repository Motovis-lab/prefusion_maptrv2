import pickle
from pathlib import Path as P
import numpy as np
from scipy.spatial.transform import Rotation


def to_cmt_pkl(src_path, dst_path):
    with open(src_path, 'rb') as f:
        data = pickle.load(f)
    for scene_id, d  in data.items():
        def to_Twe(Rtl):
            T = np.eye(4)
            T[:3, :3] = Rtl['rotation']
            T[:3, 3] = Rtl['translation']
            return T

        Twes = {k: to_Twe(v['ego_pose']) for k, v in d['frame_info'].items()}
        for k, v in d['frame_info'].items():
            for c in ['camera6', 'camera2', 'camera13', 'camera4', 'camera7',
                      'camera15',  'camera3', 'camera12']:
                if c in v['camera_image'].keys():
                    v['camera_image'].pop(c)
            new_cam_dict = {k:v['camera_image'][k] for k in sorted(v['camera_image'].keys())}
            v['camera_image'] = new_cam_dict
            ts = v['camera_image']['camera1'].stem
            v['lidar_points'] = {'lidar1': P(f'{scene_id}/lidar/undistort_static_lidar1') / f'{ts}.pcd'}
            def get_sweep_timestamps(ts):
                timestamps = np.array(sorted(Twes.keys()))
                idx = int(np.where(timestamps == ts)[0][0])  # Assert no error. theoretically no error
                a = timestamps[max(0, idx - 10):idx]  # not contain self
                return list(reversed(a.tolist()))
            v['lidar_points']['lidar1_sweeps'] = [
                {'path': P(f'{scene_id}/lidar/undistort_static_lidar1') / f'{ts}.pcd',
                 'Twe': Twes[ts],
                 'timestamp': ts}
                for ts in get_sweep_timestamps(k)]
    with open(dst_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    to_cmt_pkl(src_path="/ssd1/data/4d/mv4d_infos_tmp.pkl", dst_path="/ssd1/data/4d/mv4d_infos_tmp_mini1.pkl")