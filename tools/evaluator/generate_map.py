import json
from pathlib import Path as P
import cv2
import mmengine
from mtv4d.utils.box_base import to_corners_9
from mtv4d.utils.calib_base import read_cal_data
from mtv4d.utils.draw_base import draw_boxes, to_corners_7
import numpy as np
from mtv4d.utils.geo_base import transform_pts_with_T
from mtv4d.utils.sensors import get_camera_models
from scipy.spatial.transform import Rotation
from easydict import EasyDict
from tmp_scripts.fix_pkl import Rt_to_T


def process_single_box_gt(ts, box):
    return EasyDict(
        sample_token=ts,
        translation=box['translation'],
        size=box['size'],
        rotation=box['rotation'],
        velocity=box['velocity'][:2],
        # ego_translation=box[''],
        num_pts=100,
        detection_name=box['class'],
        detection_score=box['score'],
        attribute_name=box['class'],
    )


def precess_single_box_pred(ts, box):
    return EasyDict(
        sample_token=ts,
        translation=box['translation'],
        size=box['size'],
        rotation=box['rotation'],  # nuscenes shi wxyz, better transform to that format
        velocity=box['velocity'][:2],
        # ego_translation=box[''],
        num_pts=100,
        detection_name=box['class'],
        detection_score=box['score'],
        attribute_name=box['class'],
    )


def process_pred_data_to_coco(ts, data):
    return [precess_single_box_pred(ts, box) for box in data['pred']['bboxes']]


def input_dir(input_dir):
    output = []
    a = sorted(P(input_dir).glob('*.json'))
    for path in a:
        ts = path.stem
        with open(str(path)) as f:
            data = json.load(f)
        output += [process_pred_data_to_coco(ts, data)]
    return output


def draw_json(input_dir, output_json_path="/tmp/1234/pred.json", draw_im=True):
    sensor, calib_path = "camera8", "/ssd1/MV4D_12V3L/20230823_110018/calibration_center.yml"
    a = sorted(P(input_dir).glob('*.json'))
    im_dir = f"/ssd1/MV4D_12V3L/20230823_110018/camera/{sensor}"
    scene_infos = mmengine.load('/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20230823_110018_evaluate.pkl')[
        '20230823_110018']
    Tes = Rt_to_T(scene_infos['scene_info']['calibration'][sensor]['extrinsic'][0],
                  scene_infos['scene_info']['calibration'][sensor]['extrinsic'][1])
    Tse = np.linalg.inv(Tes)
    output = []
    for path in a:
        ts = path.stem
        with open(str(path)) as f:
            data = json.load(f)
        output += [process_pred_data_to_coco(data)]
        if draw_im:
            im_path = str(P(im_dir) / f"{ts}.jpg")
            im = cv2.imread(str(im_path))

            def to_1(rotation):
                xyz = Rotation.from_matrix(np.array(rotation).reshape(3, 3)).as_euler("XYZ")
                return xyz.tolist()

            box9d = [i['translation'] + i['size'] + to_1(i['rotation']) for i in data['pred']['bboxes']]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, Tse).reshape(-1, 3)
            camera_model = get_camera_models(calib_path, [sensor])[sensor]
            corners2d = camera_model.project_points(corners3d)
            im = draw_boxes(im, corners2d)
            cv2.imwrite(f'/tmp/1234/2/{ts}.jpg', im)
    return output


def load_jsons_from_single_hooks(input_dir, output_json_path="/tmp/1234/pred.json"):
    # sensor, calib_path = "camera8", "/ssd1/MV4D_12V3L/20230823_110018/calibration_center.yml"
    a = sorted(P(input_dir).glob('*.json'))
    output = {}
    for path in a:
        ts = path.stem
        with open(str(path)) as f:
            data = json.load(f)  # then change to the data structure
        output[ts] = process_pred_data_to_coco(ts, data)
    return output


def main(input_pred_dir, pkl_path):
    cfg = load_cfg(cfg_path="tools/evaluator/config.json")
    preds = load_jsons_from_single_hooks(input_pred_dir, output_json_path="/tmp/1234/pred.json")
    gts = load_jsons_from_single_hooks(pkl_path, output_json_path="/tmp/1234/pred.json")
    if False:  # draw im
        draw_json(input_dir, output_json_path="/tmp/1234/pred.json")
    #


def load_cfg(cfg_path):
    with open(cfg_path) as f:
        data = json.load(cfg_path)
    return EasyDict(data)


if __name__ == "__main__":
    input_pred_dir = "/home/yuanshiwei/4/prefusion/apa_demo_dumps_20250218/dets/20230823_110018"
    pkl_path = '/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20230823_110018_debug.pkl'
    # transform_input_to_json(input_dir)
    main(input_pred_dir, pkl_path)
