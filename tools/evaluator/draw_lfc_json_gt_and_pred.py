from pathlib import Path as P
from scipy.spatial.transform import Rotation
from mtv4d.utils.sensors import FisheyeCameraModel, get_camera_models
import numpy as np
import cv2
import os.path as op
from mtv4d.utils.geo_base import transform_pts_with_T, Rt2T
from mtv4d.utils.box_base import fbbox_to_box9d, jsonbox_to_box9d
from mtv4d.utils.draw_base import draw_boxes
from mtv4d.annos_4d.misc import read_ego_paths
from mtv4d.utils.box_base import to_corners_9
from mtv4d.utils.io_base import read_json, read_pickle
from mtv4d.utils.sensors import get_camera_models, FisheyeCameraModel
from mtv4d.utils.calib_base import read_cal_data
from tools.evaluator.generate_map import load_jsons_from_single_hooks
from mtv4d.utils.box_base import fbbox_to_box9d
def lfcjson_to_box9d(box):
    from scipy.spatial.transform import Rotation
    p = box['translation']
    s = box['size']
    r = Rotation.from_quat(box['rotation'], scalar_first=True).as_euler("XYZ")
    return list(p) + list(s) + list(r)

def draw_boxes(img, corners2d, text_list=None, clr=None):
    if text_list is None:
        text_list = np.arange(len(corners2d.reshape(-1, 8, 2)))
    for idx, (pts, txt) in enumerate(zip(corners2d.reshape(-1, 8, 2),text_list)):
        pt = pts[:, :2].astype('int')
        # if (pt < 0).any() or (pt > 1600).any():  continue
        clr = clr if clr is not None else (0, 255, 255)
        cv2.polylines(img, [pt[:4]], 2, clr, 2)
        cv2.polylines(img, [pt[4:]], 2, (0, 0, 255), 2)
        for i in range(4):
            cv2.line(img, tuple(pt[i]), tuple(pt[i + 4]), (0, 0, 255), 2)
        # if int(txt) < 20:
        if txt:
            cv2.putText(img, str(txt), tuple(pt[0]), 1, 2, (255, 255,0),2)
    return img

def draw_pickle_lfcjson(json_dir, data_root):
    preds, preds_list = load_jsons_from_single_hooks(json_dir, 'pred', obj_type='bboxes',
                                                     output_json_path="/tmp/1234/pred.json")
    gts, gts_list = load_jsons_from_single_hooks(json_dir, 'gt', obj_type='bboxes',
                                                 output_json_path="/tmp/1234/gt.json")
    sensor_id = 'camera8'
    for key, objs in gts.items():
        calib = read_cal_data('/ssd1/MV4D_12V3L/20230820_105813/calibration_center.yml')
        Trf = np.eye(4)
        Trf[:3, :3] = Rotation.from_euler('Z', [90], degrees=True).as_matrix()
        T_se = calib[sensor_id]['T_se'] @ Trf
        camera_model = get_camera_models('/ssd1/MV4D_12V3L/20230820_105813/calibration_center.yml', None)[sensor_id]
        objs = [i for i in objs if i['detection_name'] == 'truck' and abs(i['translation'][0]) < 12 and abs(i['translation'][1]) < 9]
        if len(objs) > 0:
            sid, ts = key.split('/')
            box_filtered = objs
            # box_filtered = [box for box in box_info if fbbox_filter_dist_class(box, draw_class_list, THRES_dist_to_ego)]
            box9d, box_class_names = [lfcjson_to_box9d(box) for box in box_filtered], [box['detection_name'] for box in
                                                                                     box_filtered]
            corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
            corners3d = transform_pts_with_T(corners, T_se).reshape(-1, 3)
            corners2d = camera_model.project_points(corners3d)
            im = cv2.imread(op.join(data_root, sid, f'camera/{sensor_id}/{ts}.jpg' ))
            im = draw_boxes(im, corners2d)

            if False:  # draw label
                for idx, (box, text) in enumerate(zip(corners2d.reshape(-1, 8), box_class_names)):
                    cv2.putText(im, f'{idx}_{text.split(".")[-1]}', tuple(box[:2].astype('int')), 1, 1,
                                (255, 0, 0))
            if True:
                pred_obj = preds[key]

                box_filtered = [i for i in pred_obj if i['detection_name'] == 'truck' ]
                # box_filtered = [box for box in box_info if fbbox_filter_dist_class(box, draw_class_list, THRES_dist_to_ego)]
                box9d, box_class_names = [lfcjson_to_box9d(box) for box in box_filtered], [box['detection_name'] for box in
                                                                                         box_filtered]
                corners = np.array([to_corners_9(np.array(i)) for i in box9d])  # ego_base -> camera
                corners3d = transform_pts_with_T(corners, T_se).reshape(-1, 3)
                corners2d = camera_model.project_points(corners3d)
                im = draw_boxes(im, corners2d, clr=(255, 128,0))


            save_path = f"/tmp/1234/lfcjson/{sensor_id}_{sid}_{ts}.jpg"
            P(save_path).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(save_path, im)


if __name__ == '__main__':
    json_path = "/home/yuanshiwei/4/prefusion/work_dirs/borui_dets_71/gt_pred_dumps/dets"
    data_root = '/ssd1/MV4D_12V3L'
    draw_pickle_lfcjson(json_path, data_root)
