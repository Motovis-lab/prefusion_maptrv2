import json
from pathlib import Path as P
import cv2
import mmengine
from mtv4d.utils.box_base import to_corners_9
from mtv4d.utils.calib_base import read_cal_data
from mtv4d.utils.draw_base import draw_boxes, to_corners_7
import numpy as np
from mtv4d.utils.geo_base import transform_pts_with_T
from mtv4d.utils.io_base import read_json, read_pickle
from mtv4d.utils.sensors import get_camera_models
from nuscenes.eval.detection.algo import calc_ap
from scipy.spatial.transform import Rotation
from easydict import EasyDict
from sympy.stats.sampling.sample_scipy import scipy
from nuscenes.eval.detection.algo import calc_ap

from tmp_scripts.fix_pkl import Rt_to_T
from nuscenes.eval.common.utils import center_distance, velocity_l2, scale_iou, yaw_diff, attr_acc, cummean


def process_single_box_gt(ts, box):
    return EasyDict(
        sample_token=ts,
        translation=box['translation'],
        size=box['size'],
        rotation=rotation_to_wxyz(box['rotation']),
        velocity=box['velocity'][:2],
        # ego_translation=box[''],
        num_pts=100,
        detection_name=box['class'].split('.')[-1],
        # detection_score=box['score'],
        attribute_name=box['class'].split('.')[-1],
    )


def rotation_to_wxyz(rot):
    R = np.array(rot).reshape(3, 3)
    return Rotation.from_matrix(R).as_quat(scalar_first=True).tolist()  # wxyz, default False


def precess_single_box_pred(ts, box):
    return EasyDict(
        sample_token=ts,
        translation=box['translation'],
        size=box['size'],
        rotation=rotation_to_wxyz(box['rotation']),  # nuscenes shi wxyz, better transform to that format
        velocity=box['velocity'][:2],
        # ego_translation=box[''],
        num_pts=100,
        detection_name=box['class'].split('.')[-1],
        detection_score=box['score'],
        attribute_name=box['class'].split('.')[-1],
    )


def process_pred_data_to_coco(ts, data):
    return [precess_single_box_pred(ts, box) for box in data['pred']['bboxes']]


def process_gt_data_to_coco(ts, data):
    return [process_single_box_gt(ts, box) for box in data['3d_boxes']]


def input_dir(input_dir):
    output = []
    a = sorted(P(input_dir).glob('*.json'))
    for path in a:
        ts = path.stem
        with open(str(path)) as f:
            data = json.load(f)
        output += [process_pred_data_to_coco(ts, data)]
    return output


def draw_json_not_used_but_can_draw(input_dir, output_json_path="/tmp/1234/pred.json", draw_im=True):
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
    output, output_list = {}, []
    for path in a:
        ts = path.stem
        data = read_json(str(path))
        output[ts] = process_pred_data_to_coco(ts, data)
        output_list += output[ts]
    return output, output_list


def generate_ap(gt_boxes, pred_boxes, gts_list, preds_list, class_name, dist_fcn, dist_th=0.5, num_ap_interval=101):
    gt_class_mask = [1 for gt_box in gts_list if gt_box['detection_name'] == class_name]
    npos = len(gt_class_mask)
    if len(gt_class_mask) == 0:
        return None
    pred_boxes_list = [box for box in preds_list if box['detection_name'] == class_name]
    pred_confs = [box['detection_score'] for box in pred_boxes_list]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    taken = set()  # taken gt set
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box['sample_token']]):

            # Find closest match among ground truth boxes
            if gt_box['detection_name'] == class_name and not (pred_box['sample_token'], gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            # taken.add((preds_list['sample_token'], ['match_gt_idx']))
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box['sample_token']][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box['detection_score'])

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box['detection_score'])

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return None
    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, num_ap_interval)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp
    if False:
        import matplotlib.pyplot as plt
        plt.plot(rec_interp, prec), plt.plot(rec_interp, conf), plt.show()

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))
            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    return dict(recall=rec,
                precision=prec,
                confidence=conf,
                trans_err=match_data['trans_err'],
                vel_err=match_data['vel_err'],
                scale_err=match_data['scale_err'],
                orient_err=match_data['orient_err'],
                attr_err=match_data['attr_err'])


def load_pkl_to_gts(pkl_path, output_json_path='/tmp/poiu.json'):
    a = read_pickle(pkl_path)
    output, output_list = {}, []
    for scene, scene_info in a.items():
        scene_info = scene_info['frame_info']  # maybe token need to add scene_name
        for ts, boxes in scene_info.items():
            output[ts] = process_gt_data_to_coco(ts, boxes)
            output_list += output[ts]
    return output, output_list


def calculate_ap(metric_data, class_name, eval_detection_configs, dist_thres=None):
    ap = calc_ap(EasyDict(metric_data), eval_detection_configs.min_recall, eval_detection_configs.min_precision)
    return ap


def main(input_pred_dir, pkl_path, cfg_path="tools/evaluator/config.json"):
    eval_detection_configs = load_cfg(cfg_path=cfg_path)
    preds, preds_list = load_jsons_from_single_hooks(input_pred_dir, output_json_path="/tmp/1234/pred.json")
    gts, gts_list = load_pkl_to_gts(pkl_path, output_json_path="/tmp/1234/gt.json")
    # gts, gts_list = load_pkl_to_gts(input_pred_dir, output_json_path="/tmp/1234/pred.json")
    if False:
        filter_box_according_to_range(preds, eval_detection_configs)
        filter_box_according_to_range(gts, eval_detection_configs)
        filter_box_according_to_confidence(preds, eval_detection_configs)
        filter_box_according_to_confidence(gts, eval_detection_configs)
    if False:  # draw im
        draw_json(input_dir, output_json_path="/tmp/1234/pred.json")
    class_name = "passenger_car"
    pr = generate_ap(gts, preds, gts_list, preds_list, class_name, center_distance, dist_th=0.5)
    ap = calculate_ap(pr, class_name, eval_detection_configs)
    print(ap)
    # output_ap = []
    # for class_name in eval_detection_configs.class_names:
    #     for dist_th in eval_detection_configs.dist_ths:
    #         output_dist_th = generate_ap(gts, preds, gts_list, preds_list, class_name, dist_fcn, dist_th=dist_th)
    #         ap = calculate_ap(output_dist_th, dist_th, class_name)
    #         output_ap.append(ap)
    # print(np.mean(output_ap))


def load_cfg(cfg_path):
    with open(cfg_path) as f:
        data = json.load(f)
    return EasyDict(data)


if __name__ == "__main__":
    input_pred_dir = "/home/yuanshiwei/4/prefusion/apa_demo_dumps_20250218/dets/20230823_110018"
    pkl_path = '/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20230823_110018_evaluate.pkl'
    cfg_path = "tools/evaluator/config_mtv.json"
    main(input_pred_dir, pkl_path, cfg_path=cfg_path)
