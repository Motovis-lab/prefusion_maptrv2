import json
from collections import defaultdict
from pathlib import Path as P
from tkinter import NO
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


def precess_single_box(key, box, mode, mapping_label_to_pred_label={}):
    output = EasyDict(
        sample_token=key,
        translation=box['translation'],
        size=box['size'],
        rotation=rotation_to_wxyz(box['rotation']),
        velocity=box['velocity'][:2] if 'velocity' in box.keys() else None,
        num_pts=100,
        detection_name=box['class'].split('.')[-1],
        attribute_name=box['class'].split('.')[-1],
    )
    output['detection_score'] = box['score'] if mode == 'pred' else 1
    return output


def precess_single_slot(key, poly, mode, mapping_label_to_pred_label={}):
    output = EasyDict(
        sample_token=key,
        points=np.array(poly).reshape(-1, 4)[:, :3],
        detection_score=np.array(poly).reshape(-1, 4)[:, 3],
        detection_name='class.parking.parking_slot',
    )
    return output


def process_single_box_gt(ts, box, mapping_label_to_pred_label={}):
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


def precess_single_box_pred(ts, box):
    return EasyDict(
        sample_token=ts,
        translation=box['translation'],
        size=box['size'],
        rotation=rotation_to_wxyz(box['rotation']),  # nuscenes shi wxyz, better transform to that format
        velocity=box['velocity'][:2],
        # ego_translation=box[''],
        num_pts=100,
        detection_name=box['class'],
        detection_score=box['score'],
        attribute_name=box['class']
    )


def rotation_to_wxyz(rot):
    R = np.array(rot).reshape(3, 3)
    return Rotation.from_matrix(R).as_quat(scalar_first=True).tolist()  # wxyz, default False


def process_data_to_coco(ts, data, mode='pred', obj_type='bboxes'):
    if obj_type == 'bboxes':
        return [precess_single_box(ts, box, mode) for box in data[mode][obj_type]]
    elif obj_type == 'slots':
        return [precess_single_slot(ts, box, mode) for box in data[mode][obj_type]]
    elif obj_type == 'cylinders':
        raise NotImplementedError
    else:
        raise NotImplementedError


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


def load_jsons_from_single_hooks(input_dir, mode, obj_type, output_json_path="/tmp/1234/pred.json"):
    # sensor, calib_path = "camera8", "/ssd1/MV4D_12V3L/20230823_110018/calibration_center.yml"
    if mode == 'pred':
        a = [i for i in sorted(P(input_dir).rglob('*.json')) if '_gt.json' not in str(i)]
    elif mode == 'gt':
        a = sorted(P(input_dir).rglob('*_gt.json'))
    else:
        raise NotImplementedError
    output, output_list = {}, []
    for path in a:
        sid = path.parent.name
        ts = path.stem.strip('_gt')
        key = f'{sid}/{ts}'
        data = read_json(str(path))
        output[key] = process_data_to_coco(key, data, mode, obj_type=obj_type)
        output_list += output[key]
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


def cal_ap_list(tp_list, fp_list, conf_list, num_ap_interval=101):
    # tp_list means whether each sample is true in positive pred samples.
    npos = len(tp_list)
    tp = np.cumsum(tp_list).astype(float)
    fp = np.cumsum(fp_list).astype(float)
    conf = np.array(conf_list)
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, num_ap_interval)  # 101 steps, from 0% to 100% recall.
    precision = np.interp(rec_interp, rec, prec, right=0)
    confidence = np.interp(rec_interp, rec, conf, right=0)
    recall = rec_interp
    return precision, recall, confidence


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


def filter_obj_according_to_confidence(objs, configs, obj_type):
    if obj_type == 'bboxes':
        conf = configs.conf_thres
        for k, v in objs.items():
            objs[k] = [box for box in v if
                       box.detection_name in conf.keys() and box.detection_score >= conf[box.detection_name]]
    elif obj_type == 'slots':
        conf = configs.conf_thres_slot
        for k, v in objs.items():
            objs = [slot for slot in v
                    if np.mean(slot.detection_score) > conf[slot.detection_name]
                    ]
    return objs


def filter_obj_according_to_range(objs, configs, obj_type):
    if obj_type == 'bboxes':
        conf = configs.class_range
        for k, v in objs.items():
            objs[k] = [box for box in v if
                       box.detection_name in conf.keys() and
                       abs(box.translation[0]) <= conf[box.detection_name][0] and abs(box.translation[1]) <=
                       conf[box.detection_name][1]
                       ]
    elif obj_type == 'slots':
        conf = configs.class_range_slot
        # -2, 3
        for k, v in objs.items():
            if len(v) > 0:
                objs[k] = [slot for slot in v if
                           (
                                   (np.abs(slot.points[:, 0]) < conf[slot.detection_name][0]) * (
                                   np.abs(slot.points[:, 1]) < conf[slot.detection_name][1]) *
                                   (slot.points[:, 2] > conf[slot.detection_name][2]) * (
                                           slot.points[:, 2] < conf[slot.detection_name][3])
                           ).sum() >= 2
                           ]
    else:  # such as cylinders
        raise NotImplementedError
    return objs


def reconcate_boxes(boxes):
    output = []
    for k, v in boxes.items():
        output += v
    return output


def is_box_match(pred, gt, thres_dist, thres_direction=15 / 180 * np.pi, thres_iou=0.5):
    if thres_dist is None:
        thres_dist = max(gt.size) / 2
    if center_distance(pred, gt) > thres_dist: return False
    # if yaw_diff(pred, gt) > thres_direction: return False
    # if scale_iou(pred, gt) < thres_iou: return False
    return True


def is_slot_match(pred, gt, thres_dist=0.5):
    slot1 = np.array(pred.points).reshape(-1, 3)[:, :2]
    slot2 = np.array(gt.points).reshape(-1, 3)[:, :2]
    return min(np.linalg.norm(slot1 - slot2[[0, 1, 2, 3]], axis=1).sum(),
               np.linalg.norm(slot1 - slot2[[1, 2, 3, 0]], axis=1).sum(),
               np.linalg.norm(slot1 - slot2[[2, 3, 0, 1]], axis=1).sum(),
               np.linalg.norm(slot1 - slot2[[3, 0, 1, 2]], axis=1).sum()
               ) / 4 < thres_dist


def my_match_rate_box(A, B, A_list, B_list, class_name, thres_dist):
    A_cls = [i for i in A_list if i['detection_name'] == class_name]
    total_num = 0
    if len(A_cls) == 0:
        return None
    output_match = []
    error_list_dict = defaultdict(list)
    for key, A_frame in A.items():
        sid, ts = key.split('/')
        A_frame = [i for i in A_frame if i['detection_name'] == class_name]
        if len(A_frame) == 0:
            continue
        A_match_num = 0
        sample_token = A_frame[0]['sample_token']
        box2_list = [j for j in B[sample_token] if j['detection_name'] == class_name]
        for box1 in A_frame:
            for idx, box2 in enumerate(box2_list):
                if is_box_match(box1, box2, thres_dist):
                    A_match_num += 1
                    total_num += 1
                    error_list_dict["center_distance"] += [center_distance(box1, box2)]
                    error_list_dict["yaw_diff"] += [yaw_diff(box1, box2)]
                    error_list_dict["scale_iou"] += [scale_iou(box1, box2)]
                    del box2_list[idx]  # bufanghui bijiao
                    continue
        output_match += [A_match_num / len(A_frame)]
    print(class_name, len(A_cls), "average", np.mean(output_match), "total", total_num / len(A_cls),
          "center_distance", np.mean(error_list_dict["center_distance"]),
          "yaw_diff", np.mean(error_list_dict["yaw_diff"]),
          "scale_iou", np.mean(error_list_dict["scale_iou"]),
          )
    return np.mean(output_match)


def entrance_and_direction_difference(slot1, slot2):
    s1 = slot1.points.reshape(-1, 3)[:, :2]
    s2 = slot2.points.reshape(-1, 3)[:, :2]
    a = np.zeros(4)
    a[0] = np.linalg.norm(s1 - s2[[0, 1, 2, 3]], axis=1).sum() / 4
    a[1] = np.linalg.norm(s1 - s2[[1, 2, 3, 0]], axis=1).sum() / 4
    a[2] = np.linalg.norm(s1 - s2[[2, 3, 0, 1]], axis=1).sum() / 4
    a[3] = np.linalg.norm(s1 - s2[[3, 0, 1, 2]], axis=1).sum() / 4
    idx = np.argmin(a)
    d1 = (s1[0] - s1[3]) / np.linalg.norm(s1[0] - s1[3]) + (s1[1] - s1[2]) / np.linalg.norm(s1[1] - s1[2])
    d2 = (s2[idx] - s2[(idx + 3) % 4]) / np.linalg.norm(s2[idx] - s2[(idx + 3) % 4]) + (
            s2[(idx + 1) % 4] - s2[(idx + 2) % 4]) / np.linalg.norm(s2[(idx + 1) % 4] - s2[(idx + 2) % 4])
    d1, d2 = d1 / 2, d2 / 2
    dir_cos_dist = d1 @ d2
    return np.linalg.norm(s1[:2] - s2[[idx, (idx + 1) % 4]], axis=1).sum() / 4, np.rad2deg(
        np.arccos(np.clip(dir_cos_dist, -1, 1)))


def my_match_rate_slot(A, B, A_list, B_list, class_name, thres_dist):
    A_cls = A_list
    total_num = 0
    if len(A_cls) == 0:
        return None
    output_match = []
    error_list_dict = defaultdict(list)
    for key, A_frame in A.items():
        sid, ts = key.split('/')
        A_frame = [i for i in A_frame if i['detection_name'] == class_name]
        if len(A_frame) == 0:
            continue
        A_match_num = 0
        sample_token = A_frame[0]['sample_token']
        box2_list = [j for j in B[sample_token] if j['detection_name'] == class_name]
        for box1 in A_frame:
            for idx, box2 in enumerate(box2_list):
                if is_slot_match(box1, box2, thres_dist):
                    A_match_num += 1
                    total_num += 1
                    entrance_distance, direction_difference = entrance_and_direction_difference(box1, box2)
                    error_list_dict["entrance_distance"] += [entrance_distance]
                    error_list_dict["direction_difference"] += [direction_difference]
                    del box2_list[idx]
                    continue
        output_match += [A_match_num / len(A_frame)]
    print(class_name, len(A_cls), "average", np.mean(output_match), "total", total_num / len(A_cls),
          "entrance_distance", np.mean(error_list_dict["entrance_distance"]),
          "direction_difference", np.mean(error_list_dict["direction_difference"]),
          )
    return np.mean(output_match)


def my_cal_recall(gts, preds, gts_list, preds_list, class_name, obj_type, thres_dist):
    if obj_type == 'bboxes':
        return my_match_rate_box(gts, preds, gts_list, preds_list, class_name, thres_dist=thres_dist)
    elif obj_type == "slots":
        return my_match_rate_slot(gts, preds, gts_list, preds_list, class_name, thres_dist=thres_dist)


def my_cal_precision(gts, preds, gts_list, preds_list, class_name, obj_type, thres_dist):
    if obj_type == 'bboxes':
        return my_match_rate_box(preds, gts, preds_list, gts_list, class_name, thres_dist=thres_dist)
    elif obj_type == 'slots':
        return my_match_rate_slot(preds, gts, preds_list, gts_list, class_name, thres_dist=thres_dist)


def main_map_from_json_dirs_slot(pred_dir, gt_dir, cfg_path="tools/evaluator/config.json"):
    obj_type = 'slots'
    eval_detection_configs = load_cfg(cfg_path=cfg_path)
    preds, preds_list = load_jsons_from_single_hooks(pred_dir, 'pred', obj_type=obj_type,
                                                     output_json_path="/tmp/1234/pred.json")
    gts, gts_list = load_jsons_from_single_hooks(gt_dir, 'gt', obj_type='slots', output_json_path="/tmp/1234/gt.json")
    if True:  # after this, reorganize the pred list
        filter_obj_according_to_range(preds, eval_detection_configs, obj_type=obj_type)
        filter_obj_according_to_range(gts, eval_detection_configs, obj_type=obj_type)  # has already filtered
    if True:
        filter_obj_according_to_confidence(preds, eval_detection_configs, obj_type=obj_type)
    gts_list, preds_list = reconcate_boxes(gts), reconcate_boxes(preds)
    if False:  # draw im
        draw_json(input_dir, output_json_path="/tmp/1234/pred.json")
    ### calculate recall rate

    recall_dict = {}
    print('recall ===============')
    for class_name in eval_detection_configs.class_range_slot.keys():  # type: ignore
        recall = my_cal_recall(gts, preds, gts_list, preds_list, class_name, obj_type, thres_dist=0.5)
        if recall is not None:
            recall_dict[class_name] = recall

    # for k, v in recall_dict.items():
    #     print(k, v)
    ### calculate precision rate
    precision_dict = {}
    print('precision ===============')
    for class_name in eval_detection_configs.class_range_slot.keys():  # type: ignore
        precision = my_cal_precision(gts, preds, gts_list, preds_list, class_name, obj_type, thres_dist=0.5)
        if precision is not None:
            precision_dict[class_name] = precision
    # for k, v in precision_dict.items():
    #     print(k, v)


def main_map_from_json_dirs(pred_dir, gt_dir, cfg_path="tools/evaluator/config.json"):
    obj_type = 'bboxes'
    eval_detection_configs = load_cfg(cfg_path=cfg_path)
    preds, preds_list = load_jsons_from_single_hooks(pred_dir, 'pred', obj_type=obj_type,
                                                     output_json_path="/tmp/1234/pred.json")
    gts, gts_list = load_jsons_from_single_hooks(gt_dir, 'gt', obj_type=obj_type,
                                                 output_json_path="/tmp/1234/gt.json")
    if True:  # after this, reorganize the pred list
        filter_obj_according_to_range(preds, eval_detection_configs, obj_type=obj_type)
        filter_obj_according_to_range(gts, eval_detection_configs, obj_type=obj_type)
    if True:
        filter_obj_according_to_confidence(preds,eval_detection_configs, obj_type=obj_type)
    gts_list, preds_list = reconcate_boxes(gts), reconcate_boxes(preds)
    if False:  # draw im
        draw_json(input_dir, output_json_path="/tmp/1234/pred.json")
    ### calculate recall rate

    recall_dict = {}
    print('recall ===============')
    for class_name in eval_detection_configs.class_range.keys():  # type: ignore
        recall = my_cal_recall(gts, preds, gts_list, preds_list, class_name, obj_type=obj_type, thres_dist=None)
        if recall is not None:
            recall_dict[class_name] = recall

    # for k, v in recall_dict.items():
    #     print(k, v)
    ### calculate precision rate
    precision_dict = {}
    print('precision ===============')
    for class_name in eval_detection_configs.class_range.keys():  # type: ignore
        precision = my_cal_precision(gts, preds, gts_list, preds_list, class_name,  obj_type=obj_type, thres_dist=None)
        if precision is not None:
            precision_dict[class_name] = precision
    # for k, v in precision_dict.items():
    #     print(k, v)
    print("-------------------")

    # class_name = "passenger_car"
    # pr = generate_ap(gts, preds, gts_list, preds_list, class_name, center_distance, dist_th=0.5)
    # ap = calculate_ap(pr, class_name, eval_detection_configs)
    map_list = []
    for pred_cls in eval_detection_configs.class_range.keys():  # type: ignore
        pr = generate_ap(gts, preds, gts_list, preds_list, pred_cls, center_distance, dist_th=0.5)
        if pr is None: continue
        ap = calculate_ap(pr, pred_cls, eval_detection_configs)
        print(pred_cls, ap)
        map_list += [ap]
    print('map', np.array(map_list).mean())
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_pkl_path",
                    default='/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20230823_110018_evaluate.pkl')
parser.add_argument("--input_json_dir",
                    default="/home/yuanshiwei/4/prefusion/apa_demo_dumps_20250218/dets/20230823_110018")
parser.add_argument("--cfg_path", default="tools/evaluator/config_mtv.json")

if __name__ == "__main__":
    args = parser.parse_args()
    input_pred_dir = args.input_json_dir
    pkl_path = args.input_pkl_path
    cfg_path = args.cfg_path
    if False:
        input_pred_dir = "eval_results/apa_result_json_0224/pred_dumps/dets/20230823_110018"
        pkl_path = "/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20230823_110018_infer.pkl"
        input_pred_dir = args.input_json_dir
        pkl_path = args.input_pkl_path
        main(input_pred_dir, pkl_path, cfg_path=cfg_path)
    if True:
        pred_dir = "/home/yuanshiwei/4/prefusion/work_dirs/borui_dets_71/gt_pred_dumps/dets"
        gt_dir = "/home/yuanshiwei/4/prefusion/work_dirs/borui_dets_71/gt_pred_dumps/dets"
        main_map_from_json_dirs(pred_dir, gt_dir, cfg_path=cfg_path)
        # main_map_from_json_dirs_slot(pred_dir, gt_dir, cfg_path=cfg_path)
