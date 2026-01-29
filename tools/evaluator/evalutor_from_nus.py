import json
import time

from mtv4d.utils.misc_base import mp_pool
from nuscenes.eval.detection.algo import calc_ap
from tqdm import tqdm
import torch
from nuscenes import NuScenes
from nuscenes.eval.common.utils import center_distance, velocity_l2, scale_iou, yaw_diff, attr_acc, cummean
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from easydict import EasyDict

import numpy as np


def main1():
    p = '/home/yuanshiwei/2/stream3dppe/test/streampetr_3dppe_vov_flash_800_bs2_seq_24e_4x4_no_context_womv/Wed_Feb_19_16_02_51_2025/pts_bbox/metrics_details.json'
    with open(p) as f:
        data = json.load(f)
    q1 = []
    for class_name in eval_detection_configs.class_names:
        q = np.mean([calculate_ap(v, k) for k, v in data.items() if class_name in k])
        q1 += [q]
    print(np.mean(q1))


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


def calculate_ap(metric_data, class_name, dist_thres=None):
    # # for class_name in self.cfg.class_na mes:
    #     Compute APs.
    #     for dist_th in self.cfg.dist_ths:
    # metric_data = metric_data_list[(class_name, dist_th)]
    ap = calc_ap(EasyDict(metric_data), eval_detection_configs.min_recall, eval_detection_configs.min_precision)
    print(class_name, dist_thres, ap)
    return ap


def func(data):
    k, v = data
    return k, [EasyDict(i) for i in v]


def input_gt_pred_dicts_to_map():
    # gt, pred = torch.load('/tmp/1234_boxes.pth')  # ´óÔ¼10s loadÉÏÀ´ Õû¸öÊý¾Ý

    gt, pred = torch.load('/tmp/final_data_to_eval.pth')  # ´óÔ¼10s loadÉÏÀ´ Õû¸öÊý¾Ý
    print('loading finished')
    gts, preds = {}, {}
    gts_list, preds_list = [], []
    from time import time
    a = time()
    for k, v in gt.boxes.items():
        gts[k] = [
            EasyDict(
                sample_token=i.sample_token,
                translation=i.translation,
                size=i.size,
                rotation=i.rotation,
                velocity=i.velocity,
                ego_translation=i.ego_translation,
                num_pts=i.num_pts,
                detection_name=i.detection_name,
                detection_score=i.detection_score,
                attribute_name=i.attribute_name,
            )
            for i in v]
        gts_list += gts[k]

    for k, v in pred.boxes.items():
        preds[k] = [
            EasyDict(
                sample_token=i.sample_token,
                translation=i.translation,
                size=i.size,
                rotation=i.rotation,
                velocity=i.velocity,
                ego_translation=i.ego_translation,
                num_pts=i.num_pts,
                detection_name=i.detection_name,
                detection_score=i.detection_score,
                attribute_name=i.attribute_name,
            )
            for i in v]
        preds_list += preds[k]
    # func(list(gts.items())[0])
    # exit()
    gts = mp_pool(func, list(gts.items()), 16)
    preds = mp_pool(func, list(preds.items()), 16)
    # preds_list, gts_list = [], []
    # # [gts_list.append(*v) for k, v in gts]
    # # [preds_list.append(*v) for k, v in preds]
    # gts = {k: v for k, v in gts}
    # preds = {k: v for k, v in preds}
    print(time() - a)
    # # exit()
    # exit()
    print('transform to easydict finished')
    output_ap = []
    for class_name in eval_detection_configs.class_names:
        for dist_th in eval_detection_configs.dist_ths:
            output_dist_th = generate_ap(gts, preds, gts_list, preds_list, class_name, dist_fcn, dist_th=dist_th)
            ap = calculate_ap(output_dist_th, dist_th, class_name)
            output_ap.append(ap)
    print(np.mean(output_ap))


if __name__ == "__main__":
    import os

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['BLIS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['PYTORCH_THREADS'] = '1'
    os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
    output_dir = '/tmp/1234/a'
    eval_detection_configs = config_factory('detection_cvpr_2019')
    dist_fcn = center_distance
    input_gt_pred_dicts_to_map()
    # main1()
