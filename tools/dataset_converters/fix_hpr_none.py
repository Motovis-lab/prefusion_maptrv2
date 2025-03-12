import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import pdb
sys.path.append(".")
sys.path.append("/home/wuhan/mtv4d/scripts/")
sys.path.append("/home/wuhan/mtv4d/")
import open3d as o3d
import mmcv
import mmengine
import pickle
from rich.progress import track
import os.path as op
from collections import defaultdict
from copy import deepcopy
import numpy as np
import json
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from pathlib import Path as P
from typing import List, Tuple, Union
import torch
from mmengine import Config
from contrib.fastbev_det.models.utils.virtual_camera import render_image, PerspectiveCamera, FisheyeCamera, create_virtual_perspective_camera, \
    pcd_lidar_point, read_pcd_lidar, load_point_cloud, render_image_with_src_camera_points
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


from mtv4d.annos_4d.helper import (  # type: ignore
    check_whether_points_behind_camera,
    CameraParam,
    torch_pool,
    transform_points_ego_to_lidar,
    write_json_from_list,
    read_json_to_list,
    anno_box_to_7_values_box,
    get_times,
    get_sync_filename,
    read_cal_data,
    generate_lidar_mask,
)

from mtv4d.annos_4d.misc import read_ego_paths  # type: ignore
from scripts.generate_4d_frame_clean import generate_DS4d_from_4dMapJson, solve_ds_occlusion_sub_id, generate_4d_frame_json_data, read_ts_json # type: ignore

def generate_labels_scene_from_4dMapjson(scene_root, Twes):
    (
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    ) = generate_DS4d_from_4dMapJson(op.join(scene_root, "4d_anno_infos/annos.json"), Twes)  # load进来转成DS4D
    print('generate finish')
    (
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    ) = solve_ds_occlusion_sub_id(   
        dn_boxes_vis_dict_ts2id,
        dn_boxes_vis_dict_id2ts,
        map_boxes_vis_dict_ts2id,
        map_boxes_vis_dict_id2ts,
        map_polylines_vis_dict_ts2id,
        map_polylines_vis_dict_id2ts,
    )

    # generate frame info txt
    output_json_frame_dlist = generate_4d_frame_json_data(dn_boxes_vis_dict_ts2id, map_boxes_vis_dict_ts2id, map_polylines_vis_dict_ts2id)
    # for ts, frames_labels in tqdm(output_json_frame_dlist.items(), desc='generating frame json'):
    #     write_json_from_list(
    #         frames_labels, op.join(scene_root, f"4d_anno_infos/4d_anno_infos_frame/frames_labels_all/{int(ts)}.json"), format_float=True, indent=4
    #     )
    return output_json_frame_dlist

if __name__ == "__main__":
    scene_root = "/home/wuhan/mv_4d_data/"
    scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("2023*") if p.is_dir()]

    print("=="*60)
    print(scene_names)
    print("=="*60)
    unvals = dict()
    for scene_name in scene_names:
        pkl_data = mmengine.load(P(scene_root) / P("mv_4d_infos_"+scene_name+".pkl"))
        unval = []
        root = os.path.join(scene_root,  scene_name)
        with open(op.join(root, "4d_anno_infos/ts_full.json"), 'r') as f:
            data = json.load(f)
            for d in data:
                if None in d.values():
                    unval.append(str(int(d['lidar'])))
        unvals.update({scene_name: unval})    
        for unval_id in unval:
            try:
                pkl_data[scene_name]['frame_info'].pop(unval_id)
            except:
                print(f"no {unval_id}")
        mmengine.dump(pkl_data, P(scene_root) / P("mv_4d_infos_"+scene_name+".pkl"))
    mmengine.dump(unvals, scene_root+"unval.pkl")