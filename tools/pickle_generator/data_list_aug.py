from collections import Counter
from itertools import cycle
from typing import *
from pathlib import Path as P
import numpy as np
import mmengine
from mmengine import Config

# pkl_path = cfg.train_loader.dataset.info_path
# a = mmengine.load(pkl_path)
# tongjishuju

# generate from the total list;
import pandas as pd
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_pkl_path', default=None)
parser.add_argument('--output_pkl_path', default='/tmp/1234/save_resampled_pkl.pkl')
parser.add_argument("--config_path", default="tools/evaluator/lidar.py")
parser.add_argument('--category_to_balanace', default="dictionary_polylines")
parser.add_argument('--save_txt_path', default="/tmp/1234/save_resampled_indice.txt")


# def count_class_and_attr_occurrence(info: Dict, groups: List):
#     for group in groups:
#         obj_cnt = Counter()
#         frm_cnt = Counter()
#         for index_info in group:
#             transformables = self.load_all_transformables(info, index_info)  # we need to load all transformables
#             no_objects_found = True
#             for _, transformable in transformables.items():
#                 classes = [ele['class'] for ele in transformable.elements]
#                 attrs = [attr_val for ele in transformable.elements for attr_val in ele['attr']]
#
#                 if classes:
#                     obj_cnt.update(classes)
#                     frm_cnt.update(set(classes))
#                     no_objects_found = False
#
#                 if oversampling_consider_object_attr:
#                     obj_cnt.update(attrs)
#                     frm_cnt.update(set(attrs))
#                     no_objects_found = False
#                 # maybe no object is not needed at all
#
#         group.obj_cnt = obj_cnt
#         group.frm_cnt = frm_cnt
#         group.grp_cnt = Counter(list(frm_cnt.keys()))
#         group.cnt = self._decide_counter(group)
#
#     return groups


def count_class_and_attr_occurrence(group_infos, class_dict_with_attr, is_box=True):
    # group_infos = [i for i in group_infos]
    class_mapping = class_dict_with_attr.class_mapping
    try:
        attr_mapping = class_dict_with_attr.attr_mapping
    except Exception:
        attr_mapping = {}
    if is_box:
        reverse_class_mapping = {v[0]: k for k, v in class_mapping.items()}
        reverse_attr_mapping = {v[0]: k for k, v in attr_mapping.items()}
    else:
        reverse_class_mapping = {v: k for k, v in class_mapping.items()}
        reverse_attr_mapping = {v: k for k, v in attr_mapping.items()}
    output, output_number = {}, {}
    for hash_key, frame_label in tqdm(group_infos.items()):
        cnt, cnt_number = Counter(), Counter()
        if is_box:  # box
            classes = [ele['class'] for ele in frame_label['3d_boxes'] if abs(ele['translation'][0]) < 12 and abs(ele['translation'][1]) < 9 ]
            attrs = [ele['attr'] for ele in frame_label['3d_boxes'] if abs(ele['translation'][0]) < 12 and abs(ele['translation'][1]) < 9 ]
        else:  # poly
            classes = [ele['class'] for ele in frame_label['3d_polylines'] if abs(ele['points'][:, 0].min()) < 12 and abs(ele['points'][:, 1].min()) < 9]
            attrs = [ele['attr'] for ele in frame_label['3d_polylines'] if abs(ele['points'][:, 0].min()) < 12 and abs(ele['points'][:, 1].min()) < 9]
        classes = [reverse_class_mapping[i] for i in classes if i in reverse_class_mapping.keys()]
        attrs = [reverse_attr_mapping[j] for i in attrs for j in i.values() if j in reverse_attr_mapping.keys()]
        cnt.update(set(classes))
        cnt.update(set(attrs))
        cnt_number.update(classes)
        cnt_number.update(attrs)
        output[hash_key] = cnt
        output_number[hash_key] = cnt_number
    return output, output_number


def cbgs_resampling(gpdf, frame_keys):
    return sampled_groups


def vanilla_resampling(gpdf, gpdf_number, frame_keys):
    cnt_per_class = gpdf.sum(axis=0)
    # max_class = cnt_per_class.idxmax()
    # max_number = cnt_per_class[max_class]
    if not list(cnt_per_class): return []
    total_numbers, total_class_numbers = len(frame_keys), len(cnt_per_class)
    number_target = int(total_numbers / total_class_numbers / 2)  # total huozhe max_number/10
    sampled_groups = []
    for name in list(gpdf.columns):
        print(name, cnt_per_class[name], number_target)
        if cnt_per_class[name] < number_target:
            sample_iterator = cycle(gpdf_number[gpdf_number[name] >= 1][name].sort_values(ascending=False).index)
            sampled_groups += [frame_keys[next(sample_iterator)] for _ in range(int(number_target - cnt_per_class[name]))]
    return sampled_groups


def generate_output_image_list(frame_infos, class_dict_with_attr, group_size, is_box, sampling_method='vanilla'):
    counted_groups_mapping, number_counted_groups_mapping = count_class_and_attr_occurrence(frame_infos,
                                                                                            class_dict_with_attr, is_box=is_box)
    frame_keys = list(counted_groups_mapping.keys())
    gpdf = pd.DataFrame(list(counted_groups_mapping.values())).fillna(0)
    gpdf_number = pd.DataFrame(list(number_counted_groups_mapping.values())).fillna(0)
    if sampling_method == "vanilla":
        sampled_groups = vanilla_resampling(gpdf, gpdf_number, frame_keys)
    elif sampling_method == "cbgs":
        sampled_groups = cbgs_resampling(gpdf, frame_keys)
    return sampled_groups


def get_save_output_scene_id(scene_id, frame_id, output_scenes):
    for i in range(10000):  # max duplication num is 10000
        save_scene_name = f"{scene_id}_{i}"
        if save_scene_name in output_scenes.keys():
            if frame_id not in output_scenes[save_scene_name]:
                return save_scene_name
        else:
            return save_scene_name
    return save_scene_name


def sampled_groups_to_save_scene_id_mapping(sampled_groups):
    output_scenes = {}  # allocate different_keys to different scenes

    for i in sampled_groups:  # allocate different resampled data to the
        scene_id, frame_id = i.split('/')  # use a /b as the simple hash key
        scene_id = get_save_output_scene_id(scene_id, frame_id, output_scenes)
        output_scenes.setdefault(scene_id, []).append(frame_id)
    for k, v in output_scenes.items():
        output_scenes[k] = sorted(v)
    return output_scenes


def save_sampled_groups_to_pkl(sampled_groups, input_infos, save_path):
    output_scenes = sampled_groups_to_save_scene_id_mapping(sampled_groups)
    # so that we get the scene_id


def save_sampled_groups_indice_to_txt(sampled_groups, frame_infos, output_txt_path):
    output_indices = [i for i in frame_infos.keys()]
    print('before sampling', len(output_indices))
    output_indices += sampled_groups
    print('after sampling', len(output_indices))
    P(output_txt_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_txt_path, 'w') as f:
        f.writelines([i + '\n' for i in output_indices])


def get_frame_info_from_pkl_format_data(args):
    input_infos = mmengine.load(args.input_pkl_path)
    output_dictionary = {}  # maybe i should use the dictionary ds; which means ordered dictionary
    for scene_name, v in input_infos.items():
        for ts, frame_info in v['frame_info'].items():
            frame_hash_key = f"{scene_name}/{ts}"
            output_dictionary[frame_hash_key] = {
                '3d_boxes': frame_info['3d_boxes'],
                '3d_polylines': frame_info['3d_polylines']
            }
    return output_dictionary


def main():
    args = parser.parse_args()
    # args.input_pkl_path = '/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20230823_110018.pkl'
    args.input_pkl_path = '/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_train_fix_label_error.pkl'
    config_file = "contrib/fastray_planar/configs/lidar_planeheading.py"
    cfg = Config.fromfile(config_file)
    frame_infos = get_frame_info_from_pkl_format_data(args)  # a dict
    # can use dictionary list
    sampled_groups = []
    if True:  # cbgs

        cfg.dictionary_polygons.class_mapping = {i:i for i in cfg.dictionary_polygons.classes}
        cfg.dictionary_polygons.attr_mapping ={}
        cfg.dictionary_polylines.class_mapping = {i:i for i in cfg.dictionary_polylines.classes}
        cfg.dictionary_polylines.attr_mapping ={}
        for dictionary in [
            cfg.mapping_heading_objects,
            cfg.mapping_plane_heading_objects,
            cfg.mapping_no_heading_objects,
            cfg.mapping_square_objects,
            cfg.mapping_cylinder_objects,
            cfg.mapping_oriented_cylinder_objects,
            cfg.dictionary_polygons,
            cfg.dictionary_polylines,
        ]:  # different dictionary; difference between poly & box dictionary
            sampled_groups += generate_output_image_list(
                frame_infos, group_size=1, class_dict_with_attr=dictionary, is_box='classes' not in dictionary.keys())  # we only need the hash key of the groups
        save_sampled_groups_indice_to_txt(sampled_groups, frame_infos, args.save_txt_path)


if __name__ == "__main__":
    main()
