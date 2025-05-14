import random
from collections import Counter
from itertools import cycle

import numpy as np
from easydict import EasyDict
from mmengine import Config
import pandas as pd
from mtv4d import read_txt, read_json, write_txt

import argparse

from mtv4d.utils.misc_base import mp_pool


def get_all_cfg_dictionary(cfg):
    output_class_mapping, output_attr_mapping = {}, {}
    cfg.dictionary_polygons.class_mapping = {i: i for i in cfg.dictionary_polygons.classes}
    cfg.dictionary_polygons.attr_mapping = {}
    cfg.dictionary_polylines.class_mapping = {i: i for i in cfg.dictionary_polylines.classes}
    cfg.dictionary_polylines.attr_mapping = {}
    for dictionary in [
        cfg.mapping_heading_objects,
        cfg.mapping_plane_heading_objects,
        cfg.mapping_no_heading_objects,
        cfg.mapping_square_objects,
        cfg.mapping_cylinder_objects,
        cfg.mapping_oriented_cylinder_objects,
        cfg.dictionary_polygons,
        cfg.dictionary_polylines,
    ]:
        if 'class_mapping' in dictionary.keys():
            for k, v in dictionary.class_mapping.items():
                v = v[0] if isinstance(v, list) else v
                if '::' not in v:
                    output_class_mapping[k] = v
                else:
                    output_class_mapping[k] = v.split('::')[0]
                    output_attr_mapping[k] = v.split('::')[1]
        if 'attr_mapping' in dictionary.keys():
            for k, v in dictionary.attr_mapping.items():
                output_attr_mapping[k] = v[0] if isinstance(v, list) else v
    for k, v in output_class_mapping.items():
        if isinstance(v, list):
            output_class_mapping[k] = v[0]
    for k, v in output_attr_mapping.items():
        if isinstance(v, list):
            output_class_mapping[k] = v[0]
    output_class_mapping['class.parking.parking_slot'] = 'class.parking.parking_slot'
    output = EasyDict()
    output['class_mapping'] = EasyDict(output_class_mapping)
    output['attr_mapping'] = EasyDict(output_attr_mapping)
    return output

def count_class_and_attr_occurrence(group_infos, class_dict_with_attr):
    class_mapping =  getattr(class_dict_with_attr, 'class_mapping', default={})
    attr_mapping = getattr(class_dict_with_attr, 'attr_mapping', default={})

    output, output_number = {}, {}
    for hash_key, frame_label in group_infos.items():
        cnt, cnt_number = Counter(), Counter()
        if '3d_boxes' in frame_label.keys():  # box
            classes = [ele['class'] for ele in frame_label['3d_boxes'] if
                       abs(ele['translation'][0]) < 12 and abs(ele['translation'][1]) < 9]
            attrs = [ele['attr'] for ele in frame_label['3d_boxes'] if
                     abs(ele['translation'][0]) < 12 and abs(ele['translation'][1]) < 9]
        else:  # poly
            classes = [ele['class'] for ele in frame_label['3d_polylines'] if
                       abs(ele['points'][:, 0].min()) < 12 and abs(ele['points'][:, 1].min()) < 9]
            attrs = [ele['attr'] for ele in frame_label['3d_polylines'] if
                     abs(ele['points'][:, 0].min()) < 12 and abs(ele['points'][:, 1].min()) < 9]
        classes = [reverse_class_mapping[i] for i in classes if i in reverse_class_mapping.keys()]
        attrs = [reverse_attr_mapping[j] for i in attrs for j in i.values() if j in reverse_attr_mapping.keys()]
        cnt.update(set(classes))
        cnt.update(set(attrs))
        cnt_number.update(classes)
        cnt_number.update(attrs)
        output[hash_key] = cnt
        output_number[hash_key] = cnt_number
    return output, output_number

def count_class_and_attr_occurrence_by_hashkey( class_dict_with_attr):
    class_mapping =  getattr(class_dict_with_attr, 'class_mapping', default={})
    attr_mapping = getattr(class_dict_with_attr, 'attr_mapping', default={})

    output, output_number = {}, {}
    for hash_key, frame_label in group_infos.items():
        cnt, cnt_number = Counter(), Counter()
        if '3d_boxes' in frame_label.keys():  # box
            classes = [ele['class'] for ele in frame_label['3d_boxes'] if
                       abs(ele['translation'][0]) < 12 and abs(ele['translation'][1]) < 9]
            attrs = [ele['attr'] for ele in frame_label['3d_boxes'] if
                     abs(ele['translation'][0]) < 12 and abs(ele['translation'][1]) < 9]
        else:  # poly
            classes = [ele['class'] for ele in frame_label['3d_polylines'] if
                       abs(ele['points'][:, 0].min()) < 12 and abs(ele['points'][:, 1].min()) < 9]
            attrs = [ele['attr'] for ele in frame_label['3d_polylines'] if
                     abs(ele['points'][:, 0].min()) < 12 and abs(ele['points'][:, 1].min()) < 9]
        classes = [reverse_class_mapping[i] for i in classes if i in reverse_class_mapping.keys()]
        attrs = [reverse_attr_mapping[j] for i in attrs for j in i.values() if j in reverse_attr_mapping.keys()]
        cnt.update(set(classes))
        cnt.update(set(attrs))
        cnt_number.update(classes)
        cnt_number.update(attrs)
        output[hash_key] = cnt
        output_number[hash_key] = cnt_number
    return output, output_number

def in_valid_range(ele, dist_range=[12, 9]):
    if ele['geometry_type'] == 'box3d':
        psr = ele['geometry']
        return abs(psr['pos_xyz'][0]) < dist_range[0] and abs(psr['pos_xyz'][1]) < dist_range[1]
    else:
        pts = np.array(ele['geometry'])
        return  np.abs(pts[:, 0]).min() < dist_range[0] and np.abs(pts[:, 1]).min() < dist_range[1]

def read_key(data):
    data_root, key, dict_with_attr, valid_range = data
    sid, ts = key.split('/')
    frame_label = read_json(f'{data_root}/{sid}/4d_anno_infos/4d_anno_infos_frame/frames_labels/{ts}.json')
    cnt, cnt_number = Counter(), Counter()
    classes = [ele['obj_type'] for ele in frame_label
               if ele['obj_type'] in dict_with_attr.class_mapping.values() and in_valid_range(ele, valid_range)]
    attrs = [[j for j in ele['obj_attr'].values() if j in dict_with_attr.attr_mapping.values() ]
             for ele in frame_label if in_valid_range(ele, valid_range)]
    cnt.update(set(classes))
    attr_set = []
    for attr in attrs:
        attr_set += attr
    cnt.update(set(attr_set))
    cnt_number.update(classes)
    cnt_number.update(attr_set)
    return cnt, cnt_number

def vanilla_resampling(gpdf, gpdf_number, frame_keys,sample_number, resampling=True):
    cnt_per_class = gpdf.sum(axis=0)
    # max_class = cnt_per_class.idxmax()
    # max_number = cnt_per_class[max_class]
    if not list(cnt_per_class): return []
    # total_numbers, total_class_numbers = len(frame_keys), len(cnt_per_class)
    number_target0 = 200
    sampled_groups = []
    # xianzou quede ,ranhou bu buquede
    output_list = set()
    # first pick
    for name in [i for i in list(gpdf.columns) if cnt_per_class[i] <= number_target0]:
        output_list.update(list(gpdf[gpdf[name] > 0].index))
        # sample_iterator = cycle(gpdf_number[gpdf_number[name] >= 1][name].sort_values(ascending=False).index)
        # sampled_groups += [frame_keys[next(sample_iterator)] for _ in
        #                    range(int(number_target - cnt_per_class[name]))]
    for name in [i for i in sorted(list(gpdf.columns), key=lambda i:cnt_per_class[i], reverse=False) if cnt_per_class[i] > number_target0]:
        target_number = number_target0
        if resampling:
            target_number = max(0, target_number - sum(gpdf.loc[list(output_list)][name]>0))
        if target_number > 0:
            picked_data_list = list(gpdf[gpdf[name] > 0].index)
            random.shuffle(picked_data_list)
            output_list.update(picked_data_list[:target_number])
    output_list = list(output_list)
    for name in list(gpdf.columns):
        # index = [i in index_list for i in range(len(target_list))]
        print(name, sum(gpdf.loc[output_list][name] >0) )
    print(len(output_list))
    return output_list


def generate_output_image_list(frame_keys, class_dict_with_attr, data_root, valid_range, sample_number, group_size=1, sampling_method='vanilla'):
    frame_infos = mp_pool(read_key, [(data_root, i, class_dict_with_attr, valid_range) for i in frame_keys])
    counted_groups_mapping = {k:i for (i, j), k in zip(frame_infos, frame_keys)}
    number_counted_groups_mapping = {k:j for (i, j), k in zip(frame_infos, frame_keys)}
    # frame_keys = list(counted_groups_mapping.keys())
    gpdf = pd.DataFrame(list(counted_groups_mapping.values())).fillna(0)
    gpdf_number = pd.DataFrame(list(number_counted_groups_mapping.values())).fillna(0)
    if sampling_method == "vanilla":
        sampled_groups = vanilla_resampling(gpdf, gpdf_number, frame_keys, sample_number=sample_number)
    else:
        raise NotImplementedError
    return [frame_keys[i] for i in sampled_groups]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', default='/ssd1/MV4D_12V3L/train_0402.txt')
    parser.add_argument('--config_file', default='contrib/fastray_planar/configs/lidar_template.py')
    parser.add_argument('--data_root', default='/ssd1/MV4D_12V3L')
    parser.add_argument('--sample_number', type=int, default=150)  #
    parser.add_argument('--save_txt_prefix', default='/ssd1/MV4D_12V3L/validation_indice')

    args = parser.parse_args()
    save_txt_path = f"{args.save_txt_prefix}_{args.sample_number}.txt"
    sample_number = args.sample_number
    cfg = Config.fromfile(args.config_file)
    dictionary = get_all_cfg_dictionary(cfg)
    key_list = read_txt(args.input_txt)
    sampled_groups = generate_output_image_list( key_list,  dictionary, args.data_root, valid_range=[12, 9],sample_number=sample_number, group_size=1)
    write_txt(sampled_groups, save_txt_path)
    if False:  # read pickle and write picked pickle
        pass  # TODO: read pickle and write picked pickle
        # use get_pickle_from_indice.py

    # cp /tmp/1234/validation_indice.txt /ssd1/MV4D_12V3L/valset_indice.txt