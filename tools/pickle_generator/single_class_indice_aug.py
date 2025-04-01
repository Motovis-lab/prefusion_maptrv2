from pathlib import Path as P

from mtv4d import write_txt, read_json, read_txt
import numpy as np
from mtv4d.utils.misc_base import mp_pool
from tqdm import tqdm

old_pkl_ids = [
    "20230820_105813", "20230820_131402", "20230822_110856", "20230822_154430", "20230823_110018", "20230823_110018",
    "20230823_162939", "20230824_115840", "20230824_134824", "20230826_102054", "20230826_122208", "20230829_170053",
    "20230830_120142", "20230830_181107", "20230831_101527", "20230831_151057", "20230901_123031", "20230901_152553",
    "20230902_142849", "20230903_123057", "20231010_141702", "20231027_185823", "20231028_124504", "20231028_134141",
    "20231028_150815", "20231028_185150", "20231031_133418", "20231031_134214", "20231031_135230", "20231031_145557",
    "20231101_172858", "20231102_144626", "20231103_133206", "20231103_140855", "20231103_173838", "20231104_155224",
    "20231105_130142", "20231105_161937", "20231107_123645", "20231107_150715", "20231107_152423", "20231107_154446",
    "20231107_183700", "20231107_212029", "20231108_143610", "20231108_153013", "20231109_143452"
]
refactored_ids = [
    "20231103_174738", "20231101_160337", "20231029_195612", "20231028_134843", "20231028_145730", "20230822_104856",
    "20230829_115909", "20230830_141232", "20230901_121703", "20231010_131855", "20231105_114621",
]
new_ids = [
    "20230823_113013", "20230824_172019", "20230826_133639", "20230828_124528", "20230901_110728", "20230901_120226",
    "20230903_175455", "20231011_202057", "20231028_142902", "20231028_170049", "20231029_123632", "20231029_161907",
    "20231103_123359", "20231104_123013", "20231105_143227", "20231105_195823", "20231106_124102", "20231107_124257",
    "20231107_185947", "20231107_205844", "20231107_220705",
    "20231108_144706", "20231108_155610", "20231108_170045", "20231108_204111",
]

new_ids_0331 = [
    "20230825_110210", "20230829_122655", "20230831_110634", "20231011_150326", "20231028_125529", "20231028_133437",
    "20231029_194228", "20231031_144111", "20231101_150226", "20231102_151151", "20231104_115532", "20231107_163857",
    "20231108_164010"
]

def is_all_item_file_exist(sid, ts, data_root='/ssd1/MV4D_12V3L', lidar=True, cam=True, occ=True):
    times_id = ts
    if lidar:
        if not P(f'{data_root}/{sid}/lidar/undistort_static_merged_lidar1_model/{ts}.pcd').exists():
            return False
    if cam:
        if not all([P(f'{data_root}/{sid}/camera/{cam}/{ts}.jpg').exists() for cam in [
            'camera1', 'camera5', 'camera8', 'camera2', 'camera3', 'camera4', 'camera6', 'camera7',
            'camera11', 'camera12', 'camera15',
            "VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT",
            "VCAMERA_FISHEYE_RIGHT", "VCAMERA_FISHEYE_BACK"
        ]]): return False
    if occ:
        if not all([
            P(path).exists()
            for path in [
                f"{data_root}/{sid}/occ/occ_2d/occ_map_-15_-15_15_15/{times_id}.png",
                f"{data_root}/{sid}/ground/ground_height_map_-15_-15_15_15/{times_id}.tif",
                f"{data_root}/{sid}/occ/occ_2d/bev_height_map_-15_-15_15_15/{times_id}.png",
                f"{data_root}/{sid}/occ/occ_2d/bev_lidar_mask_-15_-15_15_15/{times_id}.png",
                f"{data_root}/{sid}/occ/occ_2d/occ_edge_height_map_-15_-15_15_15/{times_id}.png",
                f"{data_root}/{sid}/occ/occ_2d/occ_edge_lidar_mask_-15_-15_15_15/{times_id}.png",
                f"{data_root}/{sid}/occ/occ_2d/occ_map_sdf_-15_-15_15_15/{times_id}.png",
                f"{data_root}/{sid}/occ/occ_2d/occ_edge_-15_-15_15_15/{times_id}.png",
            ]
        ]): return False
    return True


# to get aug list

def get_number_within_range(file_name, class_name, range=[9, 12]):
    a = read_json(str(file_name))
    num = 0
    for obj in a:
        if obj['obj_type'] == class_name:
            if obj['geometry_type'] == 'box3d':
                if abs(obj['geometry']['scale_xyz'][0]) < range[0] and abs(obj['geometry']['scale_xyz'][1]) < range[1]:
                    num+=1
            elif obj['geometry_type'] == 'polyline3d':
                poly = np.abs(np.array(obj['geometry']).reshape([-1, 3]))
                if poly[:, 0].min() < range[0] and poly[:, 1].min() <range[1]:
                    num += 1
    return num

def func(data):
    file_name, class_name = data
    file_name=P(file_name)
    sid = file_name.parent.parent.parent.parent.name
    ts = file_name.stem
    data_root = str(file_name.parent.parent.parent.parent.parent)
    if is_all_item_file_exist(sid, ts, data_root):
        num = get_number_within_range(file_name, class_name)
        if num > 0:
            return f'{sid}/{ts}'
        else:
            return None
    else:
        return None

def single_class_aug(class_name, sid_list, data_root, to_sort=True, indice_list=None):
    output_list = []
    if indice_list is not None:
        json_names = [f'{data_root}/{i.split("/")[0]}/4d_anno_infos/4d_anno_infos_frame/frames_labels/{i.split("/")[1]}.json' for i in read_txt(indice_list)]
        data_list = [(file_name, class_name) for file_name in json_names]
        result = mp_pool(func, data_list)
        output_list += [key for key in result if key is not None]
    else:
        for sid in sid_list:
            json_names = [i for i in P(f'{data_root}/{sid}/4d_anno_infos/4d_anno_infos_frame/frames_labels').glob('*.json')]
            data_list = [(file_name, class_name) for file_name in json_names]
            result = mp_pool(func, data_list)
            output_list += [key for key in result if key is not None]
            print(sid, len(output_list))
            # for data in [(file_name, class_name) for file_name in json_names]:
            #     a = func(data)
            #     if a is not None and a>0:
            #         output_list += [f'{sid}/{P(data[0]).stem}']
    return output_list


if __name__ == "__main__":
    scene_ids = old_pkl_ids + refactored_ids + new_ids + new_ids_0331
    # scene_ids = ['20230823_110018']
    data_root = '/ssd1/MV4D_12V3L/'
    indice_list = '/ssd1/MV4D_12V3L/valid_indice_train_96_fix.txt'
    a = single_class_aug('class.pedestrian.pedestrian', scene_ids, data_root, indice_list=indice_list)
    write_txt(a, f'{data_root}/ped_aug_96.txt')
    a = single_class_aug('class.traffic_facility.cone', scene_ids, data_root, indice_list=indice_list)
    write_txt(a, f'{data_root}/cone_aug_96.txt')
