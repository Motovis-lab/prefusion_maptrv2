import os

from tqdm import tqdm
from pathlib import Path as P

from mtv4d import write_txt, read_json



def is_slot_valid(polylines):
    for p in polylines:
        if p['obj_type'] == 'class.parking.parking_slot':
            if len(p['geometry']) != 4:
                return False
    return True


def filter_data(sid):
    scene_root = f'/ssd1/MV4D_12V3L/{i}'
    path = f'{scene_root}/4d_anno_infos/4d_anno_infos_frame/frames_labels'
    timestamps = [i.stem for i in P(path).glob('*')]
    # # filter lidar
    timestamps = [ts for ts in timestamps if
                  P(f'{scene_root}/lidar/undistort_static_merged_lidar1_model/{ts}.pcd').exists()]
    # # filter camera
    timestamps = [ts for ts in timestamps if
                  all([P(f'{scene_root}/camera/{cam}/{ts}.jpg').exists() for cam in [
                      'camera1', 'camera5', 'camera8', 'camera2', 'camera3', 'camera4', 'camera6', 'camera7',
                      'camera11', 'camera12', 'camera15',
                      "VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT",
                      "VCAMERA_FISHEYE_RIGHT", "VCAMERA_FISHEYE_BACK"
                  ]])]

    # filter occ
    timestamps = [times_id for times_id in timestamps if
                  all([
                      P(path).exists()
                      for path in [
                          f"{scene_root}/occ/occ_2d/occ_map_-15_-15_15_15/{times_id}.png",
                          f"{scene_root}/ground/ground_height_map_-15_-15_15_15/{times_id}.tif",
                          f"{scene_root}/occ/occ_2d/bev_height_map_-15_-15_15_15/{times_id}.png",
                          f"{scene_root}/occ/occ_2d/bev_lidar_mask_-15_-15_15_15/{times_id}.png",
                          f"{scene_root}/occ/occ_2d/occ_edge_height_map_-15_-15_15_15/{times_id}.png",
                          f"{scene_root}/occ/occ_2d/occ_edge_lidar_mask_-15_-15_15_15/{times_id}.png",
                          f"{scene_root}/occ/occ_2d/occ_map_sdf_-15_-15_15_15/{times_id}.png",
                          f"{scene_root}/occ/occ_2d/occ_edge_-15_-15_15_15/{times_id}.png",
                      ]
                  ])
                  ]
    # return [len(timestamps_lidar),len(timestamps_camera), len(timestamps_occ)]
    # if len(timestamps) < 2000:
    #     print(sid)
    # print(sid, len(timestamps), len(timestamps0)-len(timestamps))
    # timestamps = [ts for ts in timestamps if is_slot_ok(sid, ts)]
    timestamps = [ts for ts in timestamps if is_slot_valid( read_json(f'{scene_root}/4d_anno_infos/4d_anno_infos_frame/frames_labels/{ts}.json'))]
    return [f'{sid}/{ts}' for ts in timestamps]

# ids = sorted([i.name for i in P('/ssd1/MV4D_12V3L').glob('2023*')])
# ids = [
# "20231101_150226",
# "20231104_115532",
# "20231031_144111",
# ]

# ids = ['20231105_195823',
#        '20231104_123013']


old_pkl_ids = [
    "20230820_105813", "20230820_131402", "20230822_110856", "20230822_154430", "20230823_110018",
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
new_ids25 = [
    "20230823_113013", "20230824_172019", "20230826_133639", "20230828_124528", "20230901_110728", "20230901_120226",
    "20230903_175455", "20231011_202057", "20231028_142902", "20231028_170049", "20231029_123632", "20231029_161907",
    "20231103_123359", "20231104_123013", "20231105_143227", "20231105_195823", "20231106_124102", "20231107_124257",
    "20231107_185947", "20231107_205844", "20231107_220705", "20231108_144706", "20231108_155610", "20231108_170045",
    "20231108_204111",
]
new_ids13 = [
    "20230825_110210", "20230829_122655", "20230831_110634", "20231011_150326", "20231028_125529", "20231028_133437",
    "20231029_194228", "20231031_144111", "20231101_150226", "20231102_151151", "20231104_115532", "20231107_163857",
    "20231108_164010",
]
new_ids2025 = [
    "20250315_153606", "20250315_154251", "20250315_154856", "20250315_162419", "20250315_153359",
    "20250315_155248",
    "20250315_155653_1742025413764_1742025533764",
    "20250315_155653_1742025533764_1742025653764",
    "20250315_155653_1742025653764_1742025691764",
]
new_0416_16 = [
    "20250315_154730", "20250315_154940", "20250315_161940", "20250315_160135", "20250315_162031", "20250321_111631",
    "20250321_112847", "20250321_131221", "20250321_160904", "20250321_170901", "20250322_112035", "20250322_162647",
    "20250322_125640", "20250321_131754", "20250315_153742_1742024262664_1742024382664",
    "20250315_153742_1742024382664_1742024467964", "20250315_161540_1742026540764_1742026660764",
    "20250315_161540_1742026660764_1742026776664",
]

ids = old_pkl_ids + refactored_ids + new_ids25 + new_ids13 + new_ids2025 + new_0416_16

output = []
o1 = {}
# ids = ["20231103_123359"]
for _, i in enumerate(ids):
    output += filter_data(i)
    # o1[i] = filter_data(i)
# import torch
# torch.save(o1, '/tmp/1234/number.pth')

write_txt(output, '/ssd1/MV4D_12V3L/valid_indice_118.txt')






