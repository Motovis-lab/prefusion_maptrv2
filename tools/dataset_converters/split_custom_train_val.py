import mmengine 
from pathlib import Path as P
import random   
import copy

scene_root = "/home/wuhan/mv_4d_data"
scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("mv_4d_infos_2023*") if p.is_file()]


all_train_data = dict()
all_val_data = dict()
for scene_name in scene_names:
    data = mmengine.load(P(scene_root) / P(scene_name))
    name = scene_name[12:-4]
    frame_ids = list(data[name]['frame_info'].keys())
    val_frame_ids = random.sample(frame_ids, int(len(frame_ids) * 0.1))
    frame_infos = data[name].pop('frame_info')
    sub_val_frame_info = dict()
    sub_train_frame_info = dict()
    for frame_id in frame_infos:
        if frame_id in val_frame_ids:
            sub_val_frame_info.update({frame_id: frame_infos[frame_id]})
        else:
            sub_train_frame_info.update({frame_id: frame_infos[frame_id]})
    train_data = copy.deepcopy(data)
    val_data = copy.deepcopy(data)

    train_data[name].update({"frame_info": sub_train_frame_info})
    val_data[name].update({"frame_info": sub_val_frame_info})
    
    all_train_data.update(train_data)
    all_val_data.update(val_data)

mmengine.dump(all_train_data, "/home/wuhan/mv_4d_data/mv_4d_infos_train_custom.pkl")
mmengine.dump(all_val_data, "/home/wuhan/mv_4d_data/mv_4d_infos_val_custom.pkl")