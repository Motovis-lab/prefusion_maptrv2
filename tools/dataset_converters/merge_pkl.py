import mmengine 
from pathlib import Path as P
from argparse import ArgumentParser


parser = ArgumentParser(add_help=False)
parser.add_argument('task_name',
                    type=str)

parser.add_argument('scene_root',
                    type=str)
args = parser.parse_args()

def mv4d_merge_data(scene_root):
    # scene_root = "/home/wuhan/mv_4d_data"
    scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("mv_4d_infos_2023*") if p.is_file()]


    all_train_data = dict()
    all_val_data = dict()
    for scene_name in scene_names:
        data = mmengine.load(P(scene_root) / P(scene_name))
        name = scene_name[12:-4]
        if name not in ["20231101_150226", "20231031_144111", "20230824_153239", "20230828_134528", "20231104_115532", "20231028_145730", # 没有OK的 
                        "20230820_131402", "20230822_104856", "20231027_185823", "20231107_212029"]:                         # 缺失很多occ的 
            # Attention: look the gene_info_4d.py set camera_type
            for key in data[name]['scene_info']['calibration']:
                if "PERSPECTIVE" in key:
                    data[name]['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
                elif "FISHEYE" in key:
                    data[name]['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'
            all_train_data.update(data)
        else:
            for key in data[name]['scene_info']['calibration']:
                if "PERSPECTIVE" in key:
                    data[name]['scene_info']['calibration'][key]['camera_type'] = 'PerspectiveCamera'
                elif "FISHEYE" in key:
                    data[name]['scene_info']['calibration'][key]['camera_type'] = 'FisheyeCamera'
            all_val_data.update(data)


    mmengine.dump(all_train_data, f"{scene_root}/mv_4d_infos_train.pkl")
    mmengine.dump(all_val_data, f"{scene_root}/mv_4d_infos_val.pkl")

def pretrain_merge_data(scene_root):
    scene_names = [str(p).split('/')[-1] for p in P(scene_root).rglob("mv_4d_infos_2023*") if p.is_file()]

    all_train_data = dict()
    all_val_data = dict()
    for scene_name in scene_names:
        data = mmengine.load(P(scene_root) / P(scene_name))
        name = scene_name[12:-4]
        if name not in ["20230830_120142"]:
            all_train_data.update(data)
        else:
            all_val_data.update(data)

    mmengine.dump(all_train_data, f"{scene_root}/mv_4d_infos_pretrain_train.pkl")
    mmengine.dump(all_val_data, f"{scene_root}/mv_4d_infos_pretrain_val.pkl")


if __name__ == '__main__':
    task_name = args.task_name
    scene_root = args.scene_root
    eval(f"{task_name}_merge_data")(scene_root)
    