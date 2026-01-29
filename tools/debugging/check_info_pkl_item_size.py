import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from loguru import logger
from copious.data_structure.dict import defaultdict2dict

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-pkl-path', type=Path, required=True, help='Input file path')
    return parser.parse_args()


def get_size(obj):
    total = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set)):
        total += sum(get_size(i) for i in obj)
    elif isinstance(obj, dict):
        total += sum(get_size(obj[k]) for k in obj.keys())
    else:
        pass
    return total


def main(args):
    with open(args.input_pkl_path, 'rb') as f:
        p = defaultdict2dict(pickle.load(f))
    
    for scene_id, scene_data in p.items():
        logger.info(f"Evaluating object size for scene: {scene_id}")
        logger.info(f"{get_size(scene_data['scene_info'])=}")
        logger.info(f"{get_size(scene_data['meta_info'])=}")
        logger.info(f"{get_size(scene_data['frame_info'])=}")

        logger.info(f"Analyzing frame_info ({len(scene_data['frame_info'])} frames) ... ")
        frame_info_items = defaultdict(int)
        for frame_id, frame_data in  scene_data["frame_info"].items():
            for item_name, item_data in frame_data.items():
                frame_info_items[item_name] += get_size(item_data)
                
        frame_info_items = defaultdict2dict(frame_info_items)
        logger.info(f"{pprint(frame_info_items)=}")


if __name__ == '__main__':
    main(parse_args())
