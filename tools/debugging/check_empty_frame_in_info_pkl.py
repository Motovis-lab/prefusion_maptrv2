import argparse
import pickle
from pathlib import Path

from loguru import logger
from copious.io.args import declare_vars_as_global, g

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-info-pkl-path', type=Path, required=True, help='Input file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def main(args):
    with open(args.input_info_pkl_path, "rb") as f:
        data = pickle.load(f)
    for scene_id, scene_data in data.items():
        for frame_id, frame_data in scene_data["frame_info"].items():
            if "3d_boxes" in frame_data and len(frame_data["3d_boxes"]) == 0:
                logger.info(f"{scene_id=}, {frame_id=}")


if __name__ == '__main__':
    args = parse_args()
    declare_vars_as_global(verbose=args.verbose)
    main(args)
