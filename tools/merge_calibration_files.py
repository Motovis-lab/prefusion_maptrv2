import argparse
from pathlib import Path
import yaml

from loguru import logger
from easydict import EasyDict as edict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-calib-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-calib-file", type=Path, required=True)
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})

args = parse_arguments()


def main():
    out_calib = {"rig": {}}
    for calib_file in args.input_calib_files:
        update_calib_(calib_file, out_calib)
    write_calib(out_calib, args.output_calib_file)


def update_calib_(input_calib_path, out_calib):
    with open(input_calib_path, "r") as f:
        calib_data = yaml.load(f, Loader=yaml.FullLoader)
        out_calib["rig"].update({calib_data["sensor_id"]: calib_data})
    logger.info(f"Updated {calib_data['sensor_id']} calib")


def write_calib(calib, output_calib_path):
    with open(output_calib_path, "w") as f:
        yaml.dump(calib, f)
    logger.info(f"Successfully wrote calibration file to {output_calib_path}")


if __name__ == "__main__":
    main()
