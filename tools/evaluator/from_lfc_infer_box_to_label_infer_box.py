from pathlib import Path as P
import argparse

from mtv4d.utils.box_base import to_list_psr_box_from_vec7_conf, box_vec7_to_psr
from mtv4d.utils.io_base import read_json
from mtv4d.utils.misc_base import mp_pool
from plotly.io import write_json
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')


def box2box(box):
    rotation = Rotation.from_matrix(np.array(box['rotation']).reshape(3, 3)).as_euler('XYZ')
    return {
        "position": {
            "x": box['translation'][0],
            "y": box['translation'][1],
            "z": box['translation'][2],
        },
        "scale": {
            "x": box['size'][0],
            "y": box['size'][1],
            "z": box['size'][2],
        },
        "rotation": {
            "x": rotation[0],
            "y": rotation[1],
            "z": rotation[2],
        },
    }


def lfc_box_to_label_box(box):
    return {"obj_attr": {}, "obj_type": box['class'],
            "psr": box_vec7_to_psr(box),
            "conf": box['score']}


def func(x):
    path, input_dir, output_dir = x
    rel_path = path.relative_to(input_dir)
    output_path = P(output_dir) / rel_path
    content = read_json(str(path))['pred']['bboxes']  # list of dictionary
    output_content = [lfc_box_to_label_box(i) for i in content]
    output_path.parent.mkdir(exist_ok=True, parents=True)
    write_json(output_content, str(output_path))


if __name__ == "__main__":
    # to get the same directory structure as the input dir
    args = parser.parse_args()
    a = sorted(P(args.input_dir).rglob('*.json'))
    mp_pool(func, [(i, args.input_dir, args.output_dir) for i in a])
