import argparse
from pathlib import Path

from copious.io.fs import parent_ensured_path
import torch


def main(args):
    a = torch.load(args.input_path)
    b = {"state_dict": a["state_dict"]}
    torch.save(b, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=Path, required=True)
    parser.add_argument('-o', '--output-path', type=parent_ensured_path, required=True)
    main(parser.parse_args())
